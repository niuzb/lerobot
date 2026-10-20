"""Versioned, explicit physical measurements and coordinate conventions."""

import hashlib
import json
from pathlib import Path

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TRAIN = list(range(79))
VALIDATION = list(range(79, 88))
TASK = "Put the tape into the pen holder"


def digest(path):
    """Hash a file without modifying it."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(config):
    """Bind artifacts to the complete resolved scene configuration."""
    payload = dict(config)
    if "asset" in config:
        asset = Path(config["asset"])
        if not asset.is_file():
            raise ValueError(f"Missing pinned MuJoCo asset: {asset}")
        payload["asset_files_sha256"] = {
            str(p.relative_to(asset.parent)): digest(p)
            for p in [asset, *sorted((asset.parent / "assets").glob("*"))]
            if p.is_file()
        }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    """Write finite JSON, rejecting NaN rather than creating ambiguous reports."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_config(path):
    """Load the versioned scene and resolve paths relative to its config file."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Expected scene schema_version=1")
    if config.get("fps") != 30 or config.get("physics_hz") != 300:
        raise ValueError("This pipeline requires 30 Hz control / 300 Hz physics")
    if config.get("joint_names") != JOINTS:
        raise ValueError("Joint order must match the recorded SO-101 dataset")
    if config.get("train_episodes") != TRAIN or config.get("validation_episodes") != VALIDATION:
        raise ValueError("Preserve the source 79/9 episode split")
    for key in ("asset", "dataset_root"):
        path_value = Path(config[key]).expanduser()
        config[key] = str((Path(path).resolve().parent / path_value).resolve())
    return config


def validate_physics(config):
    """Reject missing measurements rather than invent a calibrated environment."""
    if not config.get("measurements", {}).get("source"):
        raise ValueError("measurements.source must identify the physical measurement record")
    for obj, keys in {
        "tape": ("outer_radius_m", "inner_radius_m", "height_m", "mass_kg"),
        "holder": ("inner_radius_m", "wall_m", "height_m", "bottom_m", "mass_kg"),
        "table": ("half_x_m", "half_y_m", "thickness_m"),
    }.items():
        for key in keys:
            value = config[obj].get(key)
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Missing/invalid physical measurement: {obj}.{key}")
    tape, holder = config["tape"], config["holder"]
    if tape["inner_radius_m"] >= tape["outer_radius_m"]:
        raise ValueError("Tape must have a positive wall thickness")
    if tape["outer_radius_m"] >= holder["inner_radius_m"]:
        raise ValueError("Tape does not fit horizontally inside measured holder")
    if tape["height_m"] + holder["bottom_m"] >= holder["height_m"]:
        raise ValueError("Tape cannot fit fully below the measured holder rim")
    if holder["wall_m"] >= holder["inner_radius_m"]:
        raise ValueError("Holder wall thickness is invalid")
    if holder.get("wall_style", "solid") == "diamond_mesh":
        from .lattice import validate_lattice

        validate_lattice(holder)
    elif holder.get("wall_style", "solid") != "solid":
        raise ValueError("Unknown holder wall_style")
    for name in ("tape", "holder"):
        pos = np.asarray(config[name]["position_m"], dtype=float)
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise ValueError(f"Invalid {name}.position_m")
    for name in ("front", "side"):
        camera = config["cameras"][name]
        k = np.asarray(camera["K"], dtype=float)
        transform = np.asarray(camera["T_parent_cv"], dtype=float)
        if k.shape != (3, 3) or transform.shape != (4, 4):
            raise ValueError(f"Missing camera calibration: {name}")
        if not np.allclose(k[2], [0, 0, 1]) or not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError(f"Invalid pinhole/homogeneous matrix: {name}")
        if not np.isfinite(k).all() or not np.isfinite(transform).all() or min(k[0, 0], k[1, 1]) <= 0:
            raise ValueError(f"Invalid camera calibration: {name}")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(rotation), 1
        ):
            raise ValueError(f"Invalid camera rotation: {name}")
    JointMap(config["joint_map"])


class JointMap:
    """Robot degrees / gripper percent <-> calibrated MuJoCo radians."""

    def __init__(self, mapping):
        """Validate signed body offsets and monotone measured gripper knots."""
        self.sign = np.asarray(mapping["sign"], dtype=float)
        self.offset = np.asarray(mapping["offset_rad"], dtype=float)
        self.gripper = np.asarray(mapping["gripper_knots"], dtype=float)
        if self.sign.shape != (5,) or not np.isin(self.sign, [-1, 1]).all():
            raise ValueError("Provide one explicit +/-1 sign per body joint")
        if self.offset.shape != (5,) or not np.isfinite(self.offset).all():
            raise ValueError("Provide five finite fitted joint offsets")
        if self.gripper.ndim != 2 or self.gripper.shape[1] != 2 or len(self.gripper) < 2:
            raise ValueError("Gripper requires measured [percent, sim_rad] knots")
        if not np.isfinite(self.gripper).all() or not np.all(np.diff(self.gripper[:, 0]) > 0):
            raise ValueError("Gripper percent knots must be finite and increasing")
        delta = np.diff(self.gripper[:, 1])
        if not (np.all(delta > 0) or np.all(delta < 0)):
            raise ValueError("Gripper angle mapping must be invertible")
        if not np.allclose(self.gripper[[0, -1], 0], [0, 100]):
            raise ValueError("Gripper calibration must span 0..100 percent")

    def to_sim(self, state):
        """Convert a finite robot state into simulator radians without clipping."""
        state = np.asarray(state, dtype=float)
        if state.shape != (6,) or not np.isfinite(state).all() or not 0 <= state[5] <= 100:
            raise ValueError("Expected finite six-vector, gripper 0..100")
        return np.r_[
            np.deg2rad(state[:5]) * self.sign + self.offset,
            np.interp(state[5], self.gripper[:, 0], self.gripper[:, 1]),
        ]

    def to_robot(self, q):
        """Invert the body mapping and measured gripper interpolation."""
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected finite six-vector")
        knots = self.gripper if self.gripper[-1, 1] > self.gripper[0, 1] else self.gripper[::-1]
        if not knots[0, 1] - 1e-6 <= q[5] <= knots[-1, 1] + 1e-6:
            raise ValueError("Simulator gripper outside measured calibration")
        return np.r_[
            np.rad2deg((q[:5] - self.offset) / self.sign), np.interp(q[5], knots[:, 1], knots[:, 0])
        ].astype(np.float32)


def require_gate(config, report_path):
    """Reject absent, failed or stale calibration acceptance evidence."""
    report = json.loads(Path(report_path).read_text())
    if not report.get("passed") or report.get("config_sha256") != fingerprint(config):
        raise ValueError("Calibration acceptance report is missing, failed, or belongs to another scene")
    return report
