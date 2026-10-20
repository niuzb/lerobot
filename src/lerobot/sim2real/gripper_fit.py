"""Offline, provisional jaw mapping from ruler measurements and pinned geometry."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _mujoco_available, _scipy_available, require_package

from .config import digest, fingerprint, write_json

if TYPE_CHECKING or _mujoco_available:
    import mujoco
if TYPE_CHECKING or _scipy_available:
    from scipy.optimize import least_squares


def select_measurements(document):
    """Keep reported values; never treat a clipped endpoint as an exact position."""
    excluded = {
        name
        for flag in document.get("quality_flags", [])
        if flag.get("status") == "provisionally_excluded_from_fit"
        for name in flag["measurement_ids"]
    }
    groups = {"fit": [], "check": [], "closed": [], "excluded": []}
    seen = set()
    for row in document["measurements"]:
        name = row["id"]
        if name in seen:
            raise ValueError(f"Duplicate measurement id: {name}")
        seen.add(name)
        percent, gap = float(row["actual_opening_percent"]), float(row["gap_mm"])
        if not np.isfinite([percent, gap]).all() or not 0 <= percent <= 100 or gap < 0:
            raise ValueError(f"Invalid measurement: {name}")
        if row["split"] not in ("fit", "check"):
            raise ValueError(f"Unknown fit/check split: {name}")
        item = {"id": name, "percent": percent, "gap_mm": gap, "split": row["split"]}
        if name in excluded:
            groups["excluded"].append(dict(item, reason="Previously flagged inconsistent repeat"))
        elif percent in (0, 100) and not row.get("raw_endpoint_verified", False):
            groups["excluded"].append(
                dict(item, reason="Display endpoint without synchronous raw verification")
            )
        elif gap == 0:
            groups["closed"].append(item)
        else:
            groups[row["split"]].append(item)
    if len(groups["fit"]) < 2 or np.ptp([r["percent"] for r in groups["fit"]]) < 20:
        raise ValueError("Need >=2 nonsaturated fit measurements spanning >=20 percentage points")
    return groups


class JawGeometry:
    """The same named contact spheres used by the scene's calibration check."""

    def __init__(self, asset):
        """Load the source model without editing its limits or collision shapes."""
        require_package("mujoco", extra="sim2real")
        self.model = mujoco.MjModel.from_xml_path(str(asset))
        self.data = mujoco.MjData(self.model)
        self.joint = self.model.joint("gripper").id
        self.qidx = self.model.jnt_qposadr[self.joint]
        self.fixed, self.moving = (
            self.model.geom(n).id for n in ("fixed_jaw_sph_tip1", "moving_jaw_sph_tip1")
        )
        actuator = self.model.actuator("gripper").id
        self.limits = np.array(
            [
                max(self.model.jnt_range[self.joint, 0], self.model.actuator_ctrlrange[actuator, 0]),
                min(self.model.jnt_range[self.joint, 1], self.model.actuator_ctrlrange[actuator, 1]),
            ]
        )

    def gap_mm(self, angle):
        """Evaluate kinematics only, including angles outside limits for diagnosis."""
        self.data.qpos[self.qidx] = angle
        mujoco.mj_forward(self.model, self.data)
        distance = np.linalg.norm(self.data.geom_xpos[self.fixed] - self.data.geom_xpos[self.moving])
        radii = self.model.geom_size[self.fixed, 0] + self.model.geom_size[self.moving, 0]
        return float(max(0, distance - radii) * 1000)


def fit_gripper(config, measurements, out, source_path=None):
    """Fit a two-parameter angle curve; emit evidence, never a passed scene gate."""
    require_package("scipy", extra="scipy-dep")
    groups = select_measurements(measurements)
    geometry = JawGeometry(config["asset"])

    def residual(parameters):
        scale, offset = parameters
        return [geometry.gap_mm(scale * r["percent"] + offset) - r["gap_mm"] for r in groups["fit"]]

    # Rigid motor-driven jaw: angle = scale * displayed percentage + offset.
    # Do not use held-out gaps to pick the solution, or extend the asset limits.
    fits = [
        least_squares(residual, [0.01, offset], bounds=([0.001, -0.5], [0.03, 0.5]))
        for offset in (-0.3, -0.2, 0.0)
    ]
    result = min(fits, key=lambda r: float(np.sum(r.fun**2)))
    scale, offset = result.x
    rows = {}
    for split in ("fit", "check", "closed"):
        rows[split] = [
            dict(
                r,
                predicted_gap_mm=geometry.gap_mm(scale * r["percent"] + offset),
                error_mm=geometry.gap_mm(scale * r["percent"] + offset) - r["gap_mm"],
            )
            for r in groups[split]
        ]
    curve = [
        {
            "percent": float(p),
            "sim_rad": float(scale * p + offset),
            "gap_mm": geometry.gap_mm(scale * p + offset),
        }
        for p in np.linspace(0, 100, 101)
    ]
    angle_range = np.array([offset, 100 * scale + offset])
    within_limits = bool(
        np.all(angle_range >= geometry.limits[0]) and np.all(angle_range <= geometry.limits[1])
    )
    check_max = max((abs(r["error_mm"]) for r in rows["check"]), default=None)
    candidate = {
        "status": "diagnostic_only_not_accepted",
        "formula": "sim_rad = scale_rad_per_percent * percent + offset_rad",
        "scale_rad_per_percent": float(scale),
        "offset_rad": float(offset),
        "gripper_knots": [[0.0, float(angle_range[0])], [100.0, float(angle_range[1])]],
        "fit_opening_range_percent": [
            min(r["percent"] for r in groups["fit"]),
            max(r["percent"] for r in groups["fit"]),
        ],
        "endpoints_are_extrapolated": True,
        "asset": config["asset"],
        "source_config_sha256": fingerprint(config),
        "measurements_sha256": fingerprint(measurements),
        "full_gripper_mapping_validated": False,
    }
    report = {
        "passed": False,
        "scope": "Ruler gaps versus named collision-tip sphere surfaces; no hardware or dynamics",
        "fit_model": candidate,
        "rows": rows,
        "excluded": groups["excluded"],
        "heldout_max_error_mm": check_max,
        "heldout_gaps_within_3mm": check_max is not None and len(rows["check"]) >= 3 and check_max <= 3,
        "asset_joint_actuator_range_rad": geometry.limits.tolist(),
        "candidate_within_asset_limits": within_limits,
        "asset_gap_at_lower_limit_mm": geometry.gap_mm(geometry.limits[0]),
        "candidate_gap_is_monotone_0_100": bool(np.min(np.diff([r["gap_mm"] for r in curve])) >= -1e-6),
        "curve": curve,
        "blockers": [
            "Ruler contact locations have not been independently matched to the asset's sphere surfaces",
            "Exact raw 0/100 endpoints lack synchronized gap measurements",
            "Closed-contact geometry and joint limits require review before use in dynamic simulation",
        ],
        "original_asset_modified": False,
        "formal_config_modified": False,
    }
    if source_path is not None:
        report["source_file"] = {"path": str(Path(source_path).resolve()), "sha256": digest(source_path)}
    write_json(out / "gripper.candidate.json", candidate)
    write_json(out / "gripper_report.json", report)
    return report
