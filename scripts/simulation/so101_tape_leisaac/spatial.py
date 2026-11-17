"""Spatial conditioning and reproducible camera/workspace augmentation (metres)."""

import copy
import json
from pathlib import Path

import numpy as np
from common import JOINTS, sha256, transform

CAMERAS = ("front", "side")
SPATIAL_KEYS = ("observation.table_height", *(f"observation.camera_pose.{name}" for name in CAMERAS))


def stored_feature_rows(values, expected_shape):
    """LeRobot stores singleton numeric features as scalar parquet columns."""
    rows = np.asarray(values, dtype=np.float32)
    if tuple(expected_shape) == (1,) and rows.ndim == 1:
        rows = rows[:, None]
    if rows.ndim != 2 or rows.shape[1:] != tuple(expected_shape):
        raise ValueError("Stored spatial feature has an unexpected shape")
    return rows


def rendered_cameras(run, render, manifest):
    """Verify the rendered snapshot, including legacy JSON reserialization."""
    path = Path(render) / "camera_config.json"
    if not path.exists():
        return None
    snapshot = json.loads(path.read_text())
    expected = manifest.get("camera_snapshot_sha256")
    if expected is not None:
        if sha256(path) != expected:
            raise ValueError("Rendered camera snapshot changed")
    else:
        # Older renderers saved a reserialized copy but hashed the original
        # source bytes. Verify that original and its complete parsed content.
        original = Path(run) / "camera_views.json"
        if not original.exists() or sha256(original) != manifest["camera_config_sha256"]:
            raise ValueError("Original rendered camera configuration cannot be verified")
        if json.loads(original.read_text()) != snapshot:
            raise ValueError("Rendered camera snapshot differs from its verified source")
    return snapshot["cameras"]


def base_position(config):
    value = np.asarray(config.get("spatial", {}).get("robot_base_position_m", [0, 0, 0]), dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Robot-base position must be a finite XYZ vector in metres")
    return value


def table_height(config):
    value = float(config.get("spatial", {}).get("table_top_height_m", 0.0))
    if not np.isfinite(value):
        raise ValueError("Table height must be finite")
    return value


def validate_pose(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Camera pose must be a finite 4x4 matrix")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("Invalid homogeneous-transform last row")
    if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-7):
        raise ValueError("Camera orientation is not orthonormal")
    if not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-7):
        raise ValueError("Camera orientation must be a rotation, never a reflection")
    return pose


def rpy_rotation(rpy):
    roll, pitch, yaw = rpy
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def pose_vector(matrix):
    """XYZ plus a scalar-first quaternion; sign is canonical for metadata."""
    matrix = validate_pose(matrix)
    r = matrix[:3, :3]
    # Symmetric eigensystem remains stable at 180 degrees.
    k = np.array(
        [
            [r[0, 0] - r[1, 1] - r[2, 2], r[1, 0] + r[0, 1], r[2, 0] + r[0, 2], r[2, 1] - r[1, 2]],
            [r[1, 0] + r[0, 1], r[1, 1] - r[0, 0] - r[2, 2], r[2, 1] + r[1, 2], r[0, 2] - r[2, 0]],
            [r[2, 0] + r[0, 2], r[2, 1] + r[1, 2], r[2, 2] - r[0, 0] - r[1, 1], r[1, 0] - r[0, 1]],
            [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], np.trace(r)],
        ]
    )
    _, vectors = np.linalg.eigh(k)
    quaternion = vectors[:, -1][[3, 0, 1, 2]]
    if quaternion[0] < 0:
        quaternion *= -1
    return np.r_[matrix[:3, 3], quaternion].astype(np.float32)


def state_names():
    names = [f"{joint}.pos" for joint in JOINTS] + ["table.height_world_m"]
    for camera in CAMERAS:
        names += [f"{camera}.{axis}_base_m" for axis in ("x", "y", "z")]
        names += [f"{camera}.R{row}{column}" for column in (0, 1) for row in range(3)]
    return names


def feature_schema():
    return {
        "state_dimension": 25,
        "state_names": state_names(),
        "camera_pose_frame": "robot base; OpenCV optical axes +X right, +Y down, +Z forward",
        "camera_pose_order": ["x_m", "y_m", "z_m", "qw", "qx", "qy", "qz"],
        "policy_orientation": "first two columns of camera-to-base rotation, column-major (6D)",
        "table_height_reference": "world Z relative to the baseline tabletop, not measured floor height",
        "units": "joint degrees, gripper percent, translations metres, rotations dimensionless",
        "calibration_passed": False,
    }


def frame_features(config, base_world, gripper_world):
    """Compute conditioning from physical link poses, never from target actions."""
    base_world, gripper_world = validate_pose(base_world), validate_pose(gripper_world)
    extras = {"observation.table_height": np.array([table_height(config)], dtype=np.float32)}
    auxiliary = [extras["observation.table_height"]]
    for name in CAMERAS:
        camera = config["cameras"][name]
        relative = validate_pose(camera["T_parent_cv"])
        if camera["parent"] == "gripper":
            camera_world = gripper_world @ relative
        elif camera["parent"] == "world":
            camera_world = relative
        else:
            raise ValueError(f"Unsupported camera parent: {camera['parent']}")
        camera_base = np.linalg.solve(base_world, camera_world)
        extras[f"observation.camera_pose.{name}"] = pose_vector(camera_base)
        auxiliary.append(np.r_[camera_base[:3, 3], camera_base[:3, :2].T.ravel()])
    return np.concatenate(auxiliary).astype(np.float32), extras


def episode_features(config, environment, arrays):
    names = environment["body_names"]
    base_index, gripper_index = names.index("base"), names.index("gripper")
    output = {key: [] for key in SPATIAL_KEYS}
    auxiliary = []
    for links in arrays["robot_links"]:
        base, gripper = links[base_index], links[gripper_index]
        if not np.allclose(base[:3], base_position(config), atol=0.001):
            raise ValueError("Recorded robot base differs from the spatial scene configuration")
        row, extras = frame_features(
            config, transform(base[:3], base[3:7]), transform(gripper[:3], gripper[3:7])
        )
        auxiliary.append(row)
        for key in output:
            output[key].append(extras[key])
    return np.asarray(auxiliary, dtype=np.float32), {
        key: np.asarray(rows, dtype=np.float32) for key, rows in output.items()
    }


def augmented_scene(
    scene,
    cameras,
    *,
    height_delta_m,
    seed,
    wrist_translation_m=0.003,
    wrist_rotation_deg=2.0,
    external_translation_m=0.01,
    external_rotation_deg=3.0,
):
    """Move the entire mounted workspace; keep the external camera floor-fixed.

    Camera jitter is a mounting perturbation, held fixed for the whole episode.
    Translation is sampled per parent-frame axis; RPY rotates the optical frame.
    """
    limits = [wrist_translation_m, wrist_rotation_deg, external_translation_m, external_rotation_deg]
    if not np.isfinite([height_delta_m, *limits]).all() or min(limits) < 0:
        raise ValueError("Augmentation limits must be finite and nonnegative")
    result = copy.deepcopy(scene)
    offset = np.array([0.0, 0.0, height_delta_m])
    result["spatial"] = {
        "table_top_height_m": table_height(scene) + height_delta_m,
        "robot_base_position_m": (base_position(scene) + offset).tolist(),
        "height_reference": "baseline tabletop; absolute floor height is unknown",
        "calibration_passed": False,
    }
    for name in ("tape", "holder"):
        result[name]["position_m"] = (np.asarray(scene[name]["position_m"]) + offset).tolist()
    result["cameras"] = copy.deepcopy(cameras["cameras"])
    rng = np.random.default_rng(seed)
    sampled = {}
    for name, translation_limit, rotation_limit in (
        ("front", wrist_translation_m, wrist_rotation_deg),
        ("side", external_translation_m, external_rotation_deg),
    ):
        camera = result["cameras"][name]
        if camera["parent"] != ("gripper" if name == "front" else "world"):
            raise ValueError("Require a gripper-mounted wrist camera and world-mounted external camera")
        pose = validate_pose(camera["T_parent_cv"]).copy()
        translation = rng.uniform(-translation_limit, translation_limit, 3)
        angles = rng.uniform(-rotation_limit, rotation_limit, 3)
        pose[:3, 3] += translation
        pose[:3, :3] = pose[:3, :3] @ rpy_rotation(np.deg2rad(angles))
        validate_pose(pose)
        camera["T_parent_cv"] = pose.tolist()
        camera["position_m"] = pose[:3, 3].tolist()
        if "look_at_m" in camera:
            distance = np.linalg.norm(
                np.asarray(camera["look_at_m"]) - np.asarray(cameras["cameras"][name]["position_m"])
            )
            camera["look_at_m"] = (pose[:3, 3] + distance * pose[:3, 2]).tolist()
        sampled[name] = {"translation_parent_m": translation.tolist(), "optical_rpy_deg": angles.tolist()}
    result["spatial_augmentation"] = {
        "seed": seed,
        "workspace_height_delta_m": height_delta_m,
        "camera_mount_perturbations": sampled,
        "constant_within_episode": True,
        "intrinsics_changed": False,
        "calibration_passed": False,
    }
    return result


def new_spatial_dataset(path, repo_id):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (25,), "names": state_names()},
        "action": {"dtype": "float32", "shape": (6,), "names": [f"{joint}.pos" for joint in JOINTS]},
        "observation.table_height": {"dtype": "float32", "shape": (1,), "names": ["height_world_m"]},
    }
    for camera in CAMERAS:
        features[f"observation.camera_pose.{camera}"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": feature_schema()["camera_pose_order"],
        }
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=path,
        fps=30,
        robot_type="so101_follower",
        features=features,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18),
        image_writer_threads=8,
    )
