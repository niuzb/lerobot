"""Camera view revision for unobstructed wrist and complete workspace images."""

import argparse
import json
from pathlib import Path

import numpy as np
from common import sha256, write_json


def look_at_cv(position, target):
    """Camera-to-parent pose: optical +Z forward, +X right, +Y down."""
    position, target = np.asarray(position, dtype=float), np.asarray(target, dtype=float)
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, down, forward))
    pose[:3, 3] = position
    return pose.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text())
    cameras = {
        "front": {
            "parent": "gripper",
            "position_m": [0.015, -0.110, -0.015],
            "look_at_m": [-0.050, 0.0, -0.145],
            "focal_pixels": 420.0,
        },
        "side": {
            "parent": "world",
            "position_m": [0.490, -0.330, 0.430],
            "look_at_m": [0.270, -0.055, 0.065],
            "focal_pixels": 800.0,
        },
    }
    for values in cameras.values():
        focal = values["focal_pixels"]
        values["K"] = [[focal, 0.0, 319.5], [0.0, focal, 239.5], [0.0, 0.0, 1.0]]
        values["distortion"] = [0.0] * 5
        values["T_parent_cv"] = look_at_cv(values["position_m"], values["look_at_m"])
    write_json(
        args.output,
        {
            "revision": "camera_views_v2",
            "source_scene_sha256": sha256(args.scene),
            "original_cameras": scene["cameras"],
            "cameras": cameras,
            "calibration_passed": False,
            "physical_camera_calibration": "Not performed; these are adjusted virtual camera mounts",
            "train_allowed": False,
        },
    )


if __name__ == "__main__":
    main()
