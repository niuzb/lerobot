"""Orient virtual cameras from the user-provided wrist and external photos."""

import argparse
import json
from pathlib import Path

import numpy as np
from camera_views import look_at_cv
from common import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text())
    tilt = np.deg2rad(34.0)
    wrist = np.eye(4)
    # Camera optical +Z looks toward the tips. The roll places the jaw bases
    # below the tips, as in camera1.jpg. This is a physical camera rotation,
    # not a mirror/flip of captured pixels.
    wrist[:3, :3] = [[-1, 0, 0], [0, np.cos(tilt), np.sin(tilt)], [0, np.sin(tilt), -np.cos(tilt)]]
    wrist[:3, 3] = [-0.0075, -0.060, -0.030]
    external_position = [0.610, -0.225, 0.380]
    external_target = [0.267, -0.055, 0.050]
    write_json(
        args.output,
        {
            "revision": "reference_camera_views_v3",
            "source_scene_sha256": sha256(args.scene),
            "reference_images": {
                "front": {
                    "role": "wrist",
                    "path": str((args.reference_dir / "camera1.jpg").resolve()),
                    "sha256": sha256(args.reference_dir / "camera1.jpg"),
                },
                "side": {
                    "role": "external",
                    "path": str((args.reference_dir / "camera2.jpg").resolve()),
                    "sha256": sha256(args.reference_dir / "camera2.jpg"),
                },
            },
            "cameras": {
                "front": {
                    "parent": "gripper",
                    "position_m": wrist[:3, 3].tolist(),
                    "T_parent_cv": wrist.tolist(),
                    "K": scene["cameras"]["front"]["K"],
                    "distortion": [0.0] * 5,
                },
                "side": {
                    "parent": "world",
                    "position_m": external_position,
                    "look_at_m": external_target,
                    "T_parent_cv": look_at_cv(external_position, external_target),
                    "K": [[720.0, 0.0, 319.5], [0.0, 720.0, 239.5], [0.0, 0.0, 1.0]],
                    "distortion": [0.0] * 5,
                },
            },
            "calibration_passed": False,
            "physical_camera_calibration": "Photo-guided orientation/composition adjustment; not a metric calibration",
            "train_allowed": False,
        },
    )


if __name__ == "__main__":
    main()
