"""Check object framing over every recorded physical frame before video review."""

import argparse
import json
from pathlib import Path

import numpy as np
from common import sha256, transform, write_json


def object_points(scene):
    angles = np.arange(64) * np.pi / 32
    tape, holder = scene["tape"], scene["holder"]
    return {
        "tape": np.array(
            [
                [tape["outer_radius_m"] * np.cos(a), tape["outer_radius_m"] * np.sin(a), z]
                for a in angles
                for z in (-tape["height_m"] / 2, tape["height_m"] / 2)
            ]
        ),
        "holder": np.array(
            [
                [
                    (holder["inner_radius_m"] + holder["wall_m"]) * np.cos(a),
                    (holder["inner_radius_m"] + holder["wall_m"]) * np.sin(a),
                    z,
                ]
                for a in angles
                for z in (0, holder["height_m"])
            ]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=16.0)
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text())
    cameras = json.loads(args.camera_config.read_text())["cameras"]
    manifest = json.loads((args.physics / "generation_manifest.json").read_text())
    environment = json.loads((args.physics / "environment.json").read_text())
    gripper = environment["body_names"].index("gripper")
    points = object_points(scene)
    results = []
    for attempt, summary in enumerate(manifest["attempts"]):
        if not summary["success"]:
            continue
        source = args.physics / f"attempt_{attempt:04d}.npz"
        if sha256(source) != summary["npz_sha256"]:
            raise ValueError("Physical recording changed")
        arrays = np.load(source)
        for name, objects in (("front", ("tape",)), ("side", ("tape", "holder"))):
            values = cameras[name]
            if np.any(np.asarray(values["distortion"]) != 0):
                raise ValueError("This framing audit requires the revised pinhole views")
            intrinsic = np.asarray(values["K"])
            relative = np.asarray(values["T_parent_cv"])
            for obj in objects:
                bounds, margins, depths, sizes = [], [], [], []
                for index in range(len(arrays["qpos"])):
                    pose = arrays[obj][index]
                    object_world = transform(pose[:3], pose[3:7])
                    camera_world = relative
                    if name == "front":
                        link = arrays["robot_links"][index, gripper]
                        camera_world = transform(link[:3], link[3:7]) @ relative
                    world = points[obj] @ object_world[:3, :3].T + object_world[:3, 3]
                    local = (world - camera_world[:3, 3]) @ camera_world[:3, :3]
                    pixels = (local[:, :2] / local[:, 2:]) @ intrinsic[:2, :2].T + intrinsic[:2, 2]
                    low, high = pixels.min(axis=0), pixels.max(axis=0)
                    bounds.append([*low, *high])
                    margins.append(min(*low, 639 - high[0], 479 - high[1]))
                    depths.append(float(local[:, 2].min()))
                    sizes.append(high - low)
                margins, bounds, sizes = np.asarray(margins), np.asarray(bounds), np.asarray(sizes)
                results.append(
                    {
                        "attempt": attempt,
                        "camera": name,
                        "object": obj,
                        "frames": len(margins),
                        "frames_with_complete_object_and_margin": int(
                            ((margins >= args.margin) & (np.asarray(depths) > 0.002)).sum()
                        ),
                        "minimum_edge_margin_px": float(margins.min()),
                        "worst_frame": int(margins.argmin()),
                        "minimum_object_size_px": sizes.min(axis=0).tolist(),
                        "all_frames_bbox_px": [*bounds[:, :2].min(axis=0), *bounds[:, 2:].max(axis=0)],
                    }
                )
    passed = bool(results) and all(
        item["frames"] == item["frames_with_complete_object_and_margin"] for item in results
    )
    write_json(
        args.output,
        {
            "geometric_framing_passed": passed,
            "required_margin_px": args.margin,
            "camera_config_sha256": sha256(args.camera_config),
            "results": results,
            "occlusion_check": "Separate rendered-image review is required",
            "calibration_passed": False,
            "train_allowed": False,
        },
    )
    print(json.dumps({"geometric_framing_passed": passed, "results": results}), flush=True)


if __name__ == "__main__":
    main()
