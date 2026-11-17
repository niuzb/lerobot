"""Prepare independent table-height/camera-pose conditions without touching a running batch."""

import argparse
import json
import shutil
from pathlib import Path

from common import sha256, write_json
from spatial import augmented_scene, feature_schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--height-deltas", default="-0.02,0,0.02", help="Metres relative to the baseline tabletop"
    )
    parser.add_argument("--seed", type=int, default=71000)
    parser.add_argument(
        "--wrist-translation", type=float, default=0.003, help="Per-axis parent-frame bound, metres"
    )
    parser.add_argument(
        "--wrist-rotation", type=float, default=2.0, help="Per-axis optical RPY bound, degrees"
    )
    parser.add_argument("--external-translation", type=float, default=0.01)
    parser.add_argument("--external-rotation", type=float, default=3.0)
    args = parser.parse_args()
    heights = [float(value) for value in args.height_deltas.split(",")]
    if not heights or len(set(heights)) != len(heights):
        raise ValueError("Provide distinct table-height deltas")
    scene_path = args.source_run / "scene.json"
    scene = json.loads(scene_path.read_text())
    cameras = json.loads(args.camera_config.read_text())
    prepared = [
        augmented_scene(
            scene,
            cameras,
            height_delta_m=height,
            seed=args.seed + index,
            wrist_translation_m=args.wrist_translation,
            wrist_rotation_deg=args.wrist_rotation,
            external_translation_m=args.external_translation,
            external_rotation_deg=args.external_rotation,
        )
        for index, height in enumerate(heights)
    ]
    for name in ("assets", "site-packages"):
        if not (args.source_run / name).is_dir():
            raise ValueError(f"Missing runtime dependency: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(Path(__file__).parent, args.output / "code", ignore=shutil.ignore_patterns("__pycache__"))
    conditions = []
    for index, config in enumerate(prepared):
        run = args.output / f"condition_{index:02d}"
        run.mkdir()
        write_json(run / "scene.json", config)
        write_json(
            run / "camera_views.json",
            {
                "revision": "spatial_v1",
                "source_camera_sha256": sha256(args.camera_config),
                "cameras": config["cameras"],
                "augmentation": config["spatial_augmentation"],
                "calibration_passed": False,
                "human_review_pending": True,
            },
        )
        for name in ("assets", "site-packages"):
            (run / name).symlink_to((args.source_run / name).resolve(), target_is_directory=True)
        (run / "code").symlink_to((args.output / "code").resolve(), target_is_directory=True)
        conditions.append(
            {
                "run": str(run.resolve()),
                "height_delta_m": heights[index],
                "augmentation_seed": args.seed + index,
                "physical_seed_start": args.seed + 1000 + 100 * index,
                "scene_sha256": sha256(run / "scene.json"),
                "camera_config_sha256": sha256(run / "camera_views.json"),
            }
        )
    write_json(
        args.output / "spatial_plan.json",
        {
            "source_run": str(args.source_run.resolve()),
            "source_scene_sha256": sha256(scene_path),
            "source_camera_sha256": sha256(args.camera_config),
            "augmentation_bounds": {
                "wrist_translation_m": args.wrist_translation,
                "wrist_rotation_deg": args.wrist_rotation,
                "external_translation_m": args.external_translation,
                "external_rotation_deg": args.external_rotation,
            },
            "code_sha256": {path.name: sha256(path) for path in (args.output / "code").glob("*.py")},
            "conditions": conditions,
            "features": feature_schema(),
            "human_review_pending": True,
            "calibration_passed": False,
            "training_requested": False,
            "gpu_release_requested": False,
        },
    )
    print(json.dumps({"prepared": str(args.output), "conditions": conditions}), flush=True)


if __name__ == "__main__":
    main()
