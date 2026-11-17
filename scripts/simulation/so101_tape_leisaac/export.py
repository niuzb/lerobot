"""Export only verified full-rate physical demos into an independent LeRobot v3 dataset."""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from common import TASK, JointMap, sha256, write_json
from spatial import (
    SPATIAL_KEYS,
    episode_features,
    feature_schema,
    new_spatial_dataset,
    rendered_cameras,
    stored_feature_rows,
    table_height,
)
from texture_randomization import verified_appearance

from lerobot.sim2real.data import new_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/so101_tape_leisaac_pilot_20261108")
    parser.add_argument(
        "--spatial-features",
        action="store_true",
        help="Append table/camera conditioning to the policy state (25D); actions remain 6D",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads((args.run / "scene.json").read_text())
    rendering = json.loads((args.render / "render_manifest.json").read_text())
    if not rendering["passed"] or rendering["stride"] != 1:
        raise ValueError("Require successful full-rate paired-camera rendering")
    environment = json.loads((args.physics / "environment.json").read_text())
    if args.spatial_features:
        if "scene_config_sha256" in rendering and rendering["scene_config_sha256"] != sha256(
            args.run / "scene.json"
        ):
            raise ValueError("Rendered scene configuration differs from the feature source")
        cameras = rendered_cameras(args.run, args.render, rendering)
        if cameras is not None:
            config["cameras"] = cameras
        if "table_top_height_world_m" in environment and not np.isclose(
            environment["table_top_height_world_m"], table_height(config), atol=1e-7
        ):
            raise ValueError("Recorded physical table height differs from the feature source")
    mapping = JointMap(config["joint_map"])
    expected_states, expected_actions = [], []
    expected_spatial = {key: [] for key in SPATIAL_KEYS}
    appearance_metadata = []
    dataset = (new_spatial_dataset if args.spatial_features else new_dataset)(args.output, args.repo_id)
    episodes = []
    try:
        for episode in rendering["episodes"]:
            appearance_rows = verified_appearance(args.render, episode)
            prefix = args.physics / f"attempt_{episode['attempt']:04d}"
            source = json.loads(prefix.with_suffix(".json").read_text())
            if (
                not source["success"]
                or source["stable_control_ticks"] < 30
                or not episode["training_eligible"]
            ):
                raise ValueError("Failed/unstable demonstration cannot enter training data")
            if sha256(prefix.with_suffix(".npz")) != episode["physics_npz_sha256"]:
                raise ValueError("Physical source changed")
            arrays = np.load(prefix.with_suffix(".npz"))
            if args.spatial_features:
                auxiliary, spatial_rows = episode_features(config, environment, arrays)
            if (
                len(arrays["qpos"]) != episode["frames"]
                or not arrays["success"][-1]
                or arrays["stable"][-1] < 30
            ):
                raise ValueError("Physical terminal success evidence is incomplete")
            videos = []
            try:
                for name in ("front", "side"):
                    path = args.render / prefix.name / f"{name}.mp4"
                    if sha256(path) != episode["files_sha256"][path.name]:
                        raise ValueError("Camera video changed")
                    cap = cv2.VideoCapture(str(path))
                    if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - 30) > 0.01:
                        raise ValueError("Camera video is not 30 FPS")
                    videos.append(cap)
                for index in range(episode["frames"]):
                    state = mapping.to_robot(arrays["qpos"][index])
                    if args.spatial_features:
                        state = np.concatenate((state, auxiliary[index])).astype(np.float32)
                    action = mapping.to_robot(arrays["target"][index])
                    if not np.isfinite(np.r_[state, action]).all():
                        raise ValueError("Non-finite state/action")
                    frame = {"observation.state": state, "action": action, "task": TASK}
                    if args.spatial_features:
                        for key in SPATIAL_KEYS:
                            frame[key] = spatial_rows[key][index]
                            expected_spatial[key].append(frame[key])
                    for name, cap in zip(("front", "side"), videos, strict=True):
                        ok, image = cap.read()
                        if not ok or image.shape != (480, 640, 3):
                            raise ValueError(f"Missing paired camera frame {index}: {name}")
                        frame[f"observation.images.{name}"] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    dataset.add_frame(frame)
                    if appearance_rows is not None:
                        appearance_metadata.append(
                            {
                                **appearance_rows[index],
                                "dataset_index": len(expected_states),
                                "episode_index": len(episodes),
                                "timestamp_s": index / 30,
                                "source_attempt": episode["attempt"],
                                "physics_npz_sha256": episode["physics_npz_sha256"],
                            }
                        )
                    expected_states.append(state)
                    expected_actions.append(action)
                if any(cap.read()[0] for cap in videos):
                    raise ValueError("Camera videos contain extra unmatched frames")
                dataset.save_episode()
                episodes.append(
                    {"seed": episode["seed"], "attempt": episode["attempt"], "frames": episode["frames"]}
                )
            finally:
                for cap in videos:
                    cap.release()
    finally:
        dataset.finalize()
    if appearance_metadata:
        folder = args.output / "meta/appearance"
        folder.mkdir()
        shutil.copytree(args.render / "texture_bank", folder / "texture_bank")
        (folder / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in appearance_metadata))
        write_json(
            folder / "profile.json",
            {
                **rendering["appearance"],
                "metadata_only_not_policy_input": True,
                "frames": len(appearance_metadata),
                "source_render_manifest_sha256": sha256(args.render / "render_manifest.json"),
            },
        )
        if len(appearance_metadata) != len(expected_states):
            raise ValueError("Exported texture schedule does not cover all dataset frames")
    table = pq.read_table(sorted((args.output / "data").rglob("*.parquet"))).sort_by("index")
    for key, expected in (("observation.state", expected_states), ("action", expected_actions)):
        if not np.array_equal(
            np.asarray(table[key].to_pylist(), dtype=np.float32), np.asarray(expected, dtype=np.float32)
        ):
            raise ValueError(f"Exported {key} does not match physical records")
    if args.spatial_features:
        for key, expected in expected_spatial.items():
            expected_array = np.asarray(expected, dtype=np.float32)
            actual_array = stored_feature_rows(table[key].to_pylist(), expected_array.shape[1:])
            if not np.array_equal(actual_array, expected_array):
                raise ValueError(f"Exported spatial feature differs from physical/camera records: {key}")
    info = json.loads((args.output / "meta/info.json").read_text())
    if info["total_frames"] != len(expected_states) or info["total_episodes"] != len(episodes):
        raise ValueError("Exported dataset count differs from complete recordings")
    from torch.utils.data import DataLoader

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    reader = LeRobotDataset(args.repo_id, root=args.output, video_backend="pyav")
    checked_indices = []
    start = 0
    for episode in episodes:
        for index in (start, start + episode["frames"] // 2, start + episode["frames"] - 1):
            item = reader[index]
            if not np.array_equal(item["observation.state"].numpy(), expected_states[index]):
                raise ValueError("Dataset reader state differs from the physical record")
            for name in ("front", "side"):
                image = item[f"observation.images.{name}"]
                if tuple(image.shape) != (3, 480, 640) or not image.isfinite().all() or image.std() < 0.004:
                    raise ValueError("Dataset reader returned invalid camera data")
            checked_indices.append(index)
        start += episode["frames"]
    batch = next(iter(DataLoader(reader, batch_size=2, num_workers=0)))
    if tuple(batch["action"].shape) != (2, 6):
        raise ValueError("Training batch has incorrect action dimensions")
    state_dimension = 25 if args.spatial_features else 6
    if tuple(batch["observation.state"].shape) != (2, state_dimension):
        raise ValueError("Training batch has incorrect state dimensions")
    final = reader[len(reader) - 1]
    paired = np.concatenate(
        [
            (final[f"observation.images.{name}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            for name in ("front", "side")
        ],
        axis=1,
    )
    cv2.imwrite(str(args.output.parent / "dataset_final_frame.png"), cv2.cvtColor(paired, cv2.COLOR_RGB2BGR))
    state_description = (
        "State: 6 joint values + table height + two camera positions and 6D rotations (25D).\n"
        "Camera features are computed per frame in robot-base coordinates from recorded body poses.\n"
        "Table height is relative to the baseline tabletop, not a measured floor height.\n"
        "Actions remain six joint targets. See spatial_feature_schema in dataset_verification.json.\n\n"
        if args.spatial_features
        else "Six state/action values: five arm joint angles in degrees, then gripper opening in percent.\n\n"
    )
    (args.output / "README.md").write_text(
        "# SO-101 tape into pen holder — LeIsaac pilot\n\n"
        f"{len(episodes)} successful simulated episodes, {len(expected_states)} frames, 30 FPS.\n"
        "Two paired 640x480 RGB videos: front = wrist, side = external.\n"
        + state_description
        + "PhysX trajectories are rendered from recorded physical link/object world poses.\n"
        "Success requires grasp, lift, carry, release, full containment, support and 30 stable control ticks.\n"
        "calibration_passed=false. This pilot has not been physically calibrated or evaluated on a real robot.\n"
        "Camera occlusion still requires visual review; spatial features do not imply calibration.\n"
        "See the adjacent dataset_verification.json and the recorded physics/render paths for provenance.\n",
        encoding="utf-8",
    )
    report = {
        "passed": True,
        "dataset": str(args.output),
        "repo_id": args.repo_id,
        "episodes": episodes,
        "total_frames": len(expected_states),
        "fps": 30,
        "camera_roles": rendering["camera_roles"],
        "all_state_action_rows_exactly_matched": True,
        "dataset_reader_checked_indices": checked_indices,
        "dataset_reader_video_backend": "pyav",
        "training_batch_action_shape": list(batch["action"].shape),
        "training_batch_state_shape": list(batch["observation.state"].shape),
        "spatial_feature_schema": feature_schema() if args.spatial_features else None,
        "camera_config_sha256": rendering.get("camera_config_sha256"),
        "physical_source": str(args.physics),
        "render_source": str(args.render),
        "human_review_pending": True,
        "framewise_appearance": {
            "frames": len(appearance_metadata),
            "profile": rendering.get("appearance"),
            "metadata_file": "meta/appearance/frames.jsonl" if appearance_metadata else None,
        },
        "source_config": config["migration"],
        "calibration_passed": False,
        "files_sha256": {
            str(p.relative_to(args.output)): sha256(p) for p in args.output.rglob("*") if p.is_file()
        },
    }
    write_json(args.output.parent / "dataset_verification.json", report)
    print(json.dumps({k: report[k] for k in ("passed", "dataset", "total_frames", "fps")}), flush=True)


if __name__ == "__main__":
    main()
