"""Merge verified independent trajectories without re-encoding their videos."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from common import sha256, write_json


def validate_sources(entries, expected):
    if len(entries) != expected:
        raise ValueError(f"Require exactly {expected} verified independent trajectories")
    for key in ("physical_seed", "physics_npz_sha256"):
        if len({entry[key] for entry in entries}) != expected:
            raise ValueError(f"Repeated physical trajectory: {key}")
    for index, entry in enumerate(entries):
        if entry["index"] != index or entry["height_delta_m"] != (0.02, 0.0, -0.02)[index % 3]:
            raise ValueError("Accepted trajectory order/height balance changed")


def main():
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--expected", type=int, default=100)
    args = parser.parse_args()
    entries = json.loads(args.ledger.read_text())["accepted"][: args.expected]
    validate_sources(entries, args.expected)
    roots, repo_ids = [], []
    for entry in entries:
        root = Path(entry["run"])
        verification = json.loads((root / "dataset_verification.json").read_text())
        if not verification["passed"] or len(verification["episodes"]) != 1:
            raise ValueError("Incomplete source dataset")
        if verification["training_batch_state_shape"] != [2, 25] or verification[
            "training_batch_action_shape"
        ] != [2, 6]:
            raise ValueError("Incompatible training state/action dimensions")
        if verification["framewise_appearance"]["frames"] != verification["total_frames"]:
            raise ValueError("Missing per-frame appearance records")
        dataset = root / "dataset"
        for name, expected in verification["files_sha256"].items():
            path = (dataset / name).resolve()
            if not path.is_relative_to(dataset.resolve()) or sha256(path) != expected:
                raise ValueError(f"Source dataset changed: {name}")
        roots.append(dataset)
        repo_ids.append(verification["repo_id"])
    aggregate_datasets(
        repo_ids,
        args.repo_id,
        roots=roots,
        aggr_root=args.output,
        concatenate_videos=False,
        concatenate_data=False,
    )
    merged = LeRobotDataset(args.repo_id, root=args.output, video_backend="pyav")
    if merged.num_episodes != args.expected:
        raise ValueError("Merged episode count is incorrect")
    table = pq.read_table(sorted((args.output / "data").rglob("*.parquet"))).sort_by("index")
    appearance = args.output / "meta/appearance"
    appearance.mkdir()
    banks = appearance / "banks"
    banks.mkdir()
    bank_ids, episodes = set(), []
    cursor = 0
    with (appearance / "frames.jsonl").open("x") as out:
        for index, (entry, source_root, repo_id) in enumerate(zip(entries, roots, repo_ids, strict=True)):
            source = LeRobotDataset(repo_id, root=source_root, video_backend="pyav")
            original = pq.read_table(sorted((source_root / "data").rglob("*.parquet"))).sort_by("index")
            count = original.num_rows
            actual = table.slice(cursor, count)
            for key in (
                "observation.state",
                "action",
                "observation.table_height",
                "observation.camera_pose.front",
                "observation.camera_pose.side",
            ):
                if not np.array_equal(
                    np.asarray(original[key].to_pylist(), dtype=np.float32),
                    np.asarray(actual[key].to_pylist(), dtype=np.float32),
                ):
                    raise ValueError(f"Merge changed labels/spatial features: {index}: {key}")
            for frame in (0, count // 2, count - 1):
                a, b = source[frame], merged[cursor + frame]
                for name in ("observation.images.front", "observation.images.side"):
                    if not np.array_equal(a[name].numpy(), b[name].numpy()):
                        raise ValueError("Merged video/timestamps differ from source")
            source_bank = source_root / "meta/appearance/texture_bank"
            bank_id = sha256(source_bank / "manifest.json")
            if bank_id not in bank_ids:
                shutil.copytree(source_bank, banks / bank_id)
                bank_ids.add(bank_id)
            rows = [
                json.loads(line)
                for line in (source_root / "meta/appearance/frames.jsonl").read_text().splitlines()
            ]
            if len(rows) != count:
                raise ValueError("Merged appearance coverage is incomplete")
            for frame, row in enumerate(rows):
                row.update(dataset_index=cursor + frame, episode_index=index, texture_bank_id=bank_id)
                out.write(json.dumps(row) + "\n")
            episodes.append(
                {
                    **entry,
                    "frames": count,
                    "dataset_from_index": cursor,
                    "dataset_to_index": cursor + count,
                    "texture_bank_id": bank_id,
                }
            )
            cursor += count
    if table.num_rows != cursor or not np.array_equal(np.asarray(table["index"]), np.arange(cursor)):
        raise ValueError("Merged global frame index is incorrect")
    write_json(appearance / "episodes.json", episodes)
    split = {"train": list(range(90)), "validation": list(range(90, 100))} if args.expected == 100 else None
    (args.output / "README.md").write_text(
        f"# SO101 framewise-texture training batch\n\n{args.expected} independent physical successes; {cursor} frames at 30 FPS.\n"
        "25D state, 6D action, paired RGB. Table heights +2/0/-2 cm and independently perturbed camera mounts.\n"
        "Five appearance groups change at every frame. Texture IDs are audit metadata, not policy inputs.\n"
        "All physical seeds and NPZ hashes are unique; videos were copied without re-encoding.\n"
        "User explicitly authorized automatic training after 100 verified episodes.\n"
        "calibration_passed=false. No real-robot performance claim.\n"
    )
    report = {
        "passed": True,
        "episodes": args.expected,
        "frames": cursor,
        "state_dimension": 25,
        "action_dimension": 6,
        "videos_reencoded": False,
        "all_labels_exactly_matched": True,
        "source_video_reader_samples_exactly_matched": True,
        "split": split,
        "unique_physical_seeds": args.expected,
        "unique_physics_hashes": args.expected,
        "appearance_frames": cursor,
        "calibration_passed": False,
    }
    write_json(args.output.parent / f"{args.output.name}_verification.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
