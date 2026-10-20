"""Durable generation/export/training pipeline for the explicitly approximate scene."""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .config import JOINTS, TASK, digest, write_json


def verify_dataset(generated):
    """Match every numerical frame to its physical recording and decode distributed video samples."""
    import pyarrow.parquet as pq

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    report = json.loads((generated / "generation_manifest.json").read_text())
    if not report["passed"] or report["physics_only"]:
        raise ValueError("Dataset export is incomplete")
    root = generated / "dataset"
    info = json.loads((root / "meta/info.json").read_text())
    accepted = [row for row in report["attempts"] if row["success"]]
    expected_state, expected_action, lengths = [], [], []
    for summary in accepted:
        path = generated / f"attempt_{summary['attempt']:04d}.json"
        if digest(path) != summary["json_sha256"]:
            raise ValueError("Physical source evidence changed")
        trajectory = json.loads(path.read_text())
        if not trajectory["rows"][-1]["success"] or summary["stable_control_ticks"] < 30:
            raise ValueError("A failed attempt entered the successful dataset")
        expected_state.extend(row["state"] for row in trajectory["rows"])
        expected_action.extend(row["sent_action"] for row in trajectory["rows"])
        lengths.append(len(trajectory["rows"]))
    if info["fps"] != 30 or info["total_episodes"] != len(accepted) or info["total_frames"] != sum(lengths):
        raise ValueError("Exported counts or cadence differ from physical recordings")
    table = pq.read_table(sorted((root / "data").rglob("*.parquet"))).sort_by("index")
    for name, expected in (("observation.state", expected_state), ("action", expected_action)):
        if info["features"][name]["names"] != [f"{joint}.pos" for joint in JOINTS]:
            raise ValueError("Joint order changed")
        actual = np.asarray(table[name].to_pylist(), dtype=np.float32)
        if not np.isfinite(actual).all() or not np.array_equal(
            actual, np.asarray(expected, dtype=np.float32)
        ):
            raise ValueError("Dataset state/action does not match the pre-action physical recording")
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    ends = np.cumsum(lengths) - 1
    reader = LeRobotDataset("local/so101_r2s2r_sim", root=root, video_backend="pyav")
    # Decode first/middle/last frames of every episode, including both camera streams.
    samples = sorted(set(starts.tolist() + ends.tolist() + ((starts + ends) // 2).tolist()))
    for i in samples:
        row = reader[i]
        for camera in ("front", "side"):
            image = row[f"observation.images.{camera}"]
            if tuple(image.shape) != (3, 480, 640) or not bool(image.isfinite().all()):
                raise ValueError("Invalid camera frame")
        if row["task"] != TASK:
            raise ValueError("Wrong task text")
        np.testing.assert_array_equal(
            row["observation.state"].numpy(), np.asarray(expected_state[i], np.float32)
        )
    result = {
        "passed": True,
        "episodes": len(accepted),
        "frames": sum(lengths),
        "video_frames_sampled_per_camera": len(samples),
        "all_state_action_rows_exactly_matched": True,
        "scene_status": report["scene_status"],
        "calibration_passed": False,
    }
    write_json(generated / "dataset_verification.json", result)
    return result


def verify_checkpoint(folder):
    """Check finished training and the native final policy/processor files."""
    manifest = json.loads((folder / "training_manifest.json").read_text())
    if not manifest["completed"] or manifest["smoke"] or manifest["steps"] != 20000:
        raise ValueError("Formal training is incomplete")
    final = folder / "checkpoints/020000/pretrained_model"
    required = ["model.safetensors", "config.json", "policy_preprocessor.json", "policy_postprocessor.json"]
    for name in required:
        if not (final / name).is_file() or (final / name).stat().st_size == 0:
            raise ValueError(f"Missing final model file: {name}")
    return {
        "checkpoint": str(final),
        "files": {p.name: p.stat().st_size for p in final.iterdir() if p.is_file()},
    }


def main():
    """Run only on the allocated worker; persist the current stage and exit result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-exitcode", type=Path, required=True)
    parser.add_argument("--mixed-output", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    status = {
        "started_unix": time.time(),
        "completed": False,
        "calibration_passed": False,
        "scene_status": "unvalidated_experiment",
        "code_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
    }

    def stage(name):
        status.update(stage=name, updated_unix=time.time())
        write_json(args.root / "status.json", status)
        print(json.dumps(status), flush=True)

    def run(name, command):
        stage(name)
        subprocess.run([sys.executable, "-u", *map(str, command)], check=True)

    generator = ["-m", "lerobot.sim2real.experimental_generation", "--config", args.config]
    training = ["so101_r2s2r.py", "train", "--config", args.config, "--base", args.base, "--device", "cuda"]
    try:
        pilot_dataset = args.root / "pilot_dataset"
        run("export_pilot", [*generator, "--export-from", args.pilot, "--output", pilot_dataset])
        stage("verify_pilot")
        status["pilot_verification"] = verify_dataset(pilot_dataset)
        run(
            "mixed_two_step_smoke",
            [
                *training,
                "--variant",
                "mixed",
                "--generated",
                pilot_dataset,
                "--smoke",
                "--output",
                args.root / "mixed_smoke",
            ],
        )
        production = args.root / "production_physics"
        run(
            "generate_200_successes",
            [
                *generator,
                "--count",
                "200",
                "--seed",
                "10000",
                "--physics-only",
                "--pilot",
                args.pilot / "generation_manifest.json",
                "--output",
                production,
            ],
        )
        dataset = args.root / "production_dataset"
        run("export_200_successes", [*generator, "--export-from", production, "--output", dataset])
        stage("verify_production")
        status["production_verification"] = verify_dataset(dataset)
        stage("waiting_for_baseline_gpu")
        deadline = time.monotonic() + 48 * 3600
        while not args.baseline_exitcode.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Baseline has no exit code after 48 hours")
            time.sleep(30)
        if args.baseline_exitcode.read_text().strip() != "0":
            raise RuntimeError("Baseline failed; inspect its log before starting the paired model")
        status["baseline"] = verify_checkpoint(args.baseline)
        run(
            "train_mixed_20000",
            [*training, "--variant", "mixed", "--generated", dataset, "--output", args.mixed_output],
        )
        status["mixed"] = verify_checkpoint(args.mixed_output)
        status.update(completed=True, finished_unix=time.time())
        stage("completed")
        (args.root / "exitcode").write_text("0\n")
    except Exception as exc:
        status.update(error=type(exc).__name__, message=str(exc), failed_stage=status.get("stage"))
        stage("failed")
        (args.root / "exitcode").write_text("1\n")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
