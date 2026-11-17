"""Durable real78+sim100 training and paired real10 policy comparison."""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "manager.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code = root / "code"
    source = Path("/root/data/leisaac/runs/tape-train100-20261109-01")
    base = Path(
        "/root/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/d9f33c94a60fb382c90dea2164c96845bd955e28"
    )
    paths = {
        "real_root": "/root/code/lerobot/datasets/smolvla_tape_clean_20261002_142151_curated_v1",
        "sim_root": str(source / "dataset"),
        "scene": "/root/data/leisaac/runs/tape-pilot-20261108-01/scene.json",
        "old_checkpoint": str(source / "train/checkpoints/020000/pretrained_model"),
        "base_model": str(base),
    }
    validation = [22, 23, 38, 40, 47, 60, 71, 72, 82, 84]
    status = {
        "supervisor_pid": os.getpid(),
        "stage": "preparing",
        "target_steps": 20000,
        "root": str(root),
        "calibration_passed": False,
    }

    def update(stage, **extra):
        status.update(stage=stage, updated_at_utc=datetime.datetime.now(datetime.UTC).isoformat(), **extra)
        save(root / "status.json", status)
        print(json.dumps(status), flush=True)

    def verify_inputs(request):
        for path, expected in request["code_sha256"].items():
            if digest(code / path) != expected:
                raise ValueError("Frozen code changed: " + path)
        for path, expected in request["source_sha256"].items():
            if digest(path) != expected:
                raise ValueError("Source input changed: " + Path(path).name)

    def stage(stage_name, script, script_args, logname):
        verify_inputs(request)
        command = [
            "/root/.local/bin/uv",
            "run",
            "--no-project",
            "--offline",
            "--python",
            sys.executable,
            "python",
            "-u",
            str(code / script),
            *map(str, script_args),
        ]
        with (root / logname).open("a") as log:
            child = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            update(stage_name, child_pid=child.pid, log=str(root / logname))
            result = child.wait()
        update(stage_name + "_finished", child_pid=None, returncode=result)
        if result:
            raise RuntimeError(f"{stage_name} returned {result}; inspect {logname}")

    try:
        # Never start beside a surviving child from an interrupted supervisor.
        for process in Path("/proc").iterdir():
            if not process.name.isdigit():
                continue
            try:
                command = (process / "cmdline").read_bytes().split(bytes([0]))
                if any(
                    part == str(code / script).encode()
                    for script in ["train_mixed.py", "compare_mixed.py"]
                    for part in command
                ):
                    raise RuntimeError("A previous training/comparison process is still alive")
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
        request_path = root / "request.json"
        if request_path.exists():
            if not args.resume:
                raise FileExistsError("Existing run requires --resume")
            request = json.loads(request_path.read_text())
            verify_inputs(request)
        else:
            if args.resume:
                raise FileNotFoundError("Cannot resume an unprepared run")
            numeric_sources = []
            for key in ["real_root", "sim_root"]:
                numeric_sources += list((Path(paths[key]) / "data").rglob("*.parquet"))
                numeric_sources.append(Path(paths[key]) / "meta/info.json")
            numeric_sources += [
                Path(paths["scene"]),
                Path(paths["old_checkpoint"]) / "model.safetensors",
                base / "model.safetensors",
            ]
            asset = Path(json.loads(Path(paths["scene"]).read_text())["asset"])
            numeric_sources.append(asset)
            verification = json.loads((source / "dataset_verification.json").read_text())
            assert (
                verification["passed"]
                and verification["episodes"] == verification["unique_physics_hashes"] == 100
            )
            assert verification["appearance_frames"] == verification["frames"] == 55175
            request = {
                "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "paths": paths,
                "real_training_episodes": sorted(set(range(88)) - set(validation)),
                "real_validation_episodes": validation,
                "real_holdout_seed": 20261110,
                "sim_training_episodes": list(range(100)),
                "batch_size": 8,
                "real_per_batch": 4,
                "sim_per_batch": 4,
                "training_steps": 20000,
                "training_seed": 91000,
                "normalization": "50/50 domain-balanced moments of real78 + sim100 only",
                "evaluation": "all 3104 frames of fixed real10 every 1000 steps; paired physical-unit action error after training",
                "initialization": "fresh fine-tuning of the same cached smolvla_base snapshot, not the old fine-tuned optimizer",
                "authorization": "User requested retraining on simulated and real trajectories with exactly half of every batch from each source",
                "calibration_passed": False,
                "push_to_hub": False,
                "gpu_release_requested": False,
                "code_sha256": {p.name: digest(p) for p in code.glob("*.py")},
                "source_sha256": {str(p): digest(p) for p in numeric_sources},
            }
            save(request_path, request)
            cfg = json.loads((source / "train_config.json").read_text())
            cfg.update(
                output_dir=str(root / "train"),
                job_name="smolvla_real78_sim100_balanced",
                resume=False,
                checkpoint_path=None,
            )
            cfg["dataset"].update(
                repo_id="local/so101_real78_sim100_balanced",
                root=paths["sim_root"],
                episodes=None,
                exclude_episodes=None,
                eval_split=10 / 88,
            )
            cfg["policy"].update(pretrained_path=str(base), repo_id=None, push_to_hub=False)
            save(root / "train_config.json", cfg)
        update("prepared", real_training_episodes=78, real_validation_episodes=10, sim_training_episodes=100)
        last = root / "train/checkpoints/last"
        step_file = last / "training_state/training_step.json"
        step = json.loads(step_file.read_text())["step"] if step_file.exists() else 0
        if step < 20000:
            if (root / "train").exists() and step == 0:
                raise RuntimeError("Training output exists without checkpoint; preserve it for inspection")
            config = last / "pretrained_model/train_config.json" if step else root / "train_config.json"
            train_args = ["--root", root, "--config", config]
            if step:
                train_args.append("--resume")
            stage("training", "train_mixed.py", train_args, "train.log")
        final = root / "train/checkpoints/020000/pretrained_model"
        step = json.loads((root / "train/checkpoints/020000/training_state/training_step.json").read_text())[
            "step"
        ]
        assert step == 20000 and (final / "model.safetensors").stat().st_size > 0
        balance = json.loads((root / "batch_balance.json").read_text())
        assert (
            balance["completed_optimizer_steps"] == 20000
            and balance["real_samples"] == balance["sim_samples"] == 80000
        )
        assert balance["violations"] == 0
        log = (root / "train.log").read_text(errors="replace")
        assert "End of training" in log
        evaluations = re.findall(r"step 20000: eval_loss=([0-9.]+)", log)
        assert evaluations
        update(
            "comparison",
            checkpoint_step=20000,
            final_eval_loss=float(evaluations[-1]),
            final_checkpoint=str(final),
        )
        for label, checkpoint in [("old_sim_only", paths["old_checkpoint"]), ("new_mixed", final)]:
            result = root / f"{label}_physical_metrics.json"
            if not result.exists():
                stage(
                    "comparison_" + label,
                    "compare_mixed.py",
                    ["--root", root, "--checkpoint", checkpoint, "--label", label],
                    "comparison.log",
                )
        old = json.loads((root / "old_sim_only_physical_metrics.json").read_text())
        new = json.loads((root / "new_mixed_physical_metrics.json").read_text())
        assert old["passed"] and new["passed"] and old["frames"] == new["frames"] == 3104
        assert old["real_validation_episodes"] == new["real_validation_episodes"] == validation
        keys = ["arm_mean_mae_degrees", "gripper_mae_percentage_points", "common_training_scale_action_mse"]
        comparison = {
            "passed": True,
            "fixed_real_holdout": validation,
            "frames": 3104,
            "new_native_eval_loss": float(evaluations[-1]),
            "native_loss_scales_changed": True,
            "comparison": {k: {"old": old[k], "new": new[k], "improved": new[k] < old[k]} for k in keys},
            "calibration_passed": False,
            "not_a_closed_loop_success_rate": True,
        }
        save(root / "comparison.json", comparison)
        verify_inputs(request)
        update("complete", passed=True, child_pid=None, comparison=str(root / "comparison.json"))
    except BaseException as error:
        update("failed", error_type=type(error).__name__, error=str(error), child_pid=status.get("child_pid"))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
