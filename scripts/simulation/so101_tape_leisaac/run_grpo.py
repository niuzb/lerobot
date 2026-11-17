"""Coordinate isolated live simulation and one policy actor on the existing A30."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, default=4494099)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--curriculum", type=Path)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--expected-parent-sha256")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--epoch-offset", type=int, default=0)
    parser.add_argument("--restore-curriculum", action="store_true")
    args = parser.parse_args()
    if (args.parent_run is None) != (args.curriculum is None):
        parser.error("Parent run and audited curriculum must be supplied together")
    if not args.parent_run and (
        args.parent_checkpoint or args.expected_parent_sha256 or args.restore_curriculum
    ):
        parser.error("Checkpoint continuation options require a parent run")
    if args.epochs <= 0 or args.epoch_offset < 0 or args.epochs + args.epoch_offset != 50:
        parser.error("The continuation must use the remaining budget through global epoch 50")
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "request.json").exists():
        raise FileExistsError("Existing GRPO attempt requires inspection; no blind overwrite")
    checkpoint = Path(
        "/root/data/leisaac/runs/tape-mixed-20261110-01/train/checkpoints/020000/pretrained_model"
    )
    expected_source = "4db2d12ed38e10b816e90d8f52ad59e232d5392237f8e679f5d2e0d92c6eb808"
    parent_state = None
    if args.parent_run:
        parent_checkpoint = args.parent_checkpoint or args.parent_run / "checkpoints/final"
        checkpoint = parent_checkpoint / "pretrained_model"
        expected_source = (
            args.expected_parent_sha256 or "3696737dc48fffe4d28820c5ca946b0254ad58beffe82fd7ccb5fc636b15f80a"
        )
        parent_state = json.loads((parent_checkpoint / "training_state.json").read_text())
        assert parent_state.get("total_optimizer_steps", parent_state["optimizer_steps"]) > 0
        assert (parent_checkpoint / "optimizer.pt").stat().st_size > 0
        if args.epoch_offset:
            assert parent_state["epoch"] == args.epoch_offset
        curriculum = json.loads(args.curriculum.read_text())
        assert digest(curriculum["source_physics"]) == curriculum["source_physics_sha256"]
        if args.restore_curriculum:
            assert parent_state["curriculum_state"] is not None
            assert digest(args.curriculum) == parent_state["curriculum_sha256"]
    assert digest(checkpoint / "model.safetensors") == expected_source
    code = root / "code"
    socket_directory = tempfile.TemporaryDirectory(prefix="grpo-")
    socket_path = Path(socket_directory.name) / "actor.sock"
    request = {
        "coordinator_pid": os.getpid(),
        "worker_id": args.worker_id,
        "source_checkpoint": str(checkpoint),
        "source_model_sha256": digest(checkpoint / "model.safetensors"),
        "method": "SmolVLA Flow-SDE GRPO adaptation; not discrete-token paper reproduction",
        "paper": "https://arxiv.org/abs/2603.22876v2",
        "sde_source": "FlowVLA-RL 50e73afc07212f48bed370ea4c88a3131d6f914a; MIT",
        "group_size": 8,
        "epochs": args.epochs,
        "epoch_offset": args.epoch_offset,
        "target_global_epoch": 50,
        "groups_per_epoch": 1,
        "planned_online_rl_rollouts": args.epochs * 8,
        "baseline_ode_rollouts": 2,
        "group_accuracy_filter": [0.1, 0.9],
        "advantage": "(binary_reward - group_mean)/population_std",
        "kl_beta": 0,
        "learning_rate": 5e-6,
        "clip_low": 0.2,
        "clip_high": 0.28,
        "max_gradient_norm": 1,
        "ppo_passes_per_eligible_group": 2,
        "microbatch_chunks": 1,
        "sampling_denoise_steps": 10,
        "training_denoise_steps_per_chunk": 1,
        "training_timestep_selection": "uniform; stochastic estimate of ten-step mean objective",
        "precision": "FP32 collection and rescoring",
        "sde_eta": 0.05,
        "density_coordinates": [50, 32],
        "executed_actions_per_chunk": [30, 6],
        "all_latent_coordinates_scored": True,
        "trainable": "action expert and action/time projections; VLM and state prefix projection frozen",
        "reward": "existing contact/grasp/lift/carry/release/containment/support/30-stable-tick physical gate",
        "max_control_steps": 600,
        "group_resets": "same initial seed, object pose/friction and texture sequence; numerical reset fingerprint checked",
        "normalization": "unchanged mixed-SFT saved processors; no real-data use or refit",
        "real_validation_episodes": [22, 23, 38, 40, 47, 60, 71, 72, 82, 84],
        "calibration_passed": False,
        "new_gpu_requested": False,
        "code_sha256": {str(p.relative_to(code)): digest(p) for p in code.rglob("*.py")},
        "scene_sha256": digest(root / "scene.json"),
    }
    if args.parent_run:
        request.update(
            parent_run=str(args.parent_run),
            parent_training_state=parent_state,
            parent_checkpoint=str(parent_checkpoint),
            parent_optimizer_sha256=digest(parent_checkpoint / "optimizer.pt"),
            restore_curriculum=args.restore_curriculum,
            curriculum=curriculum,
            curriculum_sha256=digest(args.curriculum),
            reward_scope="suffix completion from audited physical reset contexts; stage successes are not full-task successes",
        )
    (root / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    env = {
        **os.environ,
        "NO_COLOR": "1",
        "TERM": "dumb",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "LEISAAC_CPU_THREADS": "4",
    }
    actor_command = [
        "/root/.local/bin/uv",
        "run",
        "--no-project",
        "--offline",
        "--python",
        sys.executable,
        "python",
        "-u",
        str(code / "grpo_actor.py"),
        "--root",
        str(root),
        "--checkpoint",
        str(checkpoint),
        "--socket",
        str(socket_path),
        "--epochs",
        "50",
    ]
    native_command = [
        "/root/data/leisaac/runs/a30-probe-20261108-01/run_native.sh",
        str(code / "grpo_env.py"),
        "--run",
        str(root),
        "--socket",
        str(socket_path),
        "--leisaac-root",
        "/root/code/leisaac-merlin",
        "--leisaac-assets",
        "/root/data/leisaac/assets",
        "--epochs",
        str(args.epochs),
        "--epoch-offset",
        str(args.epoch_offset),
    ]
    if args.parent_run:
        actor_command += [
            "--optimizer-state",
            str(parent_checkpoint / "optimizer.pt"),
            "--parent-training-state",
            str(parent_checkpoint / "training_state.json"),
            "--checkpoint-groups",
            "--curriculum-sha256",
            digest(args.curriculum),
        ]
        native_command += ["--curriculum", str(args.curriculum)]
        if args.restore_curriculum:
            native_command += ["--restore-curriculum-state", str(parent_checkpoint / "training_state.json")]
    children = []

    def coordinator_status(stage, **values):
        path = root / "coordinator_status.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"stage": stage, "pid": os.getpid(), "at_unix": time.time(), **values}, indent=2)
            + "\n"
        )
        tmp.replace(path)

    try:
        actor = subprocess.Popen(
            actor_command,
            stdout=(root / "actor.log").open("w"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
        children.append(actor)
        coordinator_status("starting_actor", actor_uv_pid=actor.pid)
        deadline = time.monotonic() + 300
        while not socket_path.exists():
            if actor.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Actor failed to become ready; inspect actor.log")
            time.sleep(1)
        native = subprocess.Popen(
            native_command,
            stdout=(root / "simulation.log").open("w"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
        children.append(native)
        coordinator_status("live_rollouts", actor_uv_pid=actor.pid, native_pid=native.pid)
        while native.poll() is None:
            if actor.poll() is not None and not (
                actor.returncode == 0 and (root / "actor_finished.json").exists()
            ):
                raise RuntimeError("Actor exited before native rollouts completed")
            time.sleep(3)
        if native.returncode or (root / "environment_failure.json").exists():
            raise RuntimeError("Live simulation failed; inspect simulation.log")
        actor.wait(timeout=60)
        status = json.loads((root / "status.json").read_text())
        if status["stage"] in ["curriculum_stalled", "no_learning_signal"]:
            coordinator_status(status["stage"], native_status=status)
            return
        assert (
            status["stage"] == "complete"
            and status["completed_epochs"] == 50
            and status["optimizer_steps"] > 0
        )
        assert digest(checkpoint / "model.safetensors") == request["source_model_sha256"]
        coordinator_status("complete", native_status=status)
    except BaseException as error:
        coordinator_status("failed", error=repr(error))
        raise
    finally:
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        if socket_path.exists():
            socket_path.unlink()
        socket_directory.cleanup()
        lock.close()


if __name__ == "__main__":
    main()
