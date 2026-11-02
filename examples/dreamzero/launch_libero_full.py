"""Run the four-update smoke, separate reload, production and final eval sequentially.

No process in this launcher owns or releases the Merlin worker. The independent
existing keepalive shell remains in place on success, failure and interruption.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--optimizer-mode", choices=("zero3_cpu_adam", "zero3_adamw8bit"))
    parser.add_argument("--offload-ratio", type=float)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Stop after four updates and independent checkpoint continuation",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not (root / "READY").is_file():
        raise FileNotFoundError("Data preparation has not passed")
    # Exclude competing launchers while leaving the resource keepalive untouched.
    import fcntl

    lock = (root / "launch.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        "--module",
        "examples.dreamzero.train_libero_full",
        "--root",
        str(root),
    ]
    if args.optimizer_mode is not None:
        command += ["--optimizer-mode", args.optimizer_mode]
    if args.offload_ratio is not None:
        command += ["--offload-ratio", str(args.offload_ratio)]
    checkpoint = root / "checkpoints/update_000004"

    def execute(stage, extra):
        (root / "stage").write_text(stage + "\n")
        with (root / f"{stage}.log").open("a") as log:
            result = subprocess.run(
                command + extra,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        if result.returncode:
            (root / "stage").write_text(f"{stage}_failed_gpu_retained\n")
            raise SystemExit(result.returncode)

    execute("smoke", ["--smoke", "--max-updates", "4"])
    execute("reload", ["--resume-from", str(checkpoint), "--verify-resume", "--max-updates", "5"])
    reload_result = json.loads((root / "resume_verification.json").read_text())
    if not reload_result["passed"] or reload_result["continued_update"] != 5:
        raise RuntimeError("Independent reload/continuation verification failed")
    evidence = {
        "smoke": json.loads((root / "smoke_training.json").read_text()),
        "reload": reload_result,
        "policy_config_sha256": hashlib.sha256((root / "policy_config/config.json").read_bytes()).hexdigest(),
        "runtime_hashes": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("runtime.json", "deepspeed.json")
        },
    }
    (root / "SMOKE_PASSED.json").write_text(json.dumps(evidence, indent=2) + "\n")
    if args.preflight_only:
        (root / "stage").write_text("preflight_passed\n")
        return
    execute("training", ["--resume-from", str(checkpoint)])
    execute("final_evaluation", ["--final-eval"])
    (root / "stage").write_text("completed_gpu_retained\n")


if __name__ == "__main__":
    main()
