"""Require independent reload verification before full training and final LIBERO evaluation."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .libero_full_utils import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--cpu-affinity")
    parser.add_argument("--max-updates", type=int, default=20000)
    parser.add_argument("--wait-for-lock", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock = (root / "launch.lock").open("w")
    if args.max_updates <= 4 or args.max_updates > 20000:
        raise ValueError("Training target must be between 5 and 20,000 optimizer updates")
    if args.wait_for_lock:
        write_json(root / "queued_run.json", {"max_updates": args.max_updates,
                                              "waiting_for_preflight_lock": True})
    fcntl.flock(lock, fcntl.LOCK_EX if args.wait_for_lock else fcntl.LOCK_EX | fcntl.LOCK_NB)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
               "--module", "examples.dreamzero.train_libero_distributed", "--root", str(root),
               "--reference", str(args.reference)]
    if args.cpu_affinity:
        command += ["--cpu-affinity", args.cpu_affinity]

    def execute(stage, executable):
        write_json(root / "stage.json", {"stage": stage})
        with (root / f"{stage}_launcher.log").open("a") as output:
            result = subprocess.run(executable, stdout=output, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
        if result.returncode:
            write_json(root / "stage.json", {"stage": stage + "_failed", "exit_code": result.returncode})
            raise SystemExit(result.returncode)

    preflight = root / "PREFLIGHT_PASSED.json"
    if not preflight.exists():
        completed = root / "smoke_complete.json"
        if not completed.exists() or json.loads(completed.read_text())["update"] != 4:
            execute("smoke", command + ["--smoke", "--max-updates", "4"])
        execute("reload", command + ["--smoke", "--resume", "update_000004", "--max-updates", "5"])
        reloaded = [json.loads((root / f"reload_rank{rank}.json").read_text()) for rank in (0, 1)]
        continuation = json.loads(completed.read_text())
        if not all(row["passed"] and row["update"] == 4 for row in reloaded):
            raise RuntimeError("Both ranks must restore update 4 exactly")
        if continuation["update"] != 5 or continuation["initial_update"] != 4:
            raise RuntimeError("Independent process did not continue to update 5")
        write_json(preflight, {"ranks": reloaded, "continuation": continuation})
    if args.preflight_only:
        write_json(root / "stage.json", {"stage": "preflight_passed"})
        return
    latest = json.loads((root / "latest.json").read_text())
    if latest["update"] >= args.max_updates:
        raise ValueError("Requested update target is not ahead of the latest checkpoint")
    write_json(root / "active_run.json", {"target_update": args.max_updates,
        "resume_tag": latest["tag"], "stop_after_evaluation": True})
    execute("training", command + ["--resume", latest["tag"], "--max-updates", str(args.max_updates)])
    evaluation = [sys.executable, "-m", "examples.dreamzero.evaluate_libero_distributed",
                  "--root", str(root), "--update", str(args.max_updates)]
    if args.max_updates < 20000:
        evaluation += ["--selection-only"]
    execute("evaluation", evaluation)
    write_json(root / "stage.json", {"stage": "completed" if args.max_updates == 20000 else "segment_completed",
                                      "update": args.max_updates})


if __name__ == "__main__":
    main()
