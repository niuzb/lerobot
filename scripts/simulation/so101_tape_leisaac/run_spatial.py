"""Generate, render and verify a small spatial-conditioning batch on the existing worker."""

import argparse
import datetime as dt
import fcntl
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

from common import write_json
from spatial import augmented_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Output of prepare_spatial.py")
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--leisaac-root", type=Path, required=True)
    parser.add_argument("--leisaac-assets", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True, help="Existing LeRobot Python environment")
    parser.add_argument("--uv", type=Path, default=Path("/root/.local/bin/uv"))
    parser.add_argument("--max-attempts", type=int, default=25)
    parser.add_argument("--framewise-textures", action="store_true")
    parser.add_argument("--texture-seed", type=int, default=81000)
    parser.add_argument("--texture-count", type=int, default=16)
    parser.add_argument("--texture-subframes", type=int, default=4)
    args = parser.parse_args()
    lock = (args.root / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.root / "status.json").exists():
        raise FileExistsError("Use a new batch directory; do not overwrite an earlier run")
    plan = json.loads((args.root / "spatial_plan.json").read_text())
    source = Path(plan["source_run"])
    source_scene = json.loads((source / "scene.json").read_text())
    source_cameras = json.loads((source / "camera_views.json").read_text())
    python = [str(args.uv), "run", "--no-project", "--offline", "--python", str(args.python), "python"]
    code = args.root / "code"
    state = {
        "supervisor_pid": os.getpid(),
        "calibration_passed": False,
        "human_review_pending": True,
        "training_started": False,
        "gpu_release_requested": False,
        "completed_conditions": [],
        "framewise_textures": args.framewise_textures,
    }
    os.environ.update(
        NO_COLOR="1", TERM="dumb", LEISAAC_CPU_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"
    )

    def status(stage, **values):
        state.update(stage=stage, updated_at_utc=dt.datetime.now(dt.UTC).isoformat(), **values)
        temporary = args.root / "status.json.tmp"
        write_json(temporary, state)
        temporary.replace(args.root / "status.json")
        print(json.dumps(state), flush=True)

    def execute(stage, command, log_path, timeout):
        with log_path.open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status(stage, child_pid=child.pid)
            try:
                result = child.wait(timeout=timeout)
                if result:
                    raise RuntimeError(f"{stage} exited with {result}; inspect {log_path}")
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                raise
        status(stage + "_finished", child_pid=None)

    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        status("starting")
        for number, condition in enumerate(plan["conditions"]):
            run = Path(condition["run"])
            status("preparing", condition=number, table_height_delta_m=condition["height_delta_m"])
            # Leave headroom for the already-running batch; never stop it.
            usage = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
            limit = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
            deadline = time.monotonic() + 1800
            while (
                usage.exists()
                and limit.exists()
                and int(limit.read_text()) - int(usage.read_text()) < 12 * 1024**3
            ):
                status("waiting_for_memory")
                if time.monotonic() > deadline:
                    raise RuntimeError("Insufficient memory headroom for the isolated spatial pilot")
                time.sleep(10)
            common = [
                "--run",
                str(run),
                "--leisaac-root",
                str(args.leisaac_root),
                "--leisaac-assets",
                str(args.leisaac_assets),
            ]
            physics, rendering = run / "physics", run / "render"
            execute(
                "collect",
                [
                    str(args.sim),
                    str(code / "collect.py"),
                    *common,
                    "--output",
                    str(physics),
                    "--count",
                    "1",
                    "--max-attempts",
                    str(args.max_attempts),
                    "--seed",
                    str(condition["physical_seed_start"]),
                ],
                run / "collect.log",
                1800,
            )
            generated = json.loads((physics / "generation_manifest.json").read_text())
            if not generated["passed"] or generated["accepted"] != 1:
                raise RuntimeError("Physical success evidence is incomplete")
            selected_camera = selected_audit = None
            for candidate in range(8):
                camera = run / f"camera_candidate_{candidate:02d}.json"
                if candidate == 0:
                    camera.write_bytes((run / "camera_views.json").read_bytes())
                else:
                    revised = augmented_scene(
                        source_scene,
                        source_cameras,
                        height_delta_m=condition["height_delta_m"],
                        seed=condition["augmentation_seed"] + candidate * 10000,
                        **plan.get("augmentation_bounds", {}),
                    )
                    write_json(
                        camera,
                        {
                            "revision": "spatial_v1",
                            "cameras": revised["cameras"],
                            "augmentation": revised["spatial_augmentation"],
                            "calibration_passed": False,
                            "human_review_pending": True,
                        },
                    )
                audit = run / f"framing_{candidate:02d}.json"
                execute(
                    "framing",
                    [
                        *python,
                        str(code / "view_audit.py"),
                        "--scene",
                        str(run / "scene.json"),
                        "--camera-config",
                        str(camera),
                        "--physics",
                        str(physics),
                        "--output",
                        str(audit),
                    ],
                    run / f"framing_{candidate:02d}.log",
                    300,
                )
                if json.loads(audit.read_text())["geometric_framing_passed"]:
                    selected_camera, selected_audit = camera, audit
                    break
            if selected_camera is None:
                raise RuntimeError(
                    "No camera pose passed complete-object framing within the configured bounds"
                )
            execute(
                "render",
                [
                    str(args.sim),
                    str(code / "render.py"),
                    *common,
                    "--physics",
                    str(physics),
                    "--camera-config",
                    str(selected_camera),
                    "--output",
                    str(rendering),
                    *(
                        [
                            "--framewise-textures",
                            "--texture-seed",
                            str(args.texture_seed + number),
                            "--texture-count",
                            str(args.texture_count),
                            "--texture-subframes",
                            str(args.texture_subframes),
                        ]
                        if args.framewise_textures
                        else []
                    ),
                ],
                run / "render.log",
                1800,
            )
            repo_id = f"local/so101_tape_spatial_20261109_{number:02d}"
            execute(
                "export",
                [
                    *python,
                    str(code / "export.py"),
                    "--run",
                    str(run),
                    "--physics",
                    str(physics),
                    "--render",
                    str(rendering),
                    "--output",
                    str(run / "dataset"),
                    "--repo-id",
                    repo_id,
                    "--spatial-features",
                ],
                run / "export.log",
                1200,
            )
            verification = json.loads((run / "dataset_verification.json").read_text())
            if not verification["passed"] or verification["training_batch_state_shape"] != [2, 25]:
                raise RuntimeError("Spatial dataset verification failed")
            execute(
                "review",
                [
                    *python,
                    str(code / "prepare_review.py"),
                    "--physics",
                    str(physics),
                    "--render",
                    str(rendering),
                    "--framing-audit",
                    str(selected_audit),
                    "--output",
                    str(run / "review"),
                ],
                run / "review.log",
                600,
            )
            state["completed_conditions"].append(
                {
                    "condition": number,
                    "height_delta_m": condition["height_delta_m"],
                    "dataset": str(run / "dataset"),
                    "frames": verification["total_frames"],
                    "camera_config": str(selected_camera),
                }
            )
            status("condition_complete")
        status("complete", child_pid=None, passed=True)
    except BaseException as error:
        traceback.print_exc()
        status("failed", child_pid=None, passed=False, error=repr(error))
        raise


if __name__ == "__main__":
    main()
