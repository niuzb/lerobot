"""Persistently generate 100 independent augmented trajectories, then train SmolVLA."""

import argparse
import datetime as dt
import fcntl
import json
import os
import signal
import subprocess
import traceback
from pathlib import Path

from common import sha256, write_json
from merge_batch import validate_sources
from spatial import augmented_scene


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def archive_partial(path):
    if not path.exists():
        return
    for index in range(1, 1000):
        target = path.with_name(f"{path.name}_incomplete_{index:03d}")
        if not target.exists():
            path.rename(target)
            return
    raise RuntimeError("Too many incomplete outputs")


class RejectedTrajectoryError(Exception):
    """Ordinary physical/framing rejection; never enters the training set."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--leisaac-root", type=Path, required=True)
    parser.add_argument("--leisaac-assets", type=Path, required=True)
    parser.add_argument("--uv", type=Path, default=Path("/root/.local/bin/uv"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root, code = args.root.resolve(), args.root.resolve() / "code"
    lock = (root / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request = read_json(root / "request.json")
    for name, expected in request.get("code_sha256", {}).items():
        if sha256(code / name) != expected:
            raise ValueError(f"Code snapshot changed: {name}")
    if request.get("requested_independent_trajectories") != 100 or not request.get(
        "automatic_training_authorized"
    ):
        raise ValueError("Require the explicit 100-trajectory automatic-training request")
    if not read_json(root / "preflight.json").get("passed"):
        raise ValueError("SmolVLA forward/backward preflight must pass before this long run")
    if (root / "status.json").exists() and not args.resume:
        raise FileExistsError("Existing pipeline; use --resume instead of starting a duplicate")
    ledger = read_json(root / "ledger.json") or {"accepted": [], "rejected": [], "candidate": 0}
    state = {
        "supervisor_pid": os.getpid(),
        "requested": 100,
        "training_started": False,
        "automatic_training_authorized": True,
        "calibration_passed": False,
        "sample_method_approved_by_user": True,
        "gpu_release_requested": False,
    }
    python = [str(args.uv), "run", "--no-project", "--offline", "--python", str(args.python), "python"]
    scene = read_json(args.source_run / "scene.json")
    cameras = read_json(args.source_run / "camera_views.json")
    os.environ.update(
        NO_COLOR="1",
        TERM="dumb",
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        LEISAAC_CPU_THREADS="4",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        CUDA_VISIBLE_DEVICES="0",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )

    def status(stage, **values):
        state.update(
            stage=stage,
            accepted=len(ledger["accepted"]),
            rejected=len(ledger["rejected"]),
            updated_at_utc=dt.datetime.now(dt.UTC).isoformat(),
            **values,
        )
        atomic_json(root / "status.json", state)
        print(json.dumps(state), flush=True)

    def execute(stage, command, log_path, timeout):
        if log_path.exists():
            for index in range(1, 1000):
                alternate = log_path.with_name(f"{log_path.stem}_retry_{index:03d}{log_path.suffix}")
                if not alternate.exists():
                    log_path = alternate
                    break
        with log_path.open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if stage == "training":
                state["training_started"] = True
            status(stage, child_pid=child.pid, log=str(log_path))
            try:
                result = child.wait(timeout=timeout)
                if result:
                    raise RuntimeError(f"{stage} exited with {result}; {log_path}")
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

    def interrupt(signum, frame):
        raise InterruptedError(f"Signal {signum}; progress was preserved for --resume")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    def sample(candidate, index):
        run = root / "samples" / f"candidate_{candidate:04d}"
        height = (0.02, 0.0, -0.02)[index % 3]
        if not run.exists():
            run.mkdir(parents=True)
            config = augmented_scene(scene, cameras, height_delta_m=height, seed=900000 + candidate)
            write_json(run / "scene.json", config)
            for name in ("assets", "site-packages"):
                (run / name).symlink_to((args.source_run / name).resolve(), target_is_directory=True)
        common = [
            "--run",
            str(run),
            "--leisaac-root",
            str(args.leisaac_root),
            "--leisaac-assets",
            str(args.leisaac_assets),
        ]
        physical = run / "physics"
        generated = read_json(physical / "generation_manifest.json")
        if not generated.get("passed"):
            archive_partial(physical)
            try:
                execute(
                    "collect",
                    [
                        str(args.sim),
                        str(code / "collect.py"),
                        *common,
                        "--output",
                        str(physical),
                        "--count",
                        "1",
                        "--max-attempts",
                        "50",
                        "--seed",
                        str(1000000 + candidate * 100),
                    ],
                    run / "collect.log",
                    2400,
                )
            except RuntimeError:
                failure = read_json(physical / "failure.json").get("error", "")
                if "Requested physical successes not reached" in failure:
                    raise RejectedTrajectoryError("physical success not reached") from None
                raise
            generated = read_json(physical / "generation_manifest.json")
        if not generated.get("passed") or generated.get("accepted") != 1:
            raise ValueError("Incomplete physical evidence")
        attempt, summary = next((i, item) for i, item in enumerate(generated["attempts"]) if item["success"])
        if sha256(physical / f"attempt_{attempt:04d}.npz") != summary["npz_sha256"]:
            raise ValueError("Physical recording changed")
        selected = None
        for choice in range(16):
            camera = run / f"camera_{choice:02d}.json"
            if not camera.exists():
                revised = augmented_scene(
                    scene, cameras, height_delta_m=height, seed=900000 + candidate + choice * 10000
                )
                write_json(
                    camera,
                    {
                        "revision": "train100_spatial",
                        "cameras": revised["cameras"],
                        "augmentation": revised["spatial_augmentation"],
                        "calibration_passed": False,
                    },
                )
            audit = run / f"framing_{choice:02d}.json"
            if not audit.exists():
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
                        str(physical),
                        "--output",
                        str(audit),
                    ],
                    run / f"framing_{choice:02d}.log",
                    300,
                )
            if read_json(audit).get("geometric_framing_passed"):
                selected = camera
                break
        if selected is None:
            raise RejectedTrajectoryError("no camera candidate passed full-frame framing")
        rendering = run / "render"
        if not read_json(rendering / "render_manifest.json").get("passed"):
            archive_partial(rendering)
            try:
                execute(
                    "render",
                    [
                        str(args.sim),
                        str(code / "render.py"),
                        *common,
                        "--physics",
                        str(physical),
                        "--camera-config",
                        str(selected),
                        "--output",
                        str(rendering),
                        "--framewise-textures",
                        "--texture-seed",
                        "81000",
                        "--texture-schedule-seed",
                        str(1100000 + candidate),
                        "--texture-count",
                        "16",
                        "--texture-subframes",
                        "4",
                    ],
                    run / "render.log",
                    1800,
                )
            except RuntimeError:
                if "Texture group has no verified visible effect" in read_json(
                    rendering / "failure.json"
                ).get("error", ""):
                    raise RejectedTrajectoryError(
                        "appearance not visibly verified in this camera configuration"
                    ) from None
                raise
        verification = read_json(run / "dataset_verification.json")
        if not verification.get("passed"):
            archive_partial(run / "dataset")
            execute(
                "export",
                [
                    *python,
                    str(code / "export.py"),
                    "--run",
                    str(run),
                    "--physics",
                    str(physical),
                    "--render",
                    str(rendering),
                    "--output",
                    str(run / "dataset"),
                    "--repo-id",
                    f"local/tape100_20261109_{index:03d}",
                    "--spatial-features",
                ],
                run / "export.log",
                1800,
            )
            verification = read_json(run / "dataset_verification.json")
        if not verification.get("passed") or len(verification.get("episodes", [])) != 1:
            raise ValueError("Incomplete episode export")
        if verification["framewise_appearance"]["frames"] != verification["total_frames"]:
            raise ValueError("Appearance does not cover every frame")
        return {
            "index": index,
            "candidate": candidate,
            "run": str(run),
            "height_delta_m": height,
            "physical_seed": summary["seed"],
            "physics_npz_sha256": summary["npz_sha256"],
            "frames": verification["total_frames"],
            "camera_config": str(selected),
            "texture_schedule_seed": 1100000 + candidate,
        }

    try:
        status("starting")
        while len(ledger["accepted"]) < 100:
            candidate = ledger["candidate"]
            if candidate >= 300:
                raise RuntimeError("Candidate budget exhausted; training was not started")
            status("preparing", candidate=candidate, episode_index=len(ledger["accepted"]))
            try:
                entry = sample(candidate, len(ledger["accepted"]))
                if any(
                    item["physical_seed"] == entry["physical_seed"]
                    or item["physics_npz_sha256"] == entry["physics_npz_sha256"]
                    for item in ledger["accepted"]
                ):
                    raise ValueError("Duplicate physical trajectory cannot count toward 100")
                ledger["accepted"].append(entry)
            except RejectedTrajectoryError as error:
                ledger["rejected"].append({"candidate": candidate, "reason": str(error)})
            ledger["candidate"] = candidate + 1
            atomic_json(root / "ledger.json", ledger)
            status("trajectory_verified")
            if len(ledger["accepted"]) == 2 and not read_json(root / "merge_smoke_verification.json").get(
                "passed"
            ):
                archive_partial(root / "merge_smoke")
                execute(
                    "merge_preflight",
                    [
                        *python,
                        str(code / "merge_batch.py"),
                        "--ledger",
                        str(root / "ledger.json"),
                        "--output",
                        str(root / "merge_smoke"),
                        "--repo-id",
                        "local/tape100_merge_smoke",
                        "--expected",
                        "2",
                    ],
                    root / "merge_smoke.log",
                    1800,
                )
        validate_sources(ledger["accepted"], 100)
        if not read_json(root / "dataset_verification.json").get("passed"):
            archive_partial(root / "dataset")
            execute(
                "merge",
                [
                    *python,
                    str(code / "merge_batch.py"),
                    "--ledger",
                    str(root / "ledger.json"),
                    "--output",
                    str(root / "dataset"),
                    "--repo-id",
                    "local/so101_tape100_framewise_20261109",
                    "--expected",
                    "100",
                ],
                root / "merge.log",
                7200,
            )
        merged = read_json(root / "dataset_verification.json")
        if not merged.get("passed") or merged.get("episodes") != 100:
            raise ValueError("Exactly 100 merged verified episodes are required before training")
        config = root / "train_config.json"
        resume_arguments = []
        if (root / "train").exists():
            last = root / "train/checkpoints/last"
            if last.exists() and last.resolve().name.isdigit() and int(last.resolve().name) == 20000:
                status("complete", child_pid=None, passed=True, final_checkpoint=str(last.resolve()))
                return
            checkpoint = root / "train/checkpoints/last/pretrained_model/train_config.json"
            if not checkpoint.exists():
                raise RuntimeError(
                    "Incomplete training output without a checkpoint; preserve it for inspection"
                )
            config = checkpoint
            resume_arguments = ["--resume", "--stats-output", str(root / "training_stats.json")]
        execute(
            "training",
            [*python, str(code / "train_batch.py"), "--config", str(config), *resume_arguments],
            root / "train.log",
            48 * 3600,
        )
        last = root / "train/checkpoints/last"
        if not last.exists() or int(last.resolve().name) != 20000:
            raise RuntimeError("Training exited without the final 20000-step checkpoint")
        status("complete", child_pid=None, passed=True, final_checkpoint=str(last.resolve()))
    except BaseException as error:
        traceback.print_exc()
        status("failed", child_pid=None, passed=False, error=repr(error))
        raise


if __name__ == "__main__":
    main()
