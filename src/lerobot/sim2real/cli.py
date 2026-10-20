"""SO-101 task entrypoint. Hardware access is isolated to explicit read-only capture."""

import argparse
import json
import math
from pathlib import Path

from .board import chessboard_points
from .config import fingerprint, load_config, write_json


def capture(out, split="intrinsic", board_transform=None, square_m=None):
    """Capture calibration frames and read COM5 without configuring or writing motors."""
    import time

    import cv2
    import numpy as np

    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    board = {"inner_corners": [9, 6], "square_m": square_m}
    if square_m is not None:
        chessboard_points(board)  # Validate before touching hardware.
    if board_transform is not None and square_m is None:
        raise ValueError("A board transform requires an explicitly measured square size")
    robot = SO101Follower(
        SO101FollowerConfig(port="COM5", id="zihao_follower_arm", use_degrees=True, cameras={})
    )
    from .config import JOINTS

    result = {"motion_commands_sent": False, "board": board, "cameras": {}}
    world_points = None
    if board_transform is not None:
        transform = np.asarray(json.loads(Path(board_transform).read_text()), dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("--board-transform must contain a measured 4x4 T_world_board matrix")
        local = chessboard_points(board)
        world_points = (local @ transform[:3, :3].T + transform[:3, 3]).tolist()
    try:
        robot.bus.connect()
        if not robot.is_calibrated:
            raise ValueError("Existing robot calibration does not match hardware")
        for name, index in [("front", 0), ("side", 2)]:
            camera = OpenCVCamera(
                OpenCVCameraConfig(
                    index_or_path=index, width=640, height=480, fps=30, fourcc="MJPG", backend=700
                )
            )
            try:
                camera.connect()
                time.sleep(1)
                before = robot.get_observation()
                frame = camera.read_latest()
                after = robot.get_observation()
                state = np.array([before[f"{n}.pos"] for n in JOINTS])
                final = np.array([after[f"{n}.pos"] for n in JOINTS])
                if np.max(np.abs(state[:5] - final[:5])) > 0.5 or abs(state[5] - final[5]) > 1:
                    raise ValueError("Arm moved during capture; hold the pose still and capture a new view")
                path = out / f"{name}.png"
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(path), bgr):
                    raise RuntimeError("Failed to save calibration capture")
                found, corners = cv2.findChessboardCornersSB(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (9, 6))
                if found:
                    cv2.drawChessboardCorners(bgr, (9, 6), corners, True)
                    for i in (0, 8, 45, 53):
                        cv2.putText(
                            bgr,
                            str(i),
                            tuple(corners[i, 0].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )
                    cv2.imwrite(str(out / f"{name}.corners.png"), bgr)
                result["cameras"][name] = {
                    "view_id": f"{out.name}_{name}",
                    "image": str(path.resolve()),
                    "captured_at_unix": time.time(),
                    "robot_state": state.tolist(),
                    "board_detected": bool(found),
                    "image_points_px": corners.reshape(-1, 2).tolist() if found else None,
                    "world_points_m": world_points,
                    "split": split,
                }
            finally:
                if camera.is_connected:
                    camera.disconnect()
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)
    write_json(out / "capture.json", result)
    return result


def main():
    """Dispatch an explicit stage into a fresh evidence directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=[
            "audit",
            "capture",
            "calibrate",
            "gripper-fit",
            "gripper-contact",
            "assemble",
            "replay",
            "generate",
            "train",
            "trials",
            "evaluate",
        ],
    )
    parser.add_argument("--config", type=Path, default=Path("configs/so101_r2s2r.json"))
    parser.add_argument("--output", type=Path, required=True, help="Must be a new directory")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--asset", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--measurements", type=Path, help="Saved gripper ruler measurements")
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--joint-references", type=Path)
    parser.add_argument("--split", choices=["intrinsic", "fit", "check"], default="intrinsic")
    parser.add_argument("--board-transform", type=Path)
    parser.add_argument("--square-mm", type=float, help="Measured square size, not the printed nominal size")
    parser.add_argument("--episodes", default="0,1,2")
    parser.add_argument("--replay-mode", choices=["state", "action"], default="state")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--replay-report", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--count", type=int, choices=[20, 200], default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--variant", choices=["baseline", "mixed"], default="baseline")
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--baseline")
    parser.add_argument("--mixed")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    if args.square_mm is not None and (not math.isfinite(args.square_mm) or args.square_mm <= 0):
        parser.error("--square-mm must be finite and positive")
    if args.square_mm is None and (
        args.mode == "assemble" or (args.mode == "capture" and args.board_transform is not None)
    ):
        parser.error("This operation requires an explicit measured --square-mm")
    square_m = None if args.square_mm is None else args.square_mm / 1000
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        if args.mode == "capture":
            result = capture(args.output, args.split, args.board_transform, square_m)
        elif args.mode == "assemble":
            if args.captures is None or args.joint_references is None:
                parser.error("assemble requires --captures and --joint-references")
            result = {"joint_reference_poses": json.loads(args.joint_references.read_text()), "cameras": {}}
            for name in ("front", "side"):
                camera = {
                    "board": {"inner_corners": [9, 6], "square_m": square_m},
                    "intrinsic_images": [],
                    "views": [],
                }
                for path in sorted(args.captures.rglob("capture.json")):
                    captured = json.loads(path.read_text(encoding="utf-8-sig"))
                    row = captured["cameras"].get(name)
                    if row is None:
                        continue  # Intrinsic captures may be collected one camera at a time.
                    if not row["board_detected"]:
                        continue
                    recorded_size = captured.get("board", {}).get("square_m")
                    if recorded_size is not None and not math.isclose(recorded_size, square_m):
                        raise ValueError(f"Measured board size differs from --square-mm in {path}")
                    if row["world_points_m"] is not None and recorded_size is None:
                        raise ValueError(
                            f"World points have no recorded board scale in {path}; regenerate them"
                        )
                    if row["split"] == "intrinsic":
                        camera["intrinsic_images"].append(row["image"])
                    elif row["world_points_m"] is None:
                        raise ValueError(f"Missing measured board transform for {path}")
                    else:
                        camera["views"].append(row)
                result["cameras"][name] = camera
            write_json(args.output / "observations.json", result)
        else:
            config = load_config(args.config)
            if args.dataset_root:
                config["dataset_root"] = str(args.dataset_root.resolve())
            if args.asset:
                config["asset"] = str(args.asset.resolve())
            write_json(args.output / "resolved_config.json", config)
            if args.mode == "audit":
                from .config import validate_physics
                from .data import SourceData

                result = SourceData(config).manifest()
                try:
                    validate_physics(config)
                    result["measurements_ready"] = True
                except (ValueError, TypeError) as exc:
                    result.update(measurements_ready=False, missing=str(exc))
            elif args.mode == "gripper-fit":
                from .gripper_fit import fit_gripper

                if args.measurements is None:
                    parser.error("gripper-fit requires --measurements")
                result = fit_gripper(
                    config,
                    json.loads(args.measurements.read_text(encoding="utf-8")),
                    args.output,
                    args.measurements,
                )
            elif args.mode == "gripper-contact":
                from .gripper_contact import align_gripper_contact

                if args.measurements is None:
                    parser.error("gripper-contact requires --measurements")
                result = align_gripper_contact(
                    config, json.loads(args.measurements.read_text(encoding="utf-8")), args.output
                )
            elif args.mode == "calibrate":
                from .calibration import calibrate

                if args.observations is None:
                    parser.error("calibrate requires --observations")
                result = calibrate(config, json.loads(args.observations.read_text()), args.output)
            elif args.mode == "replay":
                from .pipeline import replay

                result = replay(
                    config, args.output, [int(x) for x in args.episodes.split(",")], args.replay_mode
                )
            elif args.mode == "generate":
                from .pipeline import generate

                if args.calibration is None or args.replay_report is None:
                    parser.error("generate requires --calibration and --replay-report")
                result = generate(
                    config,
                    args.output,
                    args.count,
                    args.calibration,
                    args.replay_report,
                    args.pilot,
                    args.seed,
                )
            elif args.mode == "train":
                from .training import train

                if args.base is None or (args.variant == "mixed" and args.generated is None):
                    parser.error("train requires --base and mixed training requires --generated")
                train(
                    config,
                    args.output,
                    args.base.resolve(),
                    args.variant,
                    args.generated,
                    args.device,
                    args.smoke,
                )
                result = {"completed": True}
            elif args.mode == "trials":
                from .pipeline import trial_plan

                if not args.baseline or not args.mixed:
                    parser.error("trials requires --baseline and --mixed checkpoint paths")
                result = trial_plan(config, args.baseline, args.mixed)
                write_json(args.output / "trial_plan.json", result)
                rows = [
                    {
                        "candidate": candidate,
                        "layout_id": layout["layout_id"],
                        "grasped": None,
                        "placed": None,
                        "fault": None,
                        "seconds": None,
                        "evidence": None,
                    }
                    for layout in result["trials"]
                    for candidate in layout["order"]
                ]
                write_json(
                    args.output / "results.template.json",
                    {"plan_sha256": fingerprint(result), "results": rows},
                )
            else:
                if args.checkpoint is not None:
                    from .pipeline import evaluate_policy

                    if args.calibration is None or args.replay_report is None or args.plan or args.results:
                        parser.error(
                            "Simulation evaluation needs --calibration / --replay-report, without --plan / --results"
                        )
                    result = evaluate_policy(
                        config,
                        args.output,
                        args.checkpoint,
                        args.device,
                        args.seed,
                        args.eval_episodes,
                        args.calibration,
                        args.replay_report,
                    )
                else:
                    from .pipeline import compare_trials

                    if args.plan is None or args.results is None:
                        parser.error("evaluate requires --plan / --results or --checkpoint for simulation")
                    result = compare_trials(
                        json.loads(args.plan.read_text()), json.loads(args.results.read_text())
                    )
        write_json(args.output / "report.json", result)
        print(json.dumps(result, indent=2, allow_nan=False))
        if result.get("passed") is False:
            raise SystemExit(2)
    except Exception as exc:
        write_json(args.output / "failure.json", {"error": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
