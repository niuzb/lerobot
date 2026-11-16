"""Drive the existing SO-101 with the General Sequences MolmoAct2 deployment.

Default: read-only cameras/joints and three cloud predictions. --execute enables
a bounded rollout with the existing 5 units/s ramp and 3-unit tracking guard.
Credentials come only from SEQUENCES_API_KEY (or an interactive getpass prompt).

Local setup (from this checkout):
    uv pip install -r scripts/policies/sequences/requirements-sequences.txt
    uv run --no-sync python scripts/policies/sequences/setup_sequences_windows.py  # Windows only
Read-only check:
    uv run --no-sync python scripts/policies/sequences/sequences_so101_rollout.py
Five-second rollout:
    uv run --no-sync python scripts/policies/sequences/sequences_so101_rollout.py --execute --seconds 5

The template executes each 30-step chunk before requesting another; it holds
position during the next inference. Body joints use degrees; the gripper uses
0..100. This script expects the existing LeRobot calibration, not raw encoder ticks.
"""

import argparse
import getpass
import json
import math
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ACTION_KEYS = tuple(f"{joint}.pos" for joint in JOINTS)
HZ = 30
CHUNK_STEPS = 30
# Official molmoact2-so101 v2 client.py: LeRobot v3 joint frame <-> training frame.
# https://generalsequences.com/templates/molmoact2-so101/#v2
SIGNS = np.array([1, -1, 1, 1, 1, 1], dtype=np.float64)
OFFSETS = np.array([0, 90, 90, 0, 0, 0], dtype=np.float64)


def to_model_state(state):
    """Convert calibrated LeRobot degrees into MolmoAct2's training coordinates."""
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError("Expected six finite SO-101 joint positions")
    return (state * SIGNS + OFFSETS).astype(np.float32)


def to_robot_actions(answer, bounds):
    """Convert and validate the entire answer before allowing its first motor write."""
    if not isinstance(answer, dict) or "actions" not in answer:
        raise ValueError("Expected MolmoAct2 answer with actions")
    raw = np.asarray(answer["actions"], dtype=np.float64)
    if raw.shape != (CHUNK_STEPS, 6) or not np.isfinite(raw).all():
        raise ValueError("Expected a finite (30, 6) absolute-position action chunk")
    actions = (raw - OFFSETS) * SIGNS
    # The gripper is a bounded 0..100 position actuator. Match MotorsBus._unnormalize's
    # RANGE_0_100 saturation explicitly; learned close/open targets can overshoot endpoints.
    # Body joints remain strictly checked against their calibrated degree limits below.
    actions[:, 5] = np.clip(actions[:, 5], bounds[5, 0], bounds[5, 1])
    violations = np.argwhere((actions < bounds[:, 0]) | (actions > bounds[:, 1]))
    if violations.size:
        row, column = violations[0]
        raise ValueError(
            f"Cloud target {JOINTS[column]} at step {row}: {actions[row, column]:.3f} outside "
            f"{bounds[column].tolist()}; no actions from this chunk sent"
        )
    return actions


def joint_bounds(robot):
    """Use the same encoder-to-degrees mapping as MotorsBus._normalize."""
    bounds = []
    for name in JOINTS[:-1]:
        calibration = robot.calibration[name]
        resolution = robot.bus.model_resolution_table[robot.bus.motors[name].model] - 1
        half_range = (calibration.range_max - calibration.range_min) * 180 / resolution
        if not math.isfinite(half_range) or half_range <= 0:
            raise ValueError(f"Invalid calibrated range: {name}")
        bounds.append((-half_range, half_range))
    return np.asarray([*bounds, (0, 100)], dtype=np.float64)


def make_observation(observation, task):
    """Map existing front/side RGB cameras to the template's two JPEG inputs."""
    wire = {"state": to_model_state([observation[key] for key in ACTION_KEYS]), "prompt": task}
    for local, remote in (("front", "top_image"), ("side", "side_image")):
        frame = observation[local]
        if frame.dtype != np.uint8 or frame.shape != (480, 640, 3):
            raise ValueError(f"{local}: expected a 640x480 RGB uint8 frame")
        # The official server's _image accepts JPEG bytes as well as SDK Video.
        ok, jpeg = cv2.imencode(
            ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        if not ok:
            raise ValueError(f"Could not encode {local}")
        wire[remote] = jpeg.tobytes()
    return wire


def checked_observation(robot, bounds):
    """Reject measured states outside the calibrated workspace before any motor writes."""
    observation = robot.get_observation()
    positions = np.asarray([observation[key] for key in ACTION_KEYS], dtype=np.float64)
    to_model_state(positions)  # Also checks shape and finite feedback.
    if np.any(positions < bounds[:, 0]) or np.any(positions > bounds[:, 1]):
        raise ValueError("Measured position outside calibrated joint limits; inspect the arm before retrying")
    return observation


def bounded_call(policy, function, timeout, deadline=None):
    """Bound observe+act together, including SDK retries/reconnects, without motor threads."""
    result, failure = [], []
    done = threading.Event()

    def call():
        try:
            result.append(function())
        except BaseException as exc:
            failure.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    expires = time.monotonic() + timeout
    if deadline is not None:
        expires = min(expires, deadline)
    try:
        while not done.wait(min(0.05, max(0, expires - time.monotonic()))):
            if time.monotonic() >= expires:
                raise TimeoutError("Cloud request/run deadline reached; stopping")
        if time.monotonic() >= expires:
            raise TimeoutError("Cloud response arrived after deadline; discarded")
        if failure:
            raise failure[0]
        return result[0]
    finally:
        if not done.is_set():
            policy.interrupt()
            thread.join(timeout=1)


def request_actions(policy, observation, task, timeout, deadline=None):
    """Send one current observation and receive a chunk under a shared deadline."""
    wire = make_observation(observation, task)

    def predict():
        policy.observe(wire)
        return policy.act()

    return bounded_call(policy, predict, timeout, deadline)


def play_chunk(robot, actions, deadline, log, clock=time.monotonic, sleep=time.sleep, *, action_filter=None):
    """Play at 30 Hz on absolute deadlines, skipping missed ticks rather than bursting."""
    start = clock()
    index = 0
    count = 0
    while index < len(actions):
        now = clock()
        if now >= deadline:
            break
        index = max(index, int((now - start) * HZ))
        if index >= len(actions):
            break
        requested = dict(zip(ACTION_KEYS, actions[index].tolist(), strict=True))
        applied = requested if action_filter is None else action_filter(requested, now)
        sent = robot.send_action(applied)
        record = {
            "time": clock(),
            "requested": requested,
            "sent": sent,
            "measured": robot._last_stable_present_pos,
        }
        if action_filter is not None:
            record["applied"] = applied
        log.write(json.dumps(record) + "\n")
        count += 1
        index += 1
        sleep(max(0, min(deadline, start + index / HZ) - clock()))
    return count


def run(args):
    """Check real observations with the cloud, then optionally execute a bounded run."""
    import sequence_ai

    if not os.environ.get("SEQUENCES_API_KEY"):
        os.environ["SEQUENCES_API_KEY"] = getpass.getpass("General Sequences robot key (hidden): ")
    output = Path("outputs") / ("sequences_so101_" + time.strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.robot_id,
            use_degrees=True,
            max_target_velocity=5.0,
            max_tracking_error=3.0,
            cameras={
                name: OpenCVCameraConfig(
                    index_or_path=index, width=640, height=480, fps=HZ, fourcc="MJPG", backend=700
                )
                for name, index in (("front", args.front_camera), ("side", args.side_camera))
            },
        )
    )
    policy = None
    motion_enabled = False
    summary = {
        "model": args.model,
        "execute": args.execute,
        "task": args.task,
        "chunks": 0,
        "actions_sent": 0,
        "status": "failed",
        "output": str(output.resolve()),
    }
    try:
        policy = sequence_ai.connect(model=args.model, timeout_s=20, startup_timeout_s=0)
        print(f"Waiting for cloud deployment {args.model} (up to 10 minutes; motors untouched)", flush=True)
        bounded_call(policy, lambda: policy.wait_until_ready(timeout_s=600), 605)
        # Avoid robot.connect(): preflight must not configure motors or change torque/calibration.
        robot.bus.connect()
        if not robot.is_calibrated:
            raise RuntimeError(
                "Existing calibration does not match the robot; stopping without recalibration"
            )
        for camera in robot.cameras.values():
            camera.connect()
        bounds = joint_bounds(robot)
        summary["joint_bounds"] = bounds.tolist()
        with (output / "predictions.jsonl").open("a", encoding="utf-8") as predictions:
            for index in range(args.preflight_rounds):
                observation = checked_observation(robot, bounds)
                if index == 0:
                    for name in robot.cameras:
                        cv2.imwrite(
                            str(output / f"{name}.jpg"), cv2.cvtColor(observation[name], cv2.COLOR_RGB2BGR)
                        )
                start = time.monotonic()
                answer = request_actions(policy, observation, args.task, 20)
                actions = to_robot_actions(answer, bounds)
                record = {
                    "phase": "preflight",
                    "round": index,
                    "round_trip_ms": (time.monotonic() - start) * 1000,
                    "robot_state": [observation[key] for key in ACTION_KEYS],
                    "model_state": to_model_state([observation[key] for key in ACTION_KEYS]).tolist(),
                    "raw_actions": np.asarray(answer["actions"]).tolist(),
                    "robot_actions": actions.tolist(),
                }
                predictions.write(json.dumps(record) + "\n")
                predictions.flush()
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in record.items()
                            if key not in ("raw_actions", "robot_actions")
                        }
                    ),
                    flush=True,
                )
            print(
                f"PREFLIGHT OK: {args.preflight_rounds} finite, bounded chunks; no motion writes; {output.resolve()}",
                flush=True,
            )
            if args.execute:
                # Set the held goal to actual measured position before configure() enables torque.
                present = robot._read_present_position()
                robot.bus.sync_write("Goal_Position", present)
                motion_enabled = True
                robot.configure()
                # First action holds current pose; subsequent sends use the driver's time-based ramp.
                robot.send_action({f"{key}.pos": value for key, value in present.items()})
                started = time.monotonic()
                deadline = started + args.seconds
                next_snapshot = started
                with (output / "actions.jsonl").open("a", encoding="utf-8") as log:
                    while time.monotonic() < deadline:
                        observation = checked_observation(robot, bounds)
                        if time.monotonic() >= deadline:
                            break
                        now = time.monotonic()
                        if now >= next_snapshot:
                            elapsed = now - started
                            for name in robot.cameras:
                                cv2.imwrite(
                                    str(output / f"t{elapsed:06.2f}_{name}.jpg"),
                                    cv2.cvtColor(observation[name], cv2.COLOR_RGB2BGR),
                                )
                            print(
                                f"EXECUTING elapsed={elapsed:.2f}s actions={summary['actions_sent']} "
                                f"chunks={summary['chunks']}",
                                flush=True,
                            )
                            next_snapshot = now + 5
                        try:
                            answer = request_actions(policy, observation, args.task, 2, deadline)
                        except TimeoutError:
                            if time.monotonic() >= deadline:
                                break
                            raise
                        actions = to_robot_actions(answer, bounds)
                        predictions.write(
                            json.dumps(
                                {
                                    "phase": "execute",
                                    "time": time.monotonic(),
                                    "raw_actions": np.asarray(answer["actions"]).tolist(),
                                    "robot_actions": actions.tolist(),
                                }
                            )
                            + "\n"
                        )
                        predictions.flush()
                        summary["chunks"] += 1
                        summary["actions_sent"] += play_chunk(robot, actions, deadline, log)
                summary["control_seconds"] = time.monotonic() - started
        summary["status"] = "ok"
    except BaseException as exc:
        summary["error"] = str(exc)
        raise
    finally:
        # Release hardware before potentially slow network cleanup; preflight preserves torque.
        try:
            if robot.bus.is_connected:
                robot.bus.disconnect(disable_torque=motion_enabled)
        finally:
            for camera in robot.cameras.values():
                if camera.is_connected:
                    camera.disconnect()
            action_log = output / "actions.jsonl"
            if action_log.exists():
                # Preserve accurate successful-send counts if a guard fails mid-chunk.
                with action_log.open(encoding="utf-8") as stream:
                    summary["actions_sent"] = sum(1 for _ in stream)
            (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print("STOPPED " + json.dumps(summary), flush=True)
            if policy is not None:
                policy.interrupt()
                policy.close()


def main():
    """Parse the local robot and deployment settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="so101")
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--robot-id", default="zihao_follower_arm")
    parser.add_argument("--front-camera", type=int, default=0)
    parser.add_argument("--side-camera", type=int, default=2)
    parser.add_argument("--task", default="Put the tape into the pen holder")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--preflight-rounds", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 300:
        parser.error("--seconds must be finite and between 0 and 300")
    if not 1 <= args.preflight_rounds <= 10:
        parser.error("--preflight-rounds must be between 1 and 10")
    run(args)


if __name__ == "__main__":
    main()
