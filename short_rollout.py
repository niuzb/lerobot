"""Short timed rollout with verbose action logging.

Monkey-patches RobotClient.control_loop_action to print:
- target action (6 joints) sent to robot
- measured positions before each write (in the action log)
- action queue size
"""
import math
import threading
import time
from pathlib import Path

import cv2

from lerobot.async_inference.robot_client import (
    RobotClient,
    RobotClientConfig,
)
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101FollowerConfig

RUN_SECONDS = 200.0
RESET_TIMEOUT_SECONDS = 60.0
ACTION_LOG_PATH = Path(f"outputs/short_rollout_actions_{time.strftime('%Y%m%d_%H%M%S')}.log")
SNAPSHOT_INTERVAL_SECONDS = 5.0
SNAPSHOT_DIR = ACTION_LOG_PATH.with_suffix("") / "snapshots"
_snapshot_state = {"started": None, "next": 0.0}


def save_camera_snapshots(observation) -> None:
    """Save both RGB observation images on the first frame and every five seconds."""
    if observation is None:
        return
    now = time.monotonic()
    if _snapshot_state["started"] is None:
        _snapshot_state["started"] = now
    elapsed = now - _snapshot_state["started"]
    if elapsed < _snapshot_state["next"]:
        return
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("camera1", "camera2"):
        frame = observation[name]
        path = SNAPSHOT_DIR / f"t{elapsed:08.2f}s_{name}.jpg"
        if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Failed to save camera snapshot: {path}")
        paths.append(str(path))
    _snapshot_state["next"] = (int(elapsed / SNAPSHOT_INTERVAL_SECONDS) + 1) * SNAPSHOT_INTERVAL_SECONDS
    message = f"SNAPSHOT elapsed={elapsed:.3f} paths={paths!r}"
    print(message, flush=True)
    with ACTION_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"t={now:.3f} {message}\n")

# --- Monkey-patch: log every action ---
_orig_control_loop_action = RobotClient.control_loop_action

def _verbose_control_loop_action(self, verbose=False):
    with self.action_queue_lock:
        qsize = self.action_queue.qsize()
        timed_action = self.action_queue.get_nowait()
    action_tensor = timed_action.get_action()
    policy_action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
    action_dict = self._action_tensor_to_action_dict(action_tensor)
    performed = self.robot.send_action(action_dict)
    with self.latest_action_lock:
        self.latest_action = timed_action.get_timestep()

    # Print requested/sent targets; measured feedback is recorded separately below.
    names = ["sh_pan", "sh_lift", "elbow", "wrist_f", "wrist_r", "grip"]
    targets = [f"{action_dict.get(k, 0):+7.2f}" for k in
               ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]]
    # performed contains sent targets, not measured positions.
    actual = []
    for k in ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
              "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]:
        v = performed.get(k) if isinstance(performed, dict) else None
        actual.append(f"{v:+7.2f}" if isinstance(v, (int, float)) else "   ?   ")
    print(f"[t={time.monotonic():6.1f}s] step={timed_action.get_timestep():3d} q={qsize:2d}  "
          + " | ".join(f"{n}:{a}/{p}" for n, a, p in zip(names, targets, actual, strict=True)),
          flush=True)
    with ACTION_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"t={time.monotonic():.3f} step={timed_action.get_timestep()} q={qsize} "
            f"policy_action={policy_action!r} requested={action_dict!r} performed={performed!r} "
            f"feedback_before_write={self.robot._last_stable_present_pos!r}\n"
        )
    return performed

# --- end patch ---

# --- Monkey-patch: log observation send FPS ---
_obs_count = {"n": 0, "t0": time.monotonic()}
_orig_obs = RobotClient.control_loop_observation

def _verbose_obs(self, task, verbose=False):
    result = _orig_obs(self, task, verbose)
    save_camera_snapshots(result)
    _obs_count["n"] += 1
    dt = time.monotonic() - _obs_count["t0"]
    if _obs_count["n"] % 10 == 0:
        fps = _obs_count["n"] / dt
        print(f"[t={time.monotonic():6.1f}s] OBS FPS={fps:5.1f} (n={_obs_count['n']}, {dt:.1f}s elapsed)", flush=True)
    return result

# --- end obs patch ---

robot_cfg = SO101FollowerConfig(
    port="COM5",
    id="zihao_follower_arm",
    use_degrees=True,  # Driver/protection/reset units stay physical; policy conversion is separate.
    cameras={
        "camera2": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30, fourcc="MJPG"),
        "camera1": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30, fourcc="MJPG"),
    },
    max_target_velocity={
        "shoulder_pan": 3.0,
        "shoulder_lift": 3.0,
        "elbow_flex": 3.0,
        "wrist_flex": 3.0,
        "wrist_roll": 1.0,
        "gripper": 3.0,
    },
    max_tracking_error={
        "shoulder_pan": 3.0,
        "shoulder_lift": 3.0,
        "elbow_flex": 3.0,
        "wrist_flex": 3.0,
        "wrist_roll": 1.0,
        "gripper": 3.0,
    },
)

cfg = RobotClientConfig(
    robot=robot_cfg,
    server_address="127.0.0.1:18080",
    policy_type="smolvla",
    pretrained_name_or_path=(
        "/root/code/lerobot/output_lerobot_train_20260704_140211/"
        "checkpoints/040000/pretrained_model"
    ),
    policy_device="cuda",
    client_device="cpu",
    actions_per_chunk=10,
    chunk_size_threshold=0.5,
    fps=30,
    observation_jpeg_quality=80,
    policy_joint_units="range_m100_100",
    aggregate_fn_name="weighted_average",
    task="Grab tapes and place into pen holder",
)

def log_reset(message: str) -> None:
    """Record reset status in the console and action log."""
    print(f"[RESET] {message}", flush=True)
    with ACTION_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"t={time.monotonic():.3f} RESET {message}\n")


def capture_initial_position(robot) -> dict[str, float]:
    """Capture this run's starting pose directly from motor feedback."""
    positions = robot.bus.sync_read("Present_Position", num_retry=robot.config.num_read_retries)
    if set(positions) != set(robot.bus.motors) or any(not math.isfinite(v) for v in positions.values()):
        raise RuntimeError("Cannot capture complete finite initial joint positions")
    return {f"{motor}.pos": value for motor, value in positions.items()}


def return_to_initial_position(robot, initial: dict[str, float]) -> None:
    """Return through the same bounded ramp; verify measured arrival, never just the sent goal."""
    if robot.config.max_target_velocity is None or robot.config.max_tracking_error is None:
        raise RuntimeError("Reset requires the time-based ramp and tracking-error limit")
    log_reset(f"start target={initial!r}")
    deadline = time.monotonic() + RESET_TIMEOUT_SECONDS
    settled_since = None
    while time.monotonic() < deadline:
        sent = robot.send_action(initial)
        if set(sent) != set(initial):
            raise RuntimeError("Reset aborted: position guard rejected action")
        feedback = robot._last_stable_present_pos
        errors = {key: abs(feedback[key.removesuffix('.pos')] - value) for key, value in initial.items()}
        log_reset(f"sent={sent!r} feedback={feedback!r} errors={errors!r}")
        arrived = all(error <= (1.0 if key == "gripper.pos" else 0.5) for key, error in errors.items())
        if arrived:
            if settled_since is None:
                settled_since = time.monotonic()
            if time.monotonic() - settled_since >= 0.5:
                log_reset("complete: measured position within tolerance for 0.5s")
                return
        else:
            settled_since = None
        time.sleep(1.0 / 30.0)
    raise TimeoutError("Reset timed out without measured arrival")


def runner() -> None:
    """Run the client for a fixed interval, then stop before disconnecting."""
    ACTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTION_LOG_PATH.write_text(
        "# policy_action=range units (queue-merged); requested=driver target after conversion; "
        "performed=sent target; feedback_before_write=measured position; "
        "driver body units=degrees, gripper units=0..100\n",
        encoding="utf-8",
    )
    RobotClient.control_loop_action = _verbose_control_loop_action
    RobotClient.control_loop_observation = _verbose_obs
    _snapshot_state.update(started=None, next=0.0)
    print(f"Task: {cfg.task}; snapshots: {SNAPSHOT_DIR.resolve()}", flush=True)
    client = RobotClient(cfg)
    receiver = None
    control = None
    completed = False
    control_errors = []
    def controlled_loop() -> None:
        try:
            client.control_loop(cfg.task)
        except Exception as exc:
            control_errors.append(exc)
            client.logger.exception("Control failed; automatic reset will be skipped")
        finally:
            # Also disconnect promptly if tracking protection aborts control.
            client.shutdown_event.set()
    try:
        if not client.start():
            raise RuntimeError("PolicyServer handshake failed")

        initial = capture_initial_position(client.robot)
        log_reset(f"captured initial={initial!r}")
        receiver = threading.Thread(target=client.receive_actions, daemon=True)
        receiver.start()
        control = threading.Thread(target=controlled_loop, daemon=True)
        control.start()

        print(f"=== Running for {RUN_SECONDS}s. WATCH THE ARM. ===", flush=True)
        completed = not client.shutdown_event.wait(RUN_SECONDS)
    finally:
        # Stop the control loop before disconnecting the robot to avoid a final action racing with shutdown.
        client.shutdown_event.set()
        if control is not None:
            control.join(timeout=3.0)
        try:
            if completed and not control_errors and control is not None and not control.is_alive():
                return_to_initial_position(client.robot, initial)
            else:
                log_reset("skipped: interrupted/failed rollout or control thread still active")
        except BaseException as exc:
            log_reset(f"aborted: {type(exc).__name__}: {exc}")
            raise
        finally:
            client.stop()
            if receiver is not None:
                receiver.join(timeout=1.0)
            print("=== Timed rollout stopped; robot disconnected ===", flush=True)
        if control_errors:
            raise RuntimeError("Rollout control failed") from control_errors[0]


if __name__ == "__main__":
    runner()
