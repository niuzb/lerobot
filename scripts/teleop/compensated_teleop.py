"""Opt-in Zihao teleoperation; never used by model inference.

Defaults to read-only. Record mode saves follower-space, actually sent actions
in degrees (gripper percentage), NOT the old model's range-normalized units.
"""

import argparse
import json
import math
import time
from pathlib import Path

WRIST_OFFSET = 1021 * 360 / 4095
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def map_action(action, calibration):
    """Apply reference-pose offset once; reject wrist wrapping, bound other axes."""
    if set(action) != {f"{n}.pos" for n in JOINTS}:
        raise ValueError("Expected all six joint positions")
    result = {}
    for name in JOINTS:
        value = float(action[f"{name}.pos"])
        if not math.isfinite(value):
            raise ValueError("Non-finite leader position")
        c = calibration[name]
        if name == "gripper":
            low, high = 0, 100
        else:
            extent = (c.range_max - c.range_min) * 360 / 4095 / 2
            low, high = -extent, extent
        if name == "wrist_roll":
            value += WRIST_OFFSET
            if not low <= value <= high:
                raise ValueError("Mapped wrist outside calibrated range; refusing wrap/clamp")
        result[f"{name}.pos"] = min(high, max(low, value))
    return result


def validate_reference(leader, follower):
    for bus in (leader, follower):
        if bus.read_calibration() != bus.calibration:
            raise ValueError("Hardware calibration changed; refusing motion")
        if any(bus.sync_read("Torque_Enable", normalize=False).values()):
            raise ValueError("Expected torque disabled before startup")
    for bus, offset in ((leader, -1405), (follower, -940)):
        c = bus.calibration["wrist_roll"]
        if (c.id, c.homing_offset, c.range_min, c.range_max) != (5, offset, 0, 4095):
            raise ValueError("Reference wrist calibration changed; remeasure compensation")


def main():
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "test", "record"), default="dry-run")
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args()
    if not 0 < args.seconds <= 60:
        parser.error("Duration must be in (0, 60]")
    cameras = {}
    if args.mode == "record":
        from lerobot.cameras.opencv import OpenCVCameraConfig

        cameras = {
            name: OpenCVCameraConfig(index_or_path=index, width=640, height=480, fps=30, fourcc="MJPG")
            for name, index in (("camera1", 2), ("camera2", 0))
        }
    follower = SO101Follower(
        SO101FollowerConfig(
            port="COM5",
            id="zihao_follower_arm",
            use_degrees=True,
            cameras=cameras,
            max_target_velocity=3.0,
            max_tracking_error=3.0,
            max_position_jump=5.0,
        )
    )
    leader = SO101Leader(SO101LeaderConfig(port="COM6", id="zihao_leader_arm", use_degrees=True))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dataset = None
    try:
        # Connect only buses: no automatic calibration, torque enable or camera setup.
        for bus in (leader.bus, follower.bus):
            bus.connect(handshake=False)
            bus.set_baudrate(1000000)
        validate_reference(leader.bus, follower.bus)
        initial = (
            follower.get_observation()
            if not cameras
            else {f"{n}.pos": v for n, v in follower.bus.sync_read("Present_Position").items()}
        )
        mapped = map_action(leader.get_action(), follower.calibration)
        delta = {k: mapped[k] - initial[k] for k in mapped}
        print(
            json.dumps(
                {"offset_degrees": WRIST_OFFSET, "initial": initial, "mapped": mapped, "delta": delta}
            ),
            flush=True,
        )
        if args.mode == "dry-run":
            return
        if any(abs(v) > 15 for v in delta.values()):
            raise ValueError("Initial mismatch exceeds 15 degrees/percentage points; align arms first")
        for camera in follower.cameras.values():
            camera.connect()
        if args.mode == "record":
            from lerobot.configs.video import RGBEncoderConfig
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            features = {
                key: {"dtype": "float32", "shape": (6,), "names": [f"{n}.pos" for n in JOINTS]}
                for key in ("observation.state", "action")
            }
            features.update(
                {
                    f"observation.images.{n}": {
                        "dtype": "video",
                        "shape": (480, 640, 3),
                        "names": ["height", "width", "channels"],
                    }
                    for n in cameras
                }
            )
            dataset = LeRobotDataset.create(
                repo_id=f"local/zihao_tape_{stamp}",
                root=Path("outputs") / f"zihao_tape_{stamp}",
                fps=30,
                robot_type="so101_follower",
                features=features,
                image_writer_threads=4,
                rgb_encoder=RGBEncoderConfig(vcodec="h264"),
            )
        Path("outputs").mkdir(exist_ok=True)
        log_path = Path("outputs") / f"compensated_teleop_{stamp}.jsonl"
        with log_path.open("x", encoding="utf-8") as log:
            log.write(
                json.dumps(
                    {
                        "mode": args.mode,
                        "offset": WRIST_OFFSET,
                        "units": "degrees/gripper_percent",
                        "calibration": {
                            role: {n: vars(c) for n, c in bus.calibration.items()}
                            for role, bus in (("leader", leader.bus), ("follower", follower.bus))
                        },
                    }
                )
                + "\n"
            )
            # Seed the current pose before configure() enables torque, avoiding stale goals.
            raw = follower.bus.sync_read("Present_Position", normalize=False)
            follower.bus.sync_write("Goal_Position", raw, normalize=False)
            follower.configure()
            print(f"START {args.mode}: {args.seconds}s; log={log_path.resolve()}", flush=True)
            start = time.monotonic()
            count = 0
            previous_leader = None
            while time.monotonic() - start < args.seconds:
                tick = time.monotonic()
                obs = follower.get_observation()
                action = leader.get_action()
                if previous_leader and any(abs(action[k] - previous_leader[k]) > 10 for k in action):
                    raise ValueError("Leader feedback jump; stopping")
                previous_leader = action
                mapped = map_action(action, follower.calibration)
                sent = follower.send_action(mapped)
                if set(sent) != set(mapped):
                    raise ValueError("Follower rejected action; stopping, no frame recorded")
                row = {
                    "t": time.monotonic() - start,
                    "leader": action,
                    "mapped": mapped,
                    "sent": sent,
                    "observed": {k: obs[k] for k in sent},
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                if dataset is not None:
                    import numpy as np

                    dataset.add_frame(
                        {
                            "observation.state": np.array(
                                [obs[f"{n}.pos"] for n in JOINTS], dtype=np.float32
                            ),
                            "action": np.array([sent[f"{n}.pos"] for n in JOINTS], dtype=np.float32),
                            **{f"observation.images.{n}": obs[n] for n in cameras},
                            "task": "Grab tapes and place into pen holder",
                        }
                    )
                count += 1
                if count % 30 == 0:
                    print(json.dumps(row), flush=True)
                time.sleep(max(0, 1 / 30 - (time.monotonic() - tick)))
        # Disable torque before potentially slow video encoding.
        follower.bus.disable_torque()
        if dataset is not None:
            dataset.save_episode(parallel_encoding=False)
        print(f"FINISHED: {count} frames; no automatic reset", flush=True)
    finally:
        # Always attempt every cleanup, even if a device fails to disconnect.
        for bus in (follower.bus, leader.bus):
            if bus.is_connected:
                try:
                    bus.disconnect(disable_torque=args.mode != "dry-run")
                except Exception as exc:
                    print(f"CLEANUP ERROR: {exc}", flush=True)
        for camera in follower.cameras.values():
            if camera.is_connected:
                camera.disconnect()
        if dataset is not None:
            dataset.finalize()


if __name__ == "__main__":
    main()
