"""No-torque observation stream test for the remote SmolVLA server.

This script validates the data sent over gRPC without driving the arm:

* connects the SO-101 bus and cameras directly (never calls ``robot.connect``),
  so it does not enable torque or write Goal_Position;
* sends timestamped observations for a short fixed interval;
* logs each payload's timestep, timestamp, byte size, joint values, camera
  shapes/dtypes/ranges, task text, and send latency;
* prints one-second receive/send-rate summaries;
* requests one action chunk at the end and inspects it without applying it.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import grpc

from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedObservation,
    encode_timed_observation,
    map_robot_keys_to_lerobot_features,
)
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

SERVER = "127.0.0.1:18080"
CKPT = "/root/code/lerobot/output_lerobot_train_20260704_140211/checkpoints/040000/pretrained_model"
TASK = "Grab the cube and place it into the cup"
DURATION_SECONDS = 10.0
TARGET_FPS = 30.0
JPEG_QUALITY = 80
LOG_PATH = Path("outputs/observation_stream.log")


def _value_summary(value: object) -> str:
    """Return a compact, deterministic summary without serializing image data."""
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        try:
            min_value = float(value.min())
            max_value = float(value.max())
            mean_value = float(value.mean())
            return (
                f"shape={tuple(shape)} dtype={dtype} "
                f"min={min_value:.1f} max={max_value:.1f} mean={mean_value:.1f}"
            )
        except (TypeError, ValueError, AttributeError):
            return f"shape={tuple(shape)} dtype={dtype}"
    if isinstance(value, (int, float)):
        return f"{value:.5f}"
    return repr(value)


def _observation_summary(observation: dict[str, object]) -> str:
    parts = []
    for key, value in observation.items():
        if key == "task":
            continue
        parts.append(f"{key}={_value_summary(value)}")
    return "; ".join(parts)


def main() -> None:
    """Capture and stream observations while keeping motor torque disabled."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file = LOG_PATH.open("w", encoding="utf-8")

    camera_cfg = {
        "camera2": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30, fourcc="MJPG"),
        "camera1": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30, fourcc="MJPG"),
    }
    robot = SO101Follower(
        SO101FollowerConfig(port="COM5", id="zihao_follower_arm", cameras=camera_cfg)
    )

    channel = None
    try:
        # Deliberately avoid robot.connect(): it configures/enables the motors.
        robot.bus.connect()
        for camera in robot.cameras.values():
            camera.connect()

        channel = grpc.insecure_channel(SERVER, options=grpc_channel_options())
        stub = services_pb2_grpc.AsyncInferenceStub(channel)
        print(f"=== Remote observation stream: {SERVER} ===", flush=True)
        stub.Ready(services_pb2.Empty(), timeout=30)

        policy_config = RemotePolicyConfig(
            policy_type="smolvla",
            pretrained_name_or_path=CKPT,
            lerobot_features=map_robot_keys_to_lerobot_features(robot),
            actions_per_chunk=10,
            device="cuda",
        )
        stub.SendPolicyInstructions(
            services_pb2.PolicySetup(data=pickle.dumps(policy_config)), timeout=120
        )

        print(f"Sending {TARGET_FPS:.0f} FPS for {DURATION_SECONDS:.1f}s (NO TORQUE)", flush=True)
        print(f"Detailed log: {LOG_PATH.resolve()}", flush=True)
        log_file.write(f"# server={SERVER} target_fps={TARGET_FPS} task={TASK!r}\n")

        start = time.perf_counter()
        next_frame = start
        last_report = start
        sent_count = 0
        report_count = 0
        total_bytes = 0
        total_send_time = 0.0

        while True:
            now = time.perf_counter()
            if now - start >= DURATION_SECONDS:
                break
            if now < next_frame:
                time.sleep(next_frame - now)
            next_frame += 1.0 / TARGET_FPS

            raw_observation = robot.get_observation()
            raw_observation["task"] = TASK
            timestamp = time.time()
            timed_observation = TimedObservation(
                timestamp=timestamp,
                observation=raw_observation,
                timestep=sent_count,
                must_go=sent_count == 0,
            )
            wire_observation = encode_timed_observation(timed_observation, JPEG_QUALITY)
            serialized = pickle.dumps(wire_observation)
            send_started = time.perf_counter()
            request_iterator = send_bytes_in_chunks(
                serialized,
                services_pb2.Observation,
                log_prefix="[OBS-TEST]",
                silent=True,
            )
            stub.SendObservations(request_iterator, timeout=10)
            send_time = time.perf_counter() - send_started

            sent_count += 1
            total_bytes += len(serialized)
            total_send_time += send_time
            log_file.write(
                f"t={timestamp:.6f} seq={timed_observation.get_timestep()} "
                f"must_go={timed_observation.must_go} payload_bytes={len(serialized)} "
                f"send_ms={send_time * 1000:.2f} jpeg_quality={JPEG_QUALITY} task={TASK!r} "
                f"{_observation_summary(raw_observation)}\n"
            )
            log_file.flush()

            report_now = time.perf_counter()
            if report_now - last_report >= 1.0:
                report_elapsed = report_now - last_report
                interval_count = sent_count - report_count
                print(
                    f"[stream] sent={sent_count} interval_fps={interval_count / report_elapsed:.2f} "
                    f"avg_fps={sent_count / (report_now - start):.2f} "
                    f"last_seq={timed_observation.get_timestep()} "
                    f"payload={len(serialized) / 1024:.1f}KiB send_ms={send_time * 1000:.1f}",
                    flush=True,
                )
                report_count = sent_count
                last_report = report_now

        elapsed = time.perf_counter() - start
        print(
            f"=== Stream complete: sent={sent_count}, avg_fps={sent_count / elapsed:.2f}, "
            f"avg_payload={total_bytes / max(sent_count, 1) / 1024:.1f}KiB, "
            f"avg_send_ms={total_send_time / max(sent_count, 1) * 1000:.2f} ===",
            flush=True,
        )

        # This only validates that the server can consume the latest observation and
        # produce a policy output. The returned action is never sent to the robot.
        print("=== Requesting one action chunk for validation (NOT APPLYING) ===", flush=True)
        response = stub.GetActions(services_pb2.Empty(), timeout=60)
        if not response.data:
            print("No action chunk returned", flush=True)
        else:
            payload = pickle.loads(response.data)  # nosec
            if isinstance(payload, list):
                print(f"action_chunk_len={len(payload)}", flush=True)
                if payload:
                    action = payload[0].get_action()
                    print(
                        f"first_action shape={tuple(action.shape)} dtype={action.dtype} "
                        f"finite={bool(action.isfinite().all())} values={action.tolist()}",
                        flush=True,
                    )
            else:
                print(f"unexpected action payload type={type(payload)!r}", flush=True)
        print("=== OBSERVATION STREAM TEST OK (NO TORQUE) ===", flush=True)
    finally:
        if channel is not None:
            channel.close()
        try:
            robot.bus.disconnect(disable_torque=True)
        finally:
            for camera in robot.cameras.values():
                camera.disconnect()
            log_file.close()


if __name__ == "__main__":
    main()
