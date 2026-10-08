"""Prediction-only smoke test.

- Connect bus + cameras (NO robot.connect() -> no torque)
- gRPC handshake with PolicyServer (triggers model load on the remote GPU)
- Send one real observation, receive one action chunk
- Print action shape / values; DO NOT call send_action.
"""
import pickle
import sys
import time
import traceback

import grpc

from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedObservation,
    map_robot_keys_to_lerobot_features,
)
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.transport import services_pb2, services_pb2_grpc

SERVER = "127.0.0.1:18080"
CKPT = "/root/code/lerobot/output_lerobot_train_20260704_140211/checkpoints/040000/pretrained_model"
TASK = "Grab the cube and place it into the cup"

camera_cfg = {
    "camera2": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30, fourcc="MJPG"),
    "camera1": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30, fourcc="MJPG"),
}

robot = SO101Follower(
    SO101FollowerConfig(port="COM5", id="zihao_follower_arm", cameras=camera_cfg)
)

# Only connect bus + cameras, NOT robot.connect() -> torque stays off.
robot.bus.connect()
for cam in robot.cameras.values():
    cam.connect()

try:
    time.sleep(1.0)
    raw_obs = robot.get_observation()
    raw_obs["task"] = TASK
    print("=== Observation keys ===")
    for k, v in raw_obs.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")

    # Build gRPC channel with official options (retry + keepalive)
    from lerobot.transport.utils import grpc_channel_options
    channel_opts = grpc_channel_options()
    channel_opts += [
        ("grpc.keepalive_time_ms", 5000),
        ("grpc.keepalive_timeout_ms", 20000),
        ("grpc.keepalive_permit_without_stream", 1),
        ("grpc.http2.max_pings_without_data", 0),
    ]
    channel = grpc.insecure_channel(SERVER, options=channel_opts)
    stub = services_pb2_grpc.AsyncInferenceStub(channel)

    print("=== Ready() handshake ===")
    stub.Ready(services_pb2.Empty(), timeout=30)

    print("=== SendPolicyInstructions (triggers model load, allow 120s) ===")
    features = map_robot_keys_to_lerobot_features(robot)
    pol_cfg = RemotePolicyConfig(
        policy_type="smolvla",
        pretrained_name_or_path=CKPT,
        lerobot_features=features,
        actions_per_chunk=10,
        device="cuda",
    )
    stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(pol_cfg)), timeout=120)

    # Model load takes ~10-20s on V100. Wait and poll GetActions.
    print("=== Waiting for first action chunk (model loading...) ===")
    t0 = time.time()

    obs = TimedObservation(timestamp=time.time(), observation=raw_obs, timestep=0, must_go=True)
    obs_bytes = pickle.dumps(obs)
    from lerobot.transport.utils import send_bytes_in_chunks
    obs_iter = send_bytes_in_chunks(obs_bytes, services_pb2.Observation, log_prefix="[OBS]", silent=True)
    stub.SendObservations(obs_iter)

    # GetActions is unary; poll until we get a non-empty chunk.
    print("=== Waiting for first action chunk ===")
    t0 = time.time()
    actions = None
    while time.time() - t0 < 60:
        resp = stub.GetActions(services_pb2.Empty(), timeout=30)
        if len(resp.data) > 0:
            actions = resp
            break
        time.sleep(0.5)

    if actions is None:
        print("TIMEOUT waiting for actions")
        sys.exit(1)

    print(f"=== Received action chunk in {time.time()-t0:.1f}s ===")
    payload = pickle.loads(actions.data)
    print(f"  type: {type(payload)}")
    if hasattr(payload, "action"):
        a = payload.action
        print(f"  action shape: {a.shape}")
        print(f"  action dtype: {a.dtype}")
        import torch
        finite = bool(torch.isfinite(a).all().item())
        print(f"  action all finite: {finite}")
        print(f"  action values:\n{a}")
    else:
        print(f"  payload: {payload}")

    print("=== PREDICTION-ONLY OK ===")

except Exception:
    traceback.print_exc()
finally:
    robot.bus.disconnect(disable_torque=True)
    for cam in robot.cameras.values():
        cam.disconnect()
