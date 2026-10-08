"""Pi05 cloud inference, based on short_rollout.py.

Default: read-only camera/joint streaming and action validation.
Pass --execute for a bounded rollout using the existing robot motion guards.
"""
import argparse
import json
import pickle
import threading
import time
from pathlib import Path

import cv2
import grpc
import torch

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import (
    RemotePolicyConfig, TimedObservation, encode_timed_observation,
    map_robot_keys_to_lerobot_features,
)
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

CHECKPOINT = '/root/code/lerobot/outputs/train/pi05_tape_curated_20261004/checkpoints/080000/pretrained_model'
TASK = 'Put the tape into the pen holder'


def robot_config():
    return SO101FollowerConfig(
        port='COM5', id='zihao_follower_arm', use_degrees=True,
        max_target_velocity=5.0, max_tracking_error=3.0,
        cameras={name: OpenCVCameraConfig(
            index_or_path=index, width=640, height=480, fps=30,
            fourcc='MJPG', backend=700,
        ) for name, index in [('front', 0), ('side', 2)]},
    )


def validate_actions(actions):
    if not actions:
        raise RuntimeError('Server returned no actions')
    for action in actions:
        tensor = action.get_action()
        if tensor.shape != (6,) or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError('Server returned malformed/nonfinite actions')


def preflight(address, checkpoint):
    out = Path('outputs') / ('pi05_preflight_' + time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    robot = SO101Follower(robot_config())
    channel = grpc.insecure_channel(address, options=grpc_channel_options())
    try:
        grpc.channel_ready_future(channel).result(timeout=10)
        # Read directly: no robot.connect(), motor setup or torque writes.
        robot.bus.connect()
        if not robot.is_calibrated:
            raise RuntimeError('Existing calibration does not match robot')
        for camera in robot.cameras.values():
            camera.connect()
        stub = services_pb2_grpc.AsyncInferenceStub(channel)
        stub.Ready(services_pb2.Empty(), timeout=10)
        config = RemotePolicyConfig('pi05', checkpoint,
                                   map_robot_keys_to_lerobot_features(robot), 30, 'cuda')
        stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(config)), timeout=300)
        for i in range(3):
            obs = robot.get_observation()
            if i == 0:
                for name in ('front', 'side'):
                    cv2.imwrite(str(out / f'{name}.jpg'), cv2.cvtColor(obs[name], cv2.COLOR_RGB2BGR))
            obs['task'] = TASK
            wire = encode_timed_observation(TimedObservation(
                timestamp=time.time(), timestep=i * 30, observation=obs, must_go=True), 80)
            payload = pickle.dumps(wire)
            start = time.perf_counter()
            stub.SendObservations(send_bytes_in_chunks(payload, services_pb2.Observation, silent=True), timeout=5)
            result = stub.GetActions(services_pb2.Empty(), timeout=30)
            actions = pickle.loads(result.data)
            validate_actions(actions)
            print(json.dumps({'round': i, 'payload_kib': len(payload) / 1024,
                              'round_trip_ms': (time.perf_counter() - start) * 1000,
                              'actions': len(actions), 'first_action': actions[0].get_action().tolist()}), flush=True)
        print(f'PREFLIGHT OK: both images + joints -> CUDA Pi05 -> finite actions; no motion commands; snapshots={out.resolve()}', flush=True)
    finally:
        for camera in robot.cameras.values():
            if camera.is_connected:
                camera.disconnect()
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)
        channel.close()


class LoggedClient(RobotClient):
    def __init__(self, config, out):
        self.out = out
        self.last_received = None
        self.first_action = None
        self.last_action_time = None
        self.action_count = 0
        self.chunk_count = 0
        self.next_snapshot = 0
        super().__init__(config)

    def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
        validate_actions(incoming_actions)
        with (self.out / 'raw_predictions.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'received_at': time.monotonic(), 'chunk': self.chunk_count,
                                     'actions': [{'step': action.get_timestep(),
                                                  'timestamp': action.get_timestamp(),
                                                  'action': action.get_action().tolist()}
                                                 for action in incoming_actions]}) + '\n')
        super()._aggregate_action_queues(incoming_actions, aggregate_fn)
        self.last_received = time.monotonic()
        self.chunk_count += 1

    def _action_tensor_to_action_dict(self, action_tensor):
        action = super()._action_tensor_to_action_dict(action_tensor)
        # Called on the actual dequeued/merged action, before driver motion guards.
        self.pending_action_diagnostic = {
            'before_write': time.monotonic(),
            'merged_policy_action': action_tensor.tolist(),
            'requested_robot_action': dict(action),
        }
        return action

    def control_loop_action(self, verbose=False):
        if self.last_received is None or time.monotonic() - self.last_received > 2:
            raise RuntimeError('No fresh cloud actions for 2 seconds; stopping')
        result = super().control_loop_action(verbose)
        now = time.monotonic()
        if self.first_action is None:
            self.first_action = now
        self.last_action_time = now
        self.action_count += 1
        with (self.out / 'actions.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'time': now, 'step': self.latest_action,
                                     **self.pending_action_diagnostic,
                                     'sent': result, 'measured': self.robot._last_stable_present_pos}) + '\n')
        return result

    def control_loop_observation(self, task, verbose=False):
        obs = super().control_loop_observation(task, verbose)
        if obs is not None and time.monotonic() >= self.next_snapshot:
            for name in ('front', 'side'):
                cv2.imwrite(str(self.out / f'{time.monotonic():010.2f}_{name}.jpg'),
                            cv2.cvtColor(obs[name], cv2.COLOR_RGB2BGR))
            self.next_snapshot = time.monotonic() + 2
        return obs


def execute(address, checkpoint, seconds):
    out = Path('outputs') / ('pi05_remote_' + time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    cfg = RobotClientConfig(
        robot=robot_config(), server_address=address, policy_type='pi05',
        pretrained_name_or_path=checkpoint, policy_device='cuda', client_device='cpu',
        actions_per_chunk=30, chunk_size_threshold=0.5, fps=30,
        observation_jpeg_quality=80, policy_joint_units='robot',
        aggregate_fn_name='weighted_average', task=TASK,
    )
    client = LoggedClient(cfg, out)
    threads = []
    errors = []

    def guarded(fn):
        try:
            fn()
        except BaseException as exc:
            errors.append(exc)
        finally:
            client.shutdown_event.set()

    try:
        if not client.start():
            raise RuntimeError('Cloud model handshake failed')
        for fn in (client.receive_actions, lambda: client.control_loop(TASK)):
            thread = threading.Thread(target=guarded, args=(fn,), daemon=True)
            threads.append(thread)
            thread.start()
        start = time.monotonic()
        print(f'EXECUTING {seconds}s after first action; output={out.resolve()}', flush=True)
        while not client.shutdown_event.wait(0.05):
            now = time.monotonic()
            if client.first_action is not None and now - client.first_action >= seconds:
                break
            if client.first_action is None and now - start > 30:
                raise TimeoutError('No first action within 30s')
            if client.last_received is not None and now - client.last_received > 2:
                raise TimeoutError('Cloud action stream stalled for 2s')
        if errors:
            raise RuntimeError('Client thread failed') from errors[0]
    finally:
        client.shutdown_event.set()
        if len(threads) > 1:
            threads[1].join(timeout=3)
        client.stop()
        for thread in threads:
            thread.join(timeout=2)
        summary = {'actions': client.action_count, 'chunks': client.chunk_count,
                   'observations_sent': client._observation_sent_count,
                   'motion_seconds': 0 if client.first_action is None else client.last_action_time - client.first_action}
        (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f'STOPPED, robot disconnected: {summary}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default='127.0.0.1:18082')
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not 0 < args.seconds <= 300:
        parser.error('--seconds must be between 0 and 300')
    torch.set_num_threads(2)
    if args.execute:
        execute(args.server, args.checkpoint, args.seconds)
    else:
        preflight(args.server, args.checkpoint)
