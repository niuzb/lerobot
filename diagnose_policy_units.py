"""Read-only unit comparison: no torque, calibration, or goal register writes."""

import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import grpc

from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedObservation,
    encode_timed_observation,
    map_robot_keys_to_lerobot_features,
)
from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks


def main():
    """Compare identical images with degree and range-normalized state inputs."""
    output = Path('outputs') / time.strftime('unit_diagnostic_%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    calibration_path = Path.home() / '.cache/huggingface/lerobot/calibration/robots/so_follower/zihao_follower_arm.json'
    calibration = {k: MotorCalibration(**v) for k, v in json.loads(calibration_path.read_text()).items()}
    names = list(calibration)
    motors = {
        n: Motor(c.id, 'sts3215', MotorNormMode.RANGE_0_100 if n == 'gripper' else MotorNormMode.DEGREES)
        for n, c in calibration.items()
    }
    bus = FeetechMotorsBus('COM5', motors, calibration=calibration)
    cameras = make_cameras_from_configs({
        'camera2': OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30, fourcc='MJPG'),
        'camera1': OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30, fourcc='MJPG'),
    })
    report = {'calibration': {n: asdict(c) for n, c in calibration.items()}, 'joint_order': names}
    try:
        bus.connect(handshake=False)
        bus.set_baudrate(1000000)
        torque = {n: bus.read('Torque_Enable', n, normalize=False) for n in names}
        report['torque_before'] = torque
        if any(torque.values()):
            raise RuntimeError('Expected all torque disabled; refusing diagnostic')
        hardware_calibration = bus.read_calibration()
        if hardware_calibration != calibration:
            raise RuntimeError('Hardware/file calibration mismatch; no changes made')
        raw = bus.sync_read('Present_Position', normalize=False, num_retry=2)
        report['raw_position'] = raw
        raw_ids = {calibration[n].id: v for n, v in raw.items()}
        states = {}
        for mode in (MotorNormMode.DEGREES, MotorNormMode.RANGE_M100_100):
            for n in names:
                if n != 'gripper':
                    bus.motors[n].norm_mode = mode
            values = bus._normalize(raw_ids)
            states[mode.value] = {f'{n}.pos': values[calibration[n].id] for n in names}
        report['states'] = states
        report['degrees_per_range_unit'] = {
            n: (c.range_max-c.range_min)*360/4095/200 for n, c in calibration.items() if n != 'gripper'
        }
        print(json.dumps(report, indent=2), flush=True)
        for camera in cameras.values():
            camera.connect()
        images = {n: camera.read_latest() for n, camera in cameras.items()}
        for n, frame in images.items():
            cv2.imwrite(str(output / f'{n}.jpg'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        features = map_robot_keys_to_lerobot_features(SimpleNamespace(observation_features={
            **{f'{n}.pos': float for n in names}, **{n: image.shape for n, image in images.items()}
        }))
        policy = RemotePolicyConfig(
            policy_type='smolvla',
            pretrained_name_or_path='/root/code/lerobot/output_lerobot_train_20260704_140211/checkpoints/040000/pretrained_model',
            lerobot_features=features, actions_per_chunk=10, device='cuda',
        )
        report['predictions'] = []
        with grpc.insecure_channel('127.0.0.1:18080', options=grpc_channel_options()) as channel:
            stub = services_pb2_grpc.AsyncInferenceStub(channel)
            for trial in range(3):
                for mode, state in states.items():
                    stub.Ready(services_pb2.Empty(), timeout=10)
                    stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(policy)), timeout=30)
                    obs = TimedObservation(
                        timestamp=time.time(), timestep=0, must_go=True,
                        observation={**images, **state, 'task': 'Grab tapes and place into pen holder'},
                    )
                    payload = pickle.dumps(encode_timed_observation(obs, jpeg_quality=80))
                    stub.SendObservations(send_bytes_in_chunks(payload, services_pb2.Observation, silent=True), timeout=10)
                    response = stub.GetActions(services_pb2.Empty(), timeout=30)
                    actions = pickle.loads(response.data)
                    row = {'trial': trial, 'mode': mode, 'actions': [a.action.tolist() for a in actions]}
                    report['predictions'].append(row)
                    print(json.dumps(row), flush=True)
        report['torque_after'] = {n: bus.read('Torque_Enable', n, normalize=False) for n in names}
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
        for camera in cameras.values():
            if camera.is_connected:
                camera.disconnect()
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Report: {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
