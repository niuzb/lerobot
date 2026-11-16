"""Read current robot state/cameras without motor setup or target writes."""
import json
import time
from pathlib import Path

import cv2
from lerobot.robots.so_follower import SO101Follower
from pi05_remote_rollout import robot_config

out = Path('outputs') / ('pi05_local_readonly_' + time.strftime('%Y%m%d_%H%M%S'))
out.mkdir(parents=True, exist_ok=False)
robot = SO101Follower(robot_config())
try:
    robot.bus.connect()
    assert robot.is_calibrated, 'Calibration does not match robot'
    for camera in robot.cameras.values():
        camera.connect()
    observation = robot.get_observation()
    state = {key: value for key, value in observation.items() if key.endswith('.pos')}
    (out / 'state.json').write_text(json.dumps(state, indent=2))
    for name in ['front', 'side']:
        cv2.imwrite(str(out / f'{name}.jpg'), cv2.cvtColor(observation[name], cv2.COLOR_RGB2BGR))
    print('LOCAL_READONLY_OK', json.dumps(state), str(out.resolve()), flush=True)
finally:
    for camera in robot.cameras.values():
        if camera.is_connected:
            camera.disconnect()
    if robot.bus.is_connected:
        robot.bus.disconnect(disable_torque=False)
