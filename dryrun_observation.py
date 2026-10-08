"""No-torque dry-run: connect bus + cameras only, read one observation.

Does NOT call robot.connect() / configure(), so torque stays off.
"""
import traceback

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

camera_cfg = {
    # camera2 = wrist camera, index 0 -- created FIRST
    "camera2": OpenCVCameraConfig(
        index_or_path=0, width=640, height=480, fps=30, fourcc="MJPG"
    ),
    # camera1 = external fixed camera, index 2 -- created SECOND
    "camera1": OpenCVCameraConfig(
        index_or_path=2, width=640, height=480, fps=30, fourcc="MJPG"
    ),
}

robot = SO101Follower(
    SO101FollowerConfig(
        port="COM5",
        id="zihao_follower_arm",
        cameras=camera_cfg,
    )
)

robot.bus.connect()
for camera in robot.cameras.values():
    camera.connect()

try:
    import time
    time.sleep(1.0)  # let cameras warm up
    observation = robot.get_observation()
    print("=== POSITIONS ===")
    for key, value in observation.items():
        if key.endswith(".pos"):
            print(f"  {key}: {value}")
    print("=== IMAGES ===")
    print(f"  camera1 shape: {observation['camera1'].shape}")
    print(f"  camera2 shape: {observation['camera2'].shape}")

    # Save a frame from each camera for visual inspection
    from PIL import Image
    Image.fromarray(observation["camera1"]).save(
        r"C:\Users\Admin\code\lerobot\outputs\dryrun_camera1.png"
    )
    Image.fromarray(observation["camera2"]).save(
        r"C:\Users\Admin\code\lerobot\outputs\dryrun_camera2.png"
    )
    print("=== FRAMES SAVED ===")
    print("dry-run OK, no torque enabled.")
except Exception:
    traceback.print_exc()
finally:
    robot.bus.disconnect(disable_torque=True)
    for camera in robot.cameras.values():
        camera.disconnect()
