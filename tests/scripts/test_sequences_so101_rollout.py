"""Check the MolmoAct2 joint frame and the local no-motion/deadline boundaries."""

import importlib.util
import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "sequences_so101_rollout", Path(__file__).resolve().parents[2] / "scripts/policies/sequences/sequences_so101_rollout.py"
)
rollout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rollout)

BOUNDS = np.array([(-180, 180)] * 5 + [(0, 100)])
REST = np.array([-0.53, -99.14, 91.41, 60.64, -3.60, 1.10])


def test_official_rest_pose_and_inverse():
    expected = [-0.53, 189.14, 181.41, 60.64, -3.60, 1.10]
    np.testing.assert_allclose(rollout.to_model_state(REST), expected, atol=1e-5)
    answer = {"actions": np.tile(expected, (30, 1))}
    np.testing.assert_allclose(rollout.to_robot_actions(answer, BOUNDS), np.tile(REST, (30, 1)))


@pytest.mark.parametrize("state", [[0] * 5, [0] * 7, [0, 0, float("nan"), 0, 0, 0]])
def test_invalid_state(state):
    with pytest.raises(ValueError):
        rollout.to_model_state(state)


@pytest.mark.parametrize("defect", ["shape", "nan", "joint_limit", "missing"])
def test_entire_bad_chunk_is_rejected(defect):
    raw = np.tile(rollout.to_model_state(REST), (30, 1))
    answer = {"actions": raw}
    if defect == "shape":
        answer["actions"] = raw[:29]
    elif defect == "missing":
        answer = {}
    else:
        raw[-1, 0] = float("nan") if defect == "nan" else 500
    with pytest.raises(ValueError):
        rollout.to_robot_actions(answer, BOUNDS)


def test_gripper_endpoint_saturation_matches_motor_driver():
    raw = np.tile(rollout.to_model_state(REST), (30, 1))
    raw[0, 5], raw[1, 5] = -0.126, 100.126
    actions = rollout.to_robot_actions({"actions": raw}, BOUNDS)
    assert actions[0, 5] == 0
    assert actions[1, 5] == 100
    np.testing.assert_allclose(actions[:, :5], np.tile(REST[:5], (30, 1)), atol=1e-5)


def test_camera_mapping_rgb_jpeg_and_state_order():
    observation = dict(zip(rollout.ACTION_KEYS, REST, strict=True))
    observation.update(
        front=np.full((480, 640, 3), [255, 0, 0], dtype=np.uint8),
        side=np.full((480, 640, 3), [0, 0, 255], dtype=np.uint8),
    )
    wire = rollout.make_observation(observation, "pick up tape")
    assert set(wire) == {"state", "prompt", "top_image", "side_image"}
    assert wire["prompt"] == "pick up tape"
    decoded = cv2.imdecode(np.frombuffer(wire["top_image"], np.uint8), cv2.IMREAD_COLOR)
    assert decoded[0, 0, 2] > 250  # RGB red was encoded as BGR red, not blue.
    assert decoded[0, 0, 0] < 5
    np.testing.assert_allclose(wire["state"], rollout.to_model_state(REST))


def test_timeout_interrupts_request_without_replaying():
    release = threading.Event()
    policy = Mock()
    policy.interrupt.side_effect = release.set
    with pytest.raises(TimeoutError):
        rollout.bounded_call(policy, lambda: release.wait(2), 0.01)
    policy.interrupt.assert_called_once()


def test_cloud_failure_is_propagated():
    def fail():
        raise ValueError("cloud failed")

    with pytest.raises(ValueError, match="cloud failed"):
        rollout.bounded_call(Mock(), fail, 1)


def test_late_motor_write_skips_ticks_and_run_deadline_stops():
    now = [0.0]

    def send(action):
        now[0] += 0.08  # Slower than two nominal 30 Hz ticks.
        return action

    def sleep(seconds):
        now[0] += seconds

    robot = SimpleNamespace(send_action=Mock(side_effect=send), _last_stable_present_pos={})
    rows = np.tile(np.arange(30)[:, None], (1, 6))
    log = io.StringIO()
    count = rollout.play_chunk(robot, rows, 0.17, log, clock=lambda: now[0], sleep=sleep)
    assert count == 3
    assert [call.args[0]["shoulder_pan.pos"] for call in robot.send_action.call_args_list] == [0, 2, 4]
    robot.send_action.reset_mock()
    assert rollout.play_chunk(robot, rows, 0.1, log, clock=lambda: now[0], sleep=sleep) == 0
    robot.send_action.assert_not_called()


@pytest.mark.parametrize("invalid_answer", [False, True])
def test_preflight_reads_only_and_closes_hardware(monkeypatch, tmp_path, invalid_answer):
    import sequence_ai

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SEQUENCES_API_KEY", "test-credential-never-sent")
    bus = SimpleNamespace(
        is_connected=False,
        motors={name: SimpleNamespace(model="sts3215") for name in rollout.JOINTS},
        model_resolution_table={"sts3215": 4096},
        sync_write=Mock(),
    )
    bus.connect = Mock(side_effect=lambda: setattr(bus, "is_connected", True))
    bus.disconnect = Mock(side_effect=lambda **_: setattr(bus, "is_connected", False))
    camera = SimpleNamespace(is_connected=False)
    camera.connect = Mock(side_effect=lambda: setattr(camera, "is_connected", True))
    camera.disconnect = Mock(side_effect=lambda: setattr(camera, "is_connected", False))
    observation = dict(zip(rollout.ACTION_KEYS, REST, strict=True))
    observation.update(
        front=np.zeros((480, 640, 3), dtype=np.uint8), side=np.zeros((480, 640, 3), dtype=np.uint8)
    )
    robot = SimpleNamespace(
        bus=bus,
        is_calibrated=True,
        cameras={"front": camera, "side": camera},
        calibration={name: SimpleNamespace(range_min=0, range_max=4095) for name in rollout.JOINTS},
        get_observation=Mock(return_value=observation),
        configure=Mock(),
        send_action=Mock(),
    )
    policy = Mock()
    policy.act.return_value = {"actions": np.tile(rollout.to_model_state(REST), (30, 1))}
    if invalid_answer:
        policy.act.return_value["actions"][-1, 0] = float("nan")
    monkeypatch.setattr(rollout, "SO101Follower", lambda _: robot)
    monkeypatch.setattr(sequence_ai, "connect", lambda **_: policy)
    args = SimpleNamespace(
        model="so101",
        port="COM5",
        robot_id="zihao_follower_arm",
        front_camera=0,
        side_camera=2,
        task="pick up tape",
        execute=False,
        preflight_rounds=3,
    )
    if invalid_answer:
        with pytest.raises(ValueError, match="finite"):
            rollout.run(args)
    else:
        rollout.run(args)
    assert policy.act.call_count == (1 if invalid_answer else 3)
    robot.configure.assert_not_called()
    robot.send_action.assert_not_called()
    bus.sync_write.assert_not_called()
    bus.disconnect.assert_called_once_with(disable_torque=False)
    assert not camera.is_connected
    policy.close.assert_called_once()
    summary = json.loads(next((tmp_path / "outputs").glob("*/summary.json")).read_text())
    assert summary["status"] == ("failed" if invalid_answer else "ok")
    assert summary["actions_sent"] == 0
