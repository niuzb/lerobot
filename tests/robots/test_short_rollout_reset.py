"""Reset logic tests with no serial connection or physical motion."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def rollout(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "short_rollout_reset_test", Path(__file__).parents[2] / "short_rollout.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "log_reset", Mock())
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))
    monkeypatch.setattr(module, "RESET_TIMEOUT_SECONDS", 1.0)
    return module


def robot_at(position):
    return SimpleNamespace(
        config=SimpleNamespace(max_target_velocity=3.0, max_tracking_error=3.0, num_read_retries=0),
        bus=SimpleNamespace(motors={"elbow_flex": None}, sync_read=Mock(return_value={"elbow_flex": position})),
        send_action=Mock(return_value={"elbow_flex.pos": 0.0}),
        _last_stable_present_pos={"elbow_flex": position},
    )


def test_capture_uses_measured_position(rollout):
    assert rollout.capture_initial_position(robot_at(12.5)) == {"elbow_flex.pos": 12.5}


@pytest.mark.parametrize("position", [float("nan"), float("inf")])
def test_capture_rejects_invalid_feedback(rollout, position):
    with pytest.raises(RuntimeError, match="initial"):
        rollout.capture_initial_position(robot_at(position))


def test_reset_requires_stable_measured_arrival(rollout):
    robot = robot_at(0.0)
    rollout.return_to_initial_position(robot, {"elbow_flex.pos": 0.0})
    assert robot.send_action.call_count >= 16
    assert "complete" in rollout.log_reset.call_args.args[0]


def test_sent_target_alone_does_not_count_as_arrival(rollout):
    robot = robot_at(10.0)
    with pytest.raises(TimeoutError):
        rollout.return_to_initial_position(robot, {"elbow_flex.pos": 0.0})


def test_reset_aborts_if_position_guard_rejects(rollout):
    robot = robot_at(0.0)
    robot.send_action.return_value = {}
    with pytest.raises(RuntimeError, match="guard"):
        rollout.return_to_initial_position(robot, {"elbow_flex.pos": 0.0})
    assert robot.send_action.call_count == 1


def test_reset_propagates_serial_failure(rollout):
    robot = robot_at(0.0)
    robot.send_action.side_effect = OSError("serial failure")
    with pytest.raises(OSError):
        rollout.return_to_initial_position(robot, {"elbow_flex.pos": 0.0})


@pytest.mark.parametrize("interrupted,thread_alive,should_reset", [(False, False, True), (True, False, False), (False, True, False)])
def test_runner_resets_only_after_normal_stop(rollout, monkeypatch, tmp_path, interrupted, thread_alive, should_reset):
    client = Mock()
    client.robot = robot_at(0.0)
    client.shutdown_event.wait.return_value = interrupted
    monkeypatch.setattr(rollout, "RobotClient", Mock(return_value=client))
    thread = Mock()
    thread.is_alive.return_value = thread_alive
    monkeypatch.setattr(rollout.threading, "Thread", Mock(return_value=thread))
    monkeypatch.setattr(rollout, "ACTION_LOG_PATH", tmp_path / "actions.log")
    reset = Mock()
    monkeypatch.setattr(rollout, "return_to_initial_position", reset)
    rollout.runner()
    assert reset.called == should_reset
    client.stop.assert_called_once()


def test_runner_disconnects_on_reset_failure(rollout, monkeypatch, tmp_path):
    client = Mock()
    client.robot = robot_at(0.0)
    client.shutdown_event.wait.return_value = False
    monkeypatch.setattr(rollout, "RobotClient", Mock(return_value=client))
    thread = Mock()
    thread.is_alive.return_value = False
    monkeypatch.setattr(rollout.threading, "Thread", Mock(return_value=thread))
    monkeypatch.setattr(rollout, "ACTION_LOG_PATH", tmp_path / "actions.log")
    monkeypatch.setattr(rollout, "return_to_initial_position", Mock(side_effect=TimeoutError("stuck")))
    with pytest.raises(TimeoutError):
        rollout.runner()
    client.stop.assert_called_once()


def test_camera_snapshots_every_five_seconds(rollout, monkeypatch, tmp_path):
    monkeypatch.setattr(rollout, "SNAPSHOT_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(rollout, "ACTION_LOG_PATH", tmp_path / "actions.log")
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :, 0] = 255
    observation = {"camera1": frame, "camera2": frame}
    rollout.save_camera_snapshots(observation)
    assert len(list(rollout.SNAPSHOT_DIR.glob("*.jpg"))) == 2
    rollout.time.sleep(4.9)
    rollout.save_camera_snapshots(observation)
    assert len(list(rollout.SNAPSHOT_DIR.glob("*.jpg"))) == 2
    rollout.time.sleep(0.1)
    rollout.save_camera_snapshots(observation)
    assert len(list(rollout.SNAPSHOT_DIR.glob("*.jpg"))) == 4
    saved = rollout.cv2.imread(str(next(rollout.SNAPSHOT_DIR.glob("*.jpg"))))
    assert saved[0, 0, 2] > 250  # RGB red remains red in OpenCV's BGR output.
    assert saved[0, 0, 0] < 5
