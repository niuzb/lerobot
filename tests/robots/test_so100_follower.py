#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.so_follower import (
    SO100Follower,
    SO100FollowerConfig,
)


def _make_bus_mock() -> MagicMock:
    """Return a bus mock with just the attributes used by the robot."""
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect

    @contextmanager
    def _dummy_cm():
        yield

    bus.torque_disabled.side_effect = _dummy_cm

    return bus


@pytest.fixture
def follower(tmp_path):
    bus_mock = _make_bus_mock()

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        motors_order: list[str] = list(bus_mock.motors)

        bus_mock.sync_read.return_value = {motor: idx for idx, motor in enumerate(motors_order, 1)}
        bus_mock.sync_write.return_value = None
        bus_mock.write.return_value = None
        bus_mock.disable_torque.return_value = None
        bus_mock.enable_torque.return_value = None
        bus_mock.is_calibrated = True
        return bus_mock

    with (
        patch(
            "lerobot.robots.so_follower.so_follower.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch.object(SO100Follower, "configure", lambda self: None),
    ):
        cfg = SO100FollowerConfig(port="/dev/null", calibration_dir=tmp_path)
        robot = SO100Follower(cfg)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_connect_disconnect(follower):
    assert not follower.is_connected

    follower.connect()
    assert follower.is_connected

    follower.disconnect()
    assert not follower.is_connected


def test_get_observation(follower):
    follower.connect()
    obs = follower.get_observation()

    expected_keys = {f"{m}.pos" for m in follower.bus.motors}
    assert set(obs.keys()) == expected_keys

    for idx, motor in enumerate(follower.bus.motors, 1):
        assert obs[f"{motor}.pos"] == idx


def test_get_observation_uses_read_retries(follower):
    # Feetech buses can intermittently fail a sync_read; the follower should forward the configured
    # retry count so transient failures don't abort the control loop (see #3131).
    follower.config.num_read_retries = 7
    follower.connect()
    follower.get_observation()

    follower.bus.sync_read.assert_called_once_with("Present_Position", num_retry=7)


def test_send_action_uses_read_retries(follower):
    follower.config.max_relative_target = 10.0
    follower.config.num_read_retries = 7
    follower.connect()

    action = {f"{motor}.pos": value * 10 for value, motor in enumerate(follower.bus.motors, 1)}
    follower.send_action(action)

    follower.bus.sync_read.assert_called_once_with("Present_Position", num_retry=7)


def test_send_action_accepts_large_feedback_change(follower):
    follower.connect()
    present = dict.fromkeys(follower.bus.motors, 0.0)
    follower.bus.sync_read.side_effect = [present, {**present, "elbow_flex": 60.0}]
    follower.send_action({"elbow_flex.pos": 1.0})
    follower.send_action({"elbow_flex.pos": 2.0})
    assert follower.bus.sync_write.call_count == 2
    assert follower._last_stable_present_pos["elbow_flex"] == 60.0


def test_observation_fault_never_returns_cached_pose_and_blocks_unlimited_actions(follower):
    follower.connect()
    present = dict.fromkeys(follower.bus.motors, 0.0)
    follower.bus.sync_read.return_value = present
    follower.get_observation()
    follower.bus.sync_read.return_value = {**present, "elbow_flex": float("nan")}
    with pytest.raises(RuntimeError, match="Feedback invalid"):
        follower.get_observation()
    follower.bus.sync_read.return_value = present
    with pytest.raises(RuntimeError, match="reconnect"):
        follower.get_observation()
    with pytest.raises(RuntimeError, match="reconnect"):
        follower.send_action({"elbow_flex.pos": 1.0})
    follower.bus.sync_write.assert_not_called()
    follower.disconnect()
    follower.connect()
    assert follower.get_observation()["elbow_flex.pos"] == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), None])
def test_invalid_feedback_rejected_before_first_observation(follower, bad):
    follower.connect()
    present = dict.fromkeys(follower.bus.motors, 0.0)
    if bad is None:
        del present["gripper"]
    else:
        present["gripper"] = bad
    follower.bus.sync_read.return_value = present
    with pytest.raises(RuntimeError, match="Present_Position"):
        follower.get_observation()
    assert follower._last_stable_present_pos is None
    with pytest.raises(RuntimeError):
        follower.send_action({"elbow_flex.pos": 1.0})
    follower.bus.sync_write.assert_not_called()


def test_read_failure_latches_fault(follower):
    follower.connect()
    follower.bus.sync_read.side_effect = ConnectionError("bad packet")
    with pytest.raises(RuntimeError, match="read failed"):
        follower.get_observation()
    with pytest.raises(RuntimeError, match="reconnect"):
        follower.send_action({"elbow_flex.pos": 1.0})
    follower.bus.sync_write.assert_not_called()


def test_valid_motion_is_reported_and_stationary_pose_is_allowed(follower):
    follower.connect()
    for pos in [0.0, 0.0, 3.0, 7.0]:
        follower.bus.sync_read.return_value = dict.fromkeys(follower.bus.motors, pos)
        assert set(follower.get_observation().values()) == {pos}


def test_reported_gripper_change_returns_real_measurement(follower):
    follower.connect()
    present = dict.fromkeys(follower.bus.motors, 0.0)
    follower.bus.sync_read.return_value = {**present, "gripper": 18.955223880597014}
    follower.get_observation()
    follower.bus.sync_read.return_value = {**present, "gripper": 13.73134328358209}
    assert follower.get_observation()["gripper.pos"] == 13.73134328358209
    assert follower._feedback_fault is None


@pytest.mark.parametrize("motor,delta", [("gripper", 80.0), ("elbow_flex", 120.0)])
def test_large_changes_return_actual_feedback_despite_legacy_limits(follower, motor, delta):
    follower.config.max_position_jump = 0.1
    follower.config.max_gripper_position_jump = 0.1
    follower.connect()
    present = dict.fromkeys(follower.bus.motors, 0.0)
    for position in [0.0, delta, -delta, 0.0]:
        follower.bus.sync_read.return_value = {**present, motor: position}
        assert follower.get_observation()[f"{motor}.pos"] == position
        assert follower._last_stable_present_pos[motor] == position
        assert follower._feedback_fault is None


def test_send_action(follower):
    follower.connect()

    action = {f"{m}.pos": i * 10 for i, m in enumerate(follower.bus.motors, 1)}
    returned = follower.send_action(action)

    assert returned == action

    goal_pos = {m: (i + 1) * 10 for i, m in enumerate(follower.bus.motors)}
    follower.bus.sync_write.assert_called_once_with("Goal_Position", goal_pos)


def test_configure_writes_position_pid_coefficients():
    bus_mock = _make_bus_mock()
    bus_mock.motors = ["shoulder_pan"]
    robot = MagicMock()
    robot.bus = bus_mock
    robot.config = SO100FollowerConfig(
        port="/dev/null",
        position_p_coefficient=32,
        position_i_coefficient=1,
        position_d_coefficient=16,
    )

    SO100Follower.configure(robot)

    bus_mock.write.assert_any_call("P_Coefficient", "shoulder_pan", 32)
    bus_mock.write.assert_any_call("I_Coefficient", "shoulder_pan", 1)
    bus_mock.write.assert_any_call("D_Coefficient", "shoulder_pan", 16)


def _enable_ramp(follower):
    follower.config.max_target_velocity = 3.0
    follower.config.max_tracking_error = 1.0
    follower._validate_ramp_config()
    follower.connect()
    follower.bus.sync_read.return_value = dict.fromkeys(follower.bus.motors, 0.0)


def test_ramp_advances_with_time_and_saturates_without_windup(follower):
    """Stationary feedback does not pin the target, but bounds accumulated error."""
    _enable_ramp(follower)
    action = {f"{m}.pos": 10.0 for m in follower.bus.motors}
    with patch("lerobot.robots.so_follower.so_follower.time.monotonic") as clock:
        for t, expected in [(0, 0), (0.1, 0.3), (0.2, 0.6), (0.3, 0.9), (0.4, 1), (1, 1)]:
            clock.return_value = t
            sent = follower.send_action(action)
            assert list(sent.values()) == pytest.approx([expected] * 6)
        clock.return_value = 1.1
        reversed_action = dict.fromkeys(action, -10.0)
        assert list(follower.send_action(reversed_action).values()) == pytest.approx([0.7] * 6)


def test_ramp_pause_and_reconnect(follower):
    """A long gap permits only a capped step; reconnect starts at measured pose."""
    _enable_ramp(follower)
    action = {f"{m}.pos": 10.0 for m in follower.bus.motors}
    with patch("lerobot.robots.so_follower.so_follower.time.monotonic") as clock:
        clock.return_value = 0
        follower.send_action(action)
        clock.return_value = 50
        assert list(follower.send_action(action).values()) == pytest.approx([0.3] * 6)
        follower.disconnect()
        follower.connect()
        assert list(follower.send_action(action).values()) == [0.0] * 6


def test_ramp_tracks_and_stops_at_requested_target(follower):
    """Following feedback allows continued progress without overshooting the model target."""
    _enable_ramp(follower)
    action = {f"{m}.pos": 1.25 for m in follower.bus.motors}
    with patch("lerobot.robots.so_follower.so_follower.time.monotonic") as clock:
        for i in range(10):
            clock.return_value = i * 0.1
            sent = follower.send_action(action)
            assert list(sent.values()) == pytest.approx([min(i * 0.3, 1.25)] * 6)
            follower.bus.sync_read.return_value = {k.removesuffix(".pos"): v for k, v in sent.items()}


def test_ramp_failed_write_does_not_advance_state(follower):
    """A failed write cannot become the next target ramp origin."""
    _enable_ramp(follower)
    follower.bus.sync_write.side_effect = ConnectionError("failed")
    with pytest.raises(ConnectionError):
        follower.send_action({"elbow_flex.pos": 3.0})
    assert follower._ramp_target is None
    assert follower._ramp_time is None


def test_ramp_rejects_nonfinite_action_and_impossible_tracking(follower):
    """Reject invalid targets or a feedback displacement that cannot satisfy both bounds."""
    _enable_ramp(follower)
    with pytest.raises(ValueError):
        follower.send_action({"elbow_flex.pos": float("nan")})
    follower.bus.sync_write.assert_not_called()
    with patch("lerobot.robots.so_follower.so_follower.time.monotonic", return_value=0):
        follower.send_action({"elbow_flex.pos": 3.0})
        follower.bus.sync_read.return_value = dict.fromkeys(follower.bus.motors, 2.0)
        with pytest.raises(RuntimeError, match="tracking envelope"):
            follower.send_action({"elbow_flex.pos": 3.0})
    assert follower.bus.sync_write.call_count == 1


@pytest.mark.parametrize("speed,error", [(0, 1), (1, None), (float("inf"), 1), (1, -1)])
def test_ramp_config_rejects_invalid_limits(follower, speed, error):
    follower.config.max_target_velocity = speed
    follower.config.max_tracking_error = error
    with pytest.raises(ValueError):
        follower._validate_ramp_config()
