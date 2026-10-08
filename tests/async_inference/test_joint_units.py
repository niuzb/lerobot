"""Pure conversion and client-boundary tests; no serial port is opened."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lerobot.async_inference.joint_units import JOINTS, SOJointUnitConverter
from lerobot.async_inference.robot_client import RobotClient
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


@pytest.fixture
def bus():
    spans = [1781, 2256, 2090, 2233, 4095, 808]
    return FeetechMotorsBus(
        "UNOPENED",
        {
            n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if n == "gripper" else MotorNormMode.DEGREES)
            for i, n in enumerate(JOINTS)
        },
        calibration={n: MotorCalibration(i + 1, 0, 0, 0, spans[i]) for i, n in enumerate(JOINTS)},
    )


def test_round_trip_and_preserves_images(bus):
    converter = SOJointUnitConverter(bus)
    policy = {f"{n}.pos": 30.0 for n in JOINTS}
    robot = converter.action_to_robot(policy)
    assert robot["wrist_roll.pos"] == pytest.approx(54.0)
    assert robot["shoulder_pan.pos"] == pytest.approx(23.4857142857)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    observation = {**robot, "camera1": frame, "task": "tape"}
    converted = converter.observation_to_policy(observation)
    for key in policy:
        assert converted[key] == pytest.approx(policy[key])
    assert converted["camera1"] is frame
    assert converted["task"] == "tape"
    assert observation["wrist_roll.pos"] == pytest.approx(54.0)


@pytest.mark.parametrize("reverse", [False, True])
def test_matches_bus_normalization_without_quantization(bus, reverse):
    for c in bus.calibration.values():
        c.drive_mode = int(reverse)
    converter = SOJointUnitConverter(bus)
    for fraction in [0.0, 0.2, 0.5, 0.8, 1.0]:
        raw = {
            c.id: int(c.range_min + fraction * (c.range_max - c.range_min)) for c in bus.calibration.values()
        }
        degrees = bus._normalize(raw)
        observation = {f"{n}.pos": degrees[bus.motors[n].id] for n in JOINTS}
        for n in JOINTS[:-1]:
            bus.motors[n].norm_mode = MotorNormMode.RANGE_M100_100
        expected = bus._normalize(raw)
        actual = converter.observation_to_policy(observation)
        for n in JOINTS:
            assert actual[f"{n}.pos"] == pytest.approx(expected[bus.motors[n].id])
        assert converter.action_to_robot(actual) == pytest.approx(observation)
        for n in JOINTS[:-1]:
            bus.motors[n].norm_mode = MotorNormMode.DEGREES


def test_bounds_and_nonfinite(bus):
    converter = SOJointUnitConverter(bus)
    for value in (-150.0, 150.0):
        result = converter.action_to_robot({f"{n}.pos": value for n in JOINTS})
        assert abs(result["wrist_roll.pos"]) == 180.0
        assert result["gripper.pos"] == (0.0 if value < 0 else 100.0)
    for value in (float("nan"), float("inf"), -float("inf")):
        action = {f"{n}.pos": value for n in JOINTS}
        with pytest.raises(ValueError):
            converter.action_to_robot(action)
        with pytest.raises(ValueError):
            converter.observation_to_policy(action)
    with pytest.raises(ValueError):
        converter.action_to_robot({})


@pytest.mark.parametrize("failure", ["span", "id", "mode", "missing"])
def test_rejects_invalid_calibration_or_double_conversion(bus, failure):
    if failure == "span":
        bus.calibration["wrist_roll"].range_max = 0
    elif failure == "id":
        bus.calibration["wrist_roll"].id = 1
    elif failure == "mode":
        bus.motors["wrist_roll"].norm_mode = MotorNormMode.RANGE_M100_100
    else:
        del bus.calibration["wrist_roll"]
    with pytest.raises(ValueError):
        SOJointUnitConverter(bus)


def test_client_converts_before_enqueue_and_before_send(bus):
    client = RobotClient.__new__(RobotClient)
    client.joint_unit_converter = SOJointUnitConverter(bus)
    actual = {f"{n}.pos": 0.0 for n in JOINTS}
    actual["wrist_roll.pos"] = 54.0
    client.robot = SimpleNamespace(
        action_features=dict.fromkeys(actual, float), get_observation=Mock(return_value=actual)
    )
    client.latest_action_lock = threading.Lock()
    client.latest_action = 0
    client.action_queue_lock = threading.Lock()
    client.action_queue = Mock()
    client.action_queue.empty.return_value = True
    client.must_go = threading.Event()
    client.must_go.set()
    client.enqueue_observation = Mock(return_value=True)
    result = client.control_loop_observation("tape")
    queued = client.enqueue_observation.call_args.args[0].observation
    assert queued["wrist_roll.pos"] == pytest.approx(30)
    assert result["wrist_roll.pos"] == 54
    action = client._action_tensor_to_action_dict(torch.full((6,), 30.0))
    assert action["wrist_roll.pos"] == pytest.approx(54)
    client.joint_unit_converter = None
    assert client._action_tensor_to_action_dict(torch.full((6,), 30.0))["wrist_roll.pos"] == 30
    with pytest.raises(ValueError):
        client._action_tensor_to_action_dict(torch.zeros(7))


def test_converted_target_still_obeys_degree_ramp_and_tracking_limit(bus, monkeypatch):
    from lerobot.robots.so_follower.so_follower import SOFollower

    robot = SOFollower.__new__(SOFollower)
    limits = {n: 1.0 if n == "wrist_roll" else 3.0 for n in JOINTS}
    robot.config = SimpleNamespace(
        max_target_velocity=limits,
        max_tracking_error=limits,
        max_relative_target=None,
        target_ramp_max_dt=0.1,
        max_position_jump=5.0,
        max_gripper_position_jump=10.0,
        num_read_retries=0,
        feedback_reference_dt=1 / 30,
        feedback_max_dt=0.1,
    )
    robot.cameras = {}
    robot.bus = SimpleNamespace(
        is_connected=True,
        sync_read=Mock(return_value=dict.fromkeys(JOINTS, 0.0)),
        sync_write=Mock(),
        motors=dict.fromkeys(JOINTS),
    )
    robot._last_stable_present_pos = None
    robot._feedback_fault = None
    robot._ramp_target = None
    robot._ramp_time = None
    clock = [0.0]
    monkeypatch.setattr("lerobot.robots.so_follower.so_follower.time.monotonic", lambda: clock[0])
    target = SOJointUnitConverter(bus).action_to_robot({f"{n}.pos": 30.0 for n in JOINTS})
    assert target["wrist_roll.pos"] == 54.0
    assert robot.send_action(target)["wrist_roll.pos"] == 0.0
    clock[0] = 0.1
    sent = robot.send_action(target)
    assert sent["wrist_roll.pos"] == pytest.approx(0.1)
    assert sent["elbow_flex.pos"] == pytest.approx(0.3)
    for _ in range(40):
        clock[0] += 0.1
        sent = robot.send_action(target)
    assert sent["wrist_roll.pos"] == pytest.approx(1.0)
    assert sent["elbow_flex.pos"] == pytest.approx(3.0)


def test_invalid_observation_stops_before_enqueue(bus):
    client = RobotClient.__new__(RobotClient)
    client.joint_unit_converter = SOJointUnitConverter(bus)
    client.robot = SimpleNamespace(
        get_observation=Mock(return_value={f"{n}.pos": float("nan") for n in JOINTS})
    )
    client.shutdown_event = threading.Event()
    client.enqueue_observation = Mock()
    with pytest.raises(ValueError):
        client.control_loop_observation("tape")
    assert client.shutdown_event.is_set()
    client.enqueue_observation.assert_not_called()
