"""Explicit SO-101 policy range units / robot degree boundary conversion."""

import math
from collections.abc import Mapping
from typing import Any

from lerobot.motors import MotorNormMode

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class SOJointUnitConverter:
    """Use local calibration, without register writes or encoder quantization.

    Body policy coordinates use [-100, 100]; robot coordinates stay in degrees.
    Gripper coordinates are already [0, 100] at both boundaries. This does not
    correct physical calibration or camera differences between robots.
    """

    def __init__(self, bus):
        if set(bus.motors) != set(JOINTS) or set(bus.calibration) != set(JOINTS):
            raise ValueError("Unit conversion requires all six calibrated SO follower joints")
        self.scales = {}
        self.signs = {}
        for name in JOINTS:
            motor = bus.motors[name]
            expected = MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES
            if motor.norm_mode != expected:
                raise ValueError(f"{name}: unexpected driver units; refusing double conversion")
            calibration = bus.calibration[name]
            span = calibration.range_max - calibration.range_min
            if not math.isfinite(span) or span <= 0 or calibration.id != motor.id:
                raise ValueError(f"{name}: invalid calibration span or motor ID")
            resolution = bus.model_resolution_table[motor.model] - 1
            if resolution <= 0:
                raise ValueError(f"{name}: invalid encoder resolution")
            self.scales[name] = span * 360 / resolution / 200
            self.signs[name] = -1 if bus.apply_drive_mode and calibration.drive_mode else 1

    @staticmethod
    def _finite(value: float, name: str) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name}: non-finite joint value")
        return value

    def observation_to_policy(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Copy observation; convert body state, leaving images/task and gripper intact."""
        result = dict(observation)
        for name in JOINTS:
            key = f"{name}.pos"
            value = self._finite(observation[key], name)
            if name != "gripper":
                value = max(-100.0, min(100.0, value / self.scales[name])) * self.signs[name]
            result[key] = value
        return result

    def action_to_robot(self, action: Mapping[str, float]) -> dict[str, float]:
        """Bound policy coordinates, then convert before degree-based motion protection."""
        if set(action) != {f"{name}.pos" for name in JOINTS}:
            raise ValueError("Policy action must contain exactly six named SO joints")
        result = {}
        for name in JOINTS:
            key = f"{name}.pos"
            value = self._finite(action[key], name)
            if name == "gripper":
                result[key] = max(0.0, min(100.0, value))
            else:
                result[key] = max(-100.0, min(100.0, value)) * self.signs[name] * self.scales[name]
        return result
