"""Direct leader-angle requests; no hardware access in this module."""

import uuid

import numpy as np


def joint_vector(value):
    """Copy a finite six-axis measurement in degrees and gripper percent."""
    result = np.asarray(value, dtype=float)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError("Expected six finite joint values")
    return result.copy()


class DirectPoseControl:
    """Follow the leader's absolute angles, with no inferred alignment offset."""

    def __init__(self, lower, upper):
        """Start disarmed with the follower's calibrated joint bounds."""
        self.lower, self.upper = joint_vector(lower), joint_vector(upper)
        if np.any(self.lower >= self.upper):
            raise ValueError("Each lower limit must be below its upper limit")
        self.armed = False
        self.token = None
        self.hold = None
        self.deadline = 0.0
        self.last_tick = None

    def arm(self, follower):
        """Request holding the current feedback; this does not access hardware."""
        position = joint_vector(follower)
        if np.any(position < self.lower) or np.any(position > self.upper):
            raise ValueError("Current pose is outside the calibrated limits")
        self.hold = position
        self.armed = True
        self.token = None
        self.last_tick = None

    def begin(self, leader, follower, now):
        """Start direct following; the driver ramps from measured follower feedback."""
        if not self.armed:
            raise ValueError("先启用当前位置保持")
        joint_vector(leader)
        self.hold = joint_vector(follower)
        self.token = uuid.uuid4().hex
        self.deadline = now + 0.8
        self.last_tick = now
        return self.token

    def heartbeat(self, token, received_at, now):
        """Keep following while the operator page is alive; never restart implicitly."""
        if token != self.token or self.token is None or now >= self.deadline:
            raise ValueError("跟随已停止，请点击开始跟随")
        if not 0 <= now - received_at < 0.8:
            raise ValueError("跟随请求已过期")
        self.deadline = received_at + 0.8

    def pause(self, follower):
        """Stop following and request holding the measured follower pose."""
        self.token = None
        self.hold = joint_vector(follower)

    def disarm(self):
        """Stop producing goals after the operator releases the servos."""
        self.armed = False
        self.token = None
        self.hold = None

    def target(self, leader, follower, now):
        """Return absolute leader angles; the robot driver applies the speed ramp."""
        if not self.armed:
            return None
        current = joint_vector(follower)
        gap = 0.0 if self.last_tick is None else now - self.last_tick
        self.last_tick = now
        if self.token is not None and (now >= self.deadline or gap > 0.2):
            self.pause(current)
        if self.token is None:
            return self.hold.copy()
        return np.clip(joint_vector(leader), self.lower, self.upper)
