"""Geometry, robot units and success checks shared by the LeIsaac tape pilot."""

import hashlib
import json
import math
import xml.etree.ElementTree as ET  # nosec B405 - Only trusted local robot MJCF assets are parsed.
from pathlib import Path

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TASK = "Put the tape into the pen holder"
FACES = [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rotation(quaternion):
    """Scalar-first quaternion to a column-vector rotation matrix."""
    q = np.asarray(quaternion, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def transform(position=(0, 0, 0), quaternion=(1, 0, 0, 0)):
    value = np.eye(4)
    value[:3, :3] = rotation(quaternion)
    value[:3, 3] = position
    return value


class JointMap:
    """Retain the existing real-degrees/gripper-percent convention and offsets."""

    def __init__(self, mapping):
        self.sign = np.asarray(mapping["sign"])
        self.offset = np.asarray(mapping["offset_rad"])
        self.knots = np.asarray(mapping["gripper_knots"])

    def to_sim(self, state):
        state = np.asarray(state)
        return np.r_[
            np.deg2rad(state[:5]) * self.sign + self.offset,
            np.interp(state[5], self.knots[:, 0], self.knots[:, 1]),
        ]

    def to_robot(self, q):
        q = np.asarray(q)
        knots = self.knots if self.knots[-1, 1] > self.knots[0, 1] else self.knots[::-1]
        percent = np.interp(q[5], knots[:, 1], knots[:, 0])
        # Observations preserve small physical soft-limit overshoots.
        if q[5] < knots[0, 1]:
            percent = knots[0, 0] + (q[5] - knots[0, 1]) * np.diff(knots[:2, 0])[0] / np.diff(knots[:2, 1])[0]
        elif q[5] > knots[-1, 1]:
            percent = (
                knots[-1, 0] + (q[5] - knots[-1, 1]) * np.diff(knots[-2:, 0])[0] / np.diff(knots[-2:, 1])[0]
            )
        return np.r_[np.rad2deg((q[:5] - self.offset) / self.sign), percent].astype(np.float32)


class Kinematics:
    """FK of the existing robot asset; checked against the imported articulation."""

    def __init__(self, asset):
        self.tree = ET.parse(asset)  # nosec B314 - Trusted local robot asset, not an uploaded XML document.
        self.body = self.tree.getroot().find("worldbody/body[@name='base']")
        self.bounds = np.array(
            [
                np.fromstring(self.tree.getroot().find(f".//joint[@name='{name}']").get("range"), sep=" ")
                for name in JOINTS
            ]
        )
        self.tip = np.fromstring(
            self.tree.getroot().find(".//geom[@name='fixed_jaw_sph_tip1']").get("pos"), sep=" "
        )

    def frames(self, q):
        result = {}
        by_name = dict(zip(JOINTS, q, strict=True))

        def visit(body, parent):
            pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
            quat = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
            frame = parent @ transform(pos, quat)
            for joint in body.findall("joint"):
                angle = by_name[joint.get("name")]
                axis = np.fromstring(joint.get("axis", "0 0 1"), sep=" ")
                axis /= np.linalg.norm(axis)
                frame = frame @ transform(quaternion=np.r_[math.cos(angle / 2), axis * math.sin(angle / 2)])
            result[body.get("name")] = frame
            for child in body.findall("body"):
                visit(child, frame)

        visit(self.body, np.eye(4))
        return result

    def fingertip(self, q):
        frame = self.frames(q)["gripper"]
        return frame[:3, :3] @ self.tip + frame[:3, 3], frame[:3, :3]


def ring_vertices(inner, outer, height, count=32):
    for index in range(count):
        a, b = 2 * np.pi * np.array([index, index + 1]) / count
        yield np.array(
            [
                [r * np.cos(angle), r * np.sin(angle), z]
                for z in (-height / 2, height / 2)
                for r in (inner, outer)
                for angle in (a, b)
            ]
        )


def lattice_vertices(holder):
    """Same perforated ribbon construction as lerobot.sim2real.lattice."""
    mesh = holder["mesh"]
    radius = holder["inner_radius_m"] + holder["wall_m"] / 2
    low = holder["bottom_m"] + mesh["rim_height_m"]
    high = holder["height_m"] - mesh["rim_height_m"]
    for direction in (-1, 1):
        for start in range(mesh["strands_per_direction"]):
            phase = 2 * np.pi * start / mesh["strands_per_direction"]
            levels = np.linspace(low, high, mesh["segments_per_strand"] + 1)
            for z0, z1 in zip(levels[:-1], levels[1:], strict=True):
                theta = (
                    phase + direction * mesh["circumferential_slope"] * (np.array([z0, z1]) - low) / radius
                )
                radial = np.array([np.cos(theta.mean()), np.sin(theta.mean()), 0])
                endpoints = np.c_[radius * np.cos(theta), radius * np.sin(theta), [z0, z1]]
                across = np.cross(radial, endpoints[1] - endpoints[0])
                across /= np.linalg.norm(across)
                yield np.array(
                    [
                        point + r * radial * holder["wall_m"] / 2 + w * across * mesh["strand_width_m"] / 2
                        for point in endpoints
                        for r in (-1, 1)
                        for w in (-1, 1)
                    ]
                )


class SuccessTracker:
    """Preserve the old physical success gate; simulator contacts supply evidence."""

    def __init__(self, config, initial):
        self.config = config
        self.initial = np.asarray(initial).copy()
        self.grasped = self.lifted = self.carried = False
        self.stable = 0

    def update(
        self,
        position,
        orientation,
        velocity,
        gripper,
        fixed,
        moving,
        support,
        holder_position,
        holder_orientation,
    ):
        tape, holder = self.config["tape"], self.config["holder"]
        position = np.asarray(position)
        self.grasped |= bool(fixed and moving)
        self.lifted |= bool(self.grasped and position[2] - self.initial[2] > 0.03)
        self.carried |= bool(self.lifted and np.linalg.norm(position[:2] - self.initial[:2]) > 0.03)
        angles = np.arange(64) * np.pi / 32
        points = np.array(
            [
                [tape["outer_radius_m"] * np.cos(a), tape["outer_radius_m"] * np.sin(a), z]
                for a in angles
                for z in (-tape["height_m"] / 2, tape["height_m"] / 2)
            ]
        )
        points = (points @ rotation(orientation).T + position - holder_position) @ rotation(
            holder_orientation
        )
        contained = (
            np.linalg.norm(points[:, :2], axis=1).max() < holder["inner_radius_m"]
            and points[:, 2].max() < holder["height_m"]
            and points[:, 2].min() >= holder["bottom_m"] - 0.002
        )
        released = gripper >= self.config["generation"]["open_percent"] - 5 and not (fixed or moving)
        valid = (
            self.carried
            and contained
            and released
            and np.linalg.norm(velocity) < 0.02
            and support
            and rotation(holder_orientation)[2, 2] > 0.95
        )
        self.stable = self.stable + 1 if valid else 0
        return self.stable >= 30
