"""Measured SO-101 scene, physical success checks and contact-based data generation."""

import math
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import cv2
import numpy as np

from lerobot.utils.import_utils import _mujoco_available, require_package

from .config import JOINTS, JointMap, validate_physics
from .lattice import add_lattice_wall

if TYPE_CHECKING or _mujoco_available:
    import mujoco


def numbers(value):
    """Serialize numeric XML attributes independently of locale."""
    return " ".join(str(float(x)) for x in np.asarray(value).ravel())


def add_ring(root, body, name, inner, outer, height, mass=None, rgba="0.08 0.08 0.08 1"):
    """A compound ring made from convex wedges; its hole remains physically open."""
    assets = root.find("asset")
    for i in range(32):
        a, b = 2 * math.pi * np.array([i, i + 1]) / 32
        vertices = [
            [r * math.cos(angle), r * math.sin(angle), z]
            for z in (-height / 2, height / 2)
            for r in (inner, outer)
            for angle in (a, b)
        ]
        mesh = f"{name}_mesh_{i}"
        ET.SubElement(assets, "mesh", name=mesh, vertex=numbers(vertices))
        attrs = {
            "name": f"{name}_{i}",
            "type": "mesh",
            "mesh": mesh,
            "rgba": rgba,
            "friction": "1 0.005 0.0001",
            "condim": "4",
        }
        if mass is not None:
            attrs["mass"] = str(mass / 32)
        ET.SubElement(body, "geom", **attrs)


def make_scene(config, path):
    """Build cameras and hollow task objects around the pinned SO-101 asset."""
    validate_physics(config)
    from pathlib import Path

    asset = Path(config["asset"])
    tree = ET.parse(asset)
    root = tree.getroot()
    root.find("compiler").set("meshdir", str((asset.parent / "assets").resolve()))
    root.find("option").set("timestep", str(1 / 300))
    experimental = config.get("simulation_parameters", {})
    if experimental:
        iterations = experimental.get("solver_iterations", 50)
        force = experimental.get("gripper_force_limit_nm", 0.12)
        if not isinstance(iterations, int) or not 10 <= iterations <= 100:
            raise ValueError("Invalid solver iteration count")
        if not np.isfinite(force) or not 0 < force <= 2.94:
            raise ValueError("Invalid experimental gripper force limit")
        root.find("option").set("iterations", str(iterations))
        root.find("./actuator/position[@name='gripper']").set("forcerange", numbers([-force, force]))
        if experimental.get("white_robot"):
            for material in root.findall("./asset/material"):
                color = np.fromstring(material.get("rgba", ""), sep=" ")
                if len(color) == 4 and np.allclose(color[:3], [1, 0.82, 0.12]):
                    material.set("rgba", "0.88 0.88 0.86 1")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    old = visual.find("global")
    if old is not None:
        visual.remove(old)
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    world = root.find("worldbody")
    for parent in root.iter():
        for camera in list(parent.findall("camera")):
            parent.remove(camera)
        for geom in list(parent.findall("geom")):
            if geom.get("mesh") == "ArduCam_M12Lens":
                parent.remove(geom)
    ET.SubElement(world, "light", name="task_light", pos="0 -0.3 1.5", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    table = config["table"]
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="box",
        pos=numbers([0, 0, -table["thickness_m"] / 2]),
        size=numbers([table["half_x_m"], table["half_y_m"], table["thickness_m"] / 2]),
        rgba="0.9 0.9 0.9 1",
    )
    tape = config["tape"]
    obj = ET.SubElement(world, "body", name="tape", pos=numbers(tape["position_m"]))
    ET.SubElement(obj, "freejoint", name="tape_free")
    add_ring(
        root, obj, "tape", tape["inner_radius_m"], tape["outer_radius_m"], tape["height_m"], tape["mass_kg"]
    )
    holder = config["holder"]
    cup = ET.SubElement(world, "body", name="holder", pos=numbers(holder["position_m"]))
    ET.SubElement(cup, "freejoint", name="holder_free")
    if holder.get("wall_style", "solid") == "diamond_mesh":
        # Wall ribbons receive 60%, two rims 20%, and the bottom 20% of the
        # supplied total mass. This mass distribution is an explicit approximation.
        add_lattice_wall(root, cup, holder, holder["mass_kg"] * 0.6)
        rim_height = holder["mesh"]["rim_height_m"]
        for name, z in (
            ("lower", holder["bottom_m"] + rim_height / 2),
            ("upper", holder["height_m"] - rim_height / 2),
        ):
            rim = ET.SubElement(cup, "body", pos=numbers([0, 0, z]))
            add_ring(
                root,
                rim,
                f"holder_wall_{name}_rim",
                holder["inner_radius_m"],
                holder["inner_radius_m"] + holder["wall_m"],
                rim_height,
                holder["mass_kg"] * 0.1,
                rgba="0.045 0.05 0.055 1",
            )
    else:
        walls = ET.SubElement(cup, "body", pos=numbers([0, 0, holder["height_m"] / 2]))
        add_ring(
            root,
            walls,
            "holder_wall",
            holder["inner_radius_m"],
            holder["inner_radius_m"] + holder["wall_m"],
            holder["height_m"],
            holder["mass_kg"] * 0.8,
        )
    ET.SubElement(
        cup,
        "geom",
        name="holder_bottom",
        type="cylinder",
        pos=numbers([0, 0, holder["bottom_m"] / 2]),
        size=numbers([holder["inner_radius_m"] + holder["wall_m"], holder["bottom_m"] / 2]),
        mass=str(holder["mass_kg"] * 0.2),
        rgba="0.1 0.1 0.1 1",
    )
    for name, camera in config["cameras"].items():
        parent = world if camera["parent"] == "world" else world.find(f".//body[@name='{camera['parent']}']")
        if parent is None:
            raise ValueError(f"Unknown camera parent {camera['parent']}")
        transform = np.asarray(camera["T_parent_cv"])
        rotation = transform[:3, :3] @ np.diag([1, -1, -1])
        k = np.asarray(camera["K"])
        ET.SubElement(
            parent,
            "camera",
            name=name,
            pos=numbers(transform[:3, 3]),
            xyaxes=numbers(np.r_[rotation[:, 0], rotation[:, 1]]),
            resolution="640 480",
            sensorsize="1 1",
            focalpixel=numbers([k[0, 0], k[1, 1]]),
            # MuJoCo uses an offset from image centre; OpenCV K uses top-left pixels.
            principalpixel=numbers([320 - k[0, 2], k[1, 2] - 240]),
        )
    tree.write(path, encoding="utf-8")


class SuccessTracker:
    """Require physical lift, transport, release and stable full containment."""

    def __init__(self, config, initial):
        """Remember the real physical starting position and reset success history."""
        self.config = config
        self.initial = np.asarray(initial)
        self.lifted = self.carried = self.grasped = False
        self.stable = 0

    def update(
        self,
        position,
        rotation,
        velocity,
        gripper,
        fixed_contact,
        moving_contact,
        support,
        holder_position=None,
        holder_rotation=None,
    ):
        """Count consecutive fully contained, released and supported control ticks."""
        tape, cup = self.config["tape"], self.config["holder"]
        position = np.asarray(position)
        if fixed_contact and moving_contact:
            self.grasped = True
        self.lifted |= bool(self.grasped and position[2] - self.initial[2] > 0.03)
        self.carried |= bool(self.lifted and np.linalg.norm(position[:2] - self.initial[:2]) > 0.03)
        # Check every outer wedge vertex, including the tilted tape's top and bottom.
        angles = np.arange(64) * math.pi / 32
        local = np.array(
            [
                [tape["outer_radius_m"] * math.cos(a), tape["outer_radius_m"] * math.sin(a), z]
                for a in angles
                for z in [-tape["height_m"] / 2, tape["height_m"] / 2]
            ]
        )
        points = local @ np.asarray(rotation).T + position
        holder_position = np.asarray(cup["position_m"] if holder_position is None else holder_position)
        holder_rotation = np.eye(3) if holder_rotation is None else holder_rotation
        relative = (points - holder_position) @ holder_rotation
        contained = (
            np.linalg.norm(relative[:, :2], axis=1).max() < cup["inner_radius_m"]
            and relative[:, 2].max() < cup["height_m"]
            and relative[:, 2].min() >= cup["bottom_m"] - 0.002
        )
        released = gripper >= self.config["generation"]["open_percent"] - 5 and not (
            fixed_contact or moving_contact
        )
        settled = np.linalg.norm(velocity) < 0.02
        valid = (
            self.carried and contained and released and settled and support and holder_rotation[2, 2] > 0.95
        )
        self.stable = self.stable + 1 if valid else 0
        return self.stable >= 30


class Simulator:
    """Isolated MuJoCo state with explicit calibration and no hardware imports."""

    def __init__(self, config, scene_path):
        """Compile a measured scene and resolve joint, actuator and camera indices."""
        require_package("mujoco", extra="sim2real")
        self.config = config
        self.mapping = JointMap(config["joint_map"])
        make_scene(config, scene_path)
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.qidx = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in JOINTS])
        self.didx = np.array([self.model.jnt_dofadr[self.model.joint(n).id] for n in JOINTS])
        self.aidx = np.array([self.model.actuator(n).id for n in JOINTS])
        # Both joint and actuator bounds apply; reject rather than silently changing targets.
        joint_bounds = self.model.jnt_range[[self.model.joint(n).id for n in JOINTS]]
        control_bounds = self.model.actuator_ctrlrange[self.aidx]
        self.bounds = np.c_[
            np.maximum(joint_bounds[:, 0], control_bounds[:, 0]),
            np.minimum(joint_bounds[:, 1], control_bounds[:, 1]),
        ]
        self.object_id = self.model.body("tape").id
        self.free_q = self.model.jnt_qposadr[self.model.joint("tape_free").id]
        self.site = self.model.site("gripperframe").id
        self.renderer = None
        self.distortion_maps = {}

    def check_target(self, q):
        """Reject targets outside the intersection of joint and actuator bounds."""
        if np.any(q < self.bounds[:, 0] - 1e-5) or np.any(q > self.bounds[:, 1] + 1e-5):
            raise ValueError("Target exceeds simulator joint/actuator limits; review calibration")

    def set_pose(self, state):
        """Set a reference pose for kinematic playback, never for success training."""
        q = self.mapping.to_sim(state)
        self.check_target(q)
        self.data.qpos[self.qidx] = q
        self.data.qvel[:] = 0
        self.data.ctrl[self.aidx] = q
        mujoco.mj_forward(self.model, self.data)

    def reset(self, state, seed=None, randomize=False):
        """Initialize an independent episode and optionally randomize its starting layout."""
        mujoco.mj_resetData(self.model, self.data)
        self.set_pose(state)
        if randomize:
            rng = np.random.default_rng(seed)
            ranges = np.asarray(self.config["generation"]["xy_range_m"])
            self.data.qpos[self.free_q : self.free_q + 2] += rng.uniform(-ranges, ranges)
            yaw = rng.uniform(
                -self.config["generation"]["yaw_range_rad"], self.config["generation"]["yaw_range_rad"]
            )
            self.data.qpos[self.free_q + 3 : self.free_q + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            self.model.light_diffuse[:] = rng.uniform(0.65, 0.95)
            for i in range(self.model.ngeom):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
                if name.startswith("tape_"):
                    self.model.geom_friction[i] = [rng.uniform(0.8, 1.2), 0.005, 0.0001]
        self.randomization = {
            "mujoco_version": mujoco.__version__,
            "seed": seed,
            "tape_initial_qpos": self.data.qpos[self.free_q : self.free_q + 7].tolist(),
            "light_diffuse": self.model.light_diffuse.tolist(),
            "tape_friction": {
                self.model.geom(i).name: self.model.geom_friction[i].tolist()
                for i in range(self.model.ngeom)
                if (self.model.geom(i).name or "").startswith("tape_")
            },
        }
        mujoco.mj_forward(self.model, self.data)
        for _ in range(150):
            mujoco.mj_step(self.model, self.data)
        self.tracker = SuccessTracker(self.config, self.data.xpos[self.object_id].copy())

    def state(self):
        """Read robot-unit observations from actual simulated joint positions."""
        return self.mapping.to_robot(self.data.qpos[self.qidx])

    def parent_transform(self, parent):
        """Return a body-to-world transform for projection diagnostics."""
        if parent == "world":
            return np.eye(4)
        body = self.model.body(parent).id
        transform = np.eye(4)
        transform[:3, :3] = self.data.xmat[body].reshape(3, 3)
        transform[:3, 3] = self.data.xpos[body]
        return transform

    def images(self):
        """Render both calibrated cameras and apply their measured lens distortion."""
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        result = {}
        for name, cfg in self.config["cameras"].items():
            self.renderer.update_scene(self.data, camera=name)
            frame = self.renderer.render().copy()
            distortion = np.asarray(cfg.get("distortion", [0] * 5), dtype=float)
            if np.any(distortion):
                if name not in self.distortion_maps:
                    pixels = np.stack(np.meshgrid(np.arange(640), np.arange(480)), -1).astype(np.float32)
                    k = np.asarray(cfg["K"], dtype=float)
                    ideal = cv2.undistortPoints(pixels.reshape(-1, 1, 2), k, distortion, P=k).reshape(
                        480, 640, 2
                    )
                    self.distortion_maps[name] = (ideal[..., 0], ideal[..., 1])
                frame = cv2.remap(frame, *self.distortion_maps[name], interpolation=cv2.INTER_LINEAR)
            result[name] = frame
        return result

    def step(self, action):
        """Apply one finite position target for exactly ten physical substeps."""
        q = self.mapping.to_sim(action)
        self.check_target(q)
        self.data.ctrl[self.aidx] = q
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Non-finite physics")
        return self.outcome()

    def outcome(self):
        """Evaluate real contacts, motion, support and full containment."""
        fixed = moving = support = False
        for contact in self.data.contact:
            names = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
                for i in (contact.geom1, contact.geom2)
            ]
            if not any(n.startswith("tape_") for n in names):
                continue
            fixed |= any(n.startswith("fixed_jaw") for n in names)
            moving |= any(n.startswith("moving_jaw") for n in names)
            support |= any(n.startswith("holder_") for n in names)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.object_id, velocity, 0)
        position = self.data.xpos[self.object_id].copy()
        success = self.tracker.update(
            position,
            self.data.xmat[self.object_id].reshape(3, 3),
            velocity,
            self.state()[5],
            fixed,
            moving,
            support,
            self.data.xpos[self.model.body("holder").id],
            self.data.xmat[self.model.body("holder").id].reshape(3, 3),
        )
        return {
            "success": bool(success),
            "lifted": bool(self.tracker.lifted),
            "fixed_contact": bool(fixed),
            "moving_contact": bool(moving),
            "holder_contact": bool(support),
            "object_position_m": position.tolist(),
        }

    def ik(self, target, rotation, gripper, initial):
        """Damped Jacobian IK on scratch state: never advances or teleports the task object."""
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        q = self.mapping.to_sim(initial)
        scratch.qpos[self.qidx] = q
        scratch.qpos[self.qidx[5]] = self.mapping.to_sim(np.r_[initial[:5], gripper])[5]
        for _ in range(150):
            mujoco.mj_forward(self.model, scratch)
            current = scratch.site_xmat[self.site].reshape(3, 3)
            orientation_error = sum(np.cross(current[:, i], rotation[:, i]) for i in range(3)) / 2
            error = np.r_[target - scratch.site_xpos[self.site], orientation_error * 0.08]
            if np.linalg.norm(error[:3]) < 0.002 and np.linalg.norm(orientation_error) < 0.12:
                return self.mapping.to_robot(scratch.qpos[self.qidx])
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, scratch, jp, jr, self.site)
            jac = np.vstack([jp[:, self.didx[:5]], 0.08 * jr[:, self.didx[:5]]])
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-4, error)
            scratch.qpos[self.qidx[:5]] = np.clip(
                scratch.qpos[self.qidx[:5]] + np.clip(delta, -0.12, 0.12),
                self.bounds[:5, 0],
                self.bounds[:5, 1],
            )
        raise RuntimeError("IK could not find a feasible target pose")

    def close(self):
        """Release offscreen rendering resources."""
        if self.renderer is not None:
            self.renderer.close()
