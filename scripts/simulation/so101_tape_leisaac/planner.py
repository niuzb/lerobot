"""Ring-wall state machine with the same execution contract as PickOrange."""

import numpy as np
import torch
from common import JOINTS, JointMap, Kinematics
from leisaac.datagen.state_machine.base import StateMachineBase
from scipy.optimize import least_squares
from spatial import base_position, table_height


class TapeStateMachine(StateMachineBase):
    """Plan actuator targets; do not write object or robot poses during a demo."""

    def __init__(self, config):
        self.config = config
        self.mapping = JointMap(config["joint_map"])
        self.fk = Kinematics(config["asset"])
        self.base_position = base_position(config)
        self.initial = np.asarray(config["initial_robot_state"]).copy()
        self.initial[5] = config["generation"]["open_percent"]
        self.stages = []
        self.reset()

    def setup(self, env):
        self.robot = env.scene["robot"]
        if not self.robot.is_fixed_base:
            raise RuntimeError("The robot must be fixed to the table")
        self.joint_ids, names = self.robot.find_joints(JOINTS, preserve_order=True)
        if names != JOINTS:
            raise RuntimeError(f"Unexpected articulation joint order: {names}")
        self.gripper_id = self.robot.find_bodies("gripper")[0][0]
        q = self.robot.data.joint_pos[0, self.joint_ids].cpu().numpy()
        predicted = self.fk.frames(q)["gripper"]
        actual = self.robot.data.body_link_pos_w[0, self.gripper_id].cpu().numpy()
        error = float(np.linalg.norm(predicted[:3, 3] + self.base_position - actual))
        if error > 0.001:
            raise RuntimeError(
                f"Imported articulation FK differs by {error:.6f} m: "
                f"expected={predicted[:3, 3].tolist()}, actual={actual.tolist()}, q={q.tolist()}"
            )
        self.fk_error_m = error

    def begin(self, env):
        self.reset()
        tape = env.scene["tape"].data.root_pos_w[0].cpu().numpy().copy()
        holder = env.scene["holder"].data.root_pos_w[0].cpu().numpy().copy()
        tape -= self.base_position
        holder -= self.base_position
        table_local_z = table_height(self.config) - self.base_position[2]
        values = self.config["generation"]
        direction = tape[:2] / np.linalg.norm(tape[:2])
        pinch = np.r_[
            tape[:2] + direction * values["inner_tip_radius_m"], table_local_z + values["tip_height_m"]
        ]
        approach = pinch.copy()
        approach[2] = table_local_z + 0.080
        lift = pinch.copy()
        lift[2] = table_local_z + 0.145
        transport = np.r_[holder[:2] + [values["inner_tip_radius_m"], 0], table_local_z + 0.145]
        opened, closed = values["open_percent"], values["closed_percent"]
        # PickOrange's approach/grasp/lift/transport/release structure is retained;
        # waypoints and grip closure come from the existing ring-wall task.
        self.stages = [
            ("approach", approach, opened, 60),
            ("descend", pinch, opened, 90),
            ("close", pinch, closed, 60),
            ("lift", lift, closed, 120),
            ("transport", transport, closed, 150),
            ("open", transport, opened, 60),
            ("settle", transport, opened, 150),
        ]

    @property
    def stage(self):
        return self.stages[min(self.stage_index, len(self.stages) - 1)][0]

    def ik(self, target, start, grip):
        q = start.copy()
        q[5] = self.mapping.to_sim(np.r_[self.initial[:5], grip])[5]

        def residual(arm):
            q[:5] = arm
            tip, orientation = self.fk.fingertip(q)
            return np.r_[
                (tip - target) * 100,
                (orientation[:, 2] - [0, 0, 1]) * 2,
                arm[4] - np.deg2rad(self.initial[4]),
            ]

        fit = least_squares(
            residual,
            np.clip(start[:5], self.fk.bounds[:5, 0] + 1e-7, self.fk.bounds[:5, 1] - 1e-7),
            bounds=(self.fk.bounds[:5, 0], self.fk.bounds[:5, 1]),
            max_nfev=120,
        )
        residual(fit.x)
        if np.linalg.norm(self.fk.fingertip(q)[0] - target) > 0.002:
            raise RuntimeError("IK position error exceeds 2 mm")
        return q.copy()

    def get_action(self, env):
        _, target, grip, steps = self.stages[self.stage_index]
        if self.tick == 0:
            self.start = np.clip(
                self.robot.data.joint_pos[0, self.joint_ids].cpu().numpy(),
                self.fk.bounds[:, 0],
                self.fk.bounds[:, 1],
            )
            self.goal = self.ik(target, self.start, grip)
        u = (self.tick + 1) / steps
        command = self.start + u * u * (3 - 2 * u) * (self.goal - self.start)
        return torch.as_tensor(command, dtype=torch.float32, device=env.device).unsqueeze(0)

    def advance(self):
        self.tick += 1
        if self.tick == self.stages[self.stage_index][3]:
            self.tick = 0
            self.stage_index += 1

    def reset(self):
        self.stage_index = 0
        self.tick = 0
        self.start = self.goal = None
        self.accepted = False

    @property
    def is_episode_done(self):
        return bool(self.stages and self.stage_index >= len(self.stages))

    def check_success(self, env):
        return self.accepted
