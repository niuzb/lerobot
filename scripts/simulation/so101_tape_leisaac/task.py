"""LeIsaac tape task using the same single-arm foundation as PickOrange."""

from pathlib import Path

import isaaclab.sim as sim_utils
import numpy as np
import torch
from common import JOINTS, TASK, JointMap
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import mdp
from isaaclab.managers import (
    DatasetExportMode,
    ObservationGroupCfg,
    ObservationTermCfg,
    RecorderManagerBaseCfg,
    TerminationTermCfg,
)
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.utils import configclass
from leisaac.tasks.template.single_arm_env_cfg import (
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from pxr import Usd, UsdPhysics
from scipy.spatial.transform import Rotation
from spatial import base_position, table_height


def make_config(config, assets, cameras=False):
    """Create one CPU-PhysX scene with A30 rendering and six absolute targets."""
    assets = Path(assets)
    robot_stage = Usd.Stage.Open(str(assets / "robot.usd"))
    robot_stage.Load()
    prefix = str(robot_stage.GetDefaultPrim().GetPath())
    links = {
        prim.GetName(): str(prim.GetPath())[len(prefix) :]
        for prim in robot_stage.TraverseAll()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    for name in ("base", "gripper", "moving_jaw_so101_v1"):
        if name not in links:
            raise ValueError(f"Imported robot lacks expected physical body: {name}: {links}")
    robot_prim = "{ENV_REGEX_NS}/Robot"

    def camera(name, parent):
        values = config["cameras"][name]
        pose = np.asarray(values["T_parent_cv"])
        xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
        return TiledCameraCfg(
            prim_path=parent + f"/{name}_camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=tuple(map(float, pose[:3, 3])),
                rot=tuple(map(float, xyzw[[3, 0, 1, 2]])),
                convention="ros",
            ),
            data_types=["rgb"],
            width=640,
            height=480,
            update_period=1 / 30,
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                np.asarray(values["K"]).ravel().tolist(),
                640,
                480,
                clipping_range=(0.002, 3.0),
                focal_length=24,
            ),
        )

    initial = np.asarray(config["initial_robot_state"]).copy()
    initial[5] = config["generation"]["open_percent"]
    q = JointMap(config["joint_map"]).to_sim(initial)

    @configclass
    class SceneCfg(SingleArmTaskSceneCfg):
        scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.CuboidCfg(
                size=(
                    2 * config["table"]["half_x_m"],
                    2 * config["table"]["half_y_m"],
                    config["table"]["thickness_m"],
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0001, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0, 0, table_height(config) - config["table"]["thickness_m"] / 2)
            ),
        )
        robot = ArticulationCfg(
            prim_path=robot_prim,
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(assets / "robot.usd"),
                # The assets already author contact APIs with zero threshold.
                # Lab 2.3's bool activation wrapper would overwrite it with 1 N.
                activate_contact_sensors=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, max_depenetration_velocity=0.1
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=16,
                    fix_root_link=True,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(base_position(config)),
                rot=(1, 0, 0, 0),
                joint_pos=dict(zip(JOINTS, map(float, q), strict=True)),
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=JOINTS[:5],
                    stiffness=998.22,
                    damping=2.731,
                    effort_limit_sim=2.94,
                    velocity_limit_sim=10,
                ),
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=["gripper"],
                    stiffness=998.22,
                    damping=2.731,
                    effort_limit_sim=0.12,
                    velocity_limit_sim=10,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )
        tape = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Tape",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(assets / "tape.usd"),
                activate_contact_sensors=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    sleep_threshold=0.0,
                    stabilization_threshold=0.0,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=16,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(config["tape"]["position_m"])),
        )
        holder = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Holder",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(assets / "holder.usd"),
                activate_contact_sensors=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    sleep_threshold=0.0,
                    stabilization_threshold=0.0,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=16,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(config["holder"]["position_m"])),
        )
        ee_frame = FrameTransformerCfg(
            prim_path=robot_prim + links["base"],
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path=robot_prim + links["gripper"], name="gripper")
            ],
        )
        wrist = None
        light = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
        )
        front = camera("front", robot_prim + links["gripper"]) if cameras else None
        side = camera("side", "{ENV_REGEX_NS}") if cameras else None
        fixed_contact = ContactSensorCfg(
            prim_path=robot_prim + links["gripper"],
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Tape"],
            update_period=0,
            max_contact_data_count_per_prim=128,
        )
        moving_contact = ContactSensorCfg(
            prim_path=robot_prim + links["moving_jaw_so101_v1"],
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Tape"],
            update_period=0,
            max_contact_data_count_per_prim=128,
        )
        support_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Tape",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Holder"],
            update_period=0,
            max_contact_data_count_per_prim=256,
        )

    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg(ObservationGroupCfg):
            joint_pos = ObservationTermCfg(func=mdp.joint_pos)
            joint_vel = ObservationTermCfg(func=mdp.joint_vel)

            def __post_init__(self):
                self.concatenate_terms = False
                self.enable_corruption = False

        policy = PolicyCfg()

    @configclass
    class TerminationsCfg(SingleArmTerminationsCfg):
        # The collector sets this only after the independent physical gate passes.
        success = TerminationTermCfg(
            func=lambda env: torch.full(
                (env.num_envs,),
                bool(getattr(env, "tape_success", False)),
                dtype=torch.bool,
                device=env.device,
            )
        )

    @configclass
    class TapeEnvCfg(SingleArmTaskEnvCfg):
        scene = SceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
        observations = ObservationsCfg()
        terminations = TerminationsCfg()
        task_description = TASK

        def __post_init__(self):
            super().__post_init__()
            self.use_teleop_device("so101leader")
            self.actions.arm_action.use_default_offset = False
            self.actions.gripper_action.use_default_offset = False
            self.actions.arm_action.preserve_order = True
            self.actions.gripper_action.preserve_order = True
            self.sim.device = "cpu"
            self.seed = 60000
            self.sim.dt = 1 / 300
            self.decimation = 10
            self.sim.render_interval = 10
            self.sim.physx.enable_ccd = True
            self.sim.physx.enable_stabilization = False
            self.sim.physx.friction_correlation_distance = 0.0005
            self.sim.physx.friction_offset_threshold = 0.001
            self.sim.render.antialiasing_mode = "FXAA"
            self.sim.render.enable_dl_denoiser = False
            self.sim.render.enable_dlssg = False
            self.sim.render.enable_translucency = False
            # Standalone headless resets can spin in Lab's texture-loading
            # counter even with these local procedural assets. The renderer
            # explicitly warms both sensors and checks every captured image.
            self.wait_for_textures = False
            self.recorders = RecorderManagerBaseCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE)
            self.episode_length_s = 120
            self.dynamic_reset_gripper_effort_limit = False

    return TapeEnvCfg(), links
