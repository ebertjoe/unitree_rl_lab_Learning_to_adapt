import math
import torch

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from isaaclab.managers import CommandTermCfg

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
    },
)


# --- 配置表 ---
GAIT_CONFIGS = {
    "6": {
        "name": "stand", "period": 1.0, "threshold": 1.0, "offset": [0.0, 0.0, 0.0, 0.0],
        "k": 0.01, "z_nom": -0.32, "x_lim": 0.10, "y_lim": 0.10},
    "1": {
        "name": "trot", "period": 0.4, "threshold": 0.5, "offset": [0.0, 0.5, 0.5, 0.0],
        "k": 0.03, "z_nom": -0.32, "x_lim": 0.10, "y_lim": 0.10},
    "7": {
        "name": "run", "period": 0.3, "threshold": 0.4, "offset": [0.0, 0.5, 0.5, 0.0],
        "k": 0.03, "z_nom": -0.32, "x_lim": 0.12, "y_lim": 0.10},
    "0": {
        "name": "bound", "period": 0.4, "threshold": 0.4, "offset": [0.5, 0.5, 0.0, 0.0],
        "k": 0.03, "z_nom": -0.32, "x_lim": 0.12, "y_lim": 0.10},
    "4": {
        "name": "pronk", "period": 0.5, "threshold": 0.5, "offset": [0.0, 0.0, 0.0, 0.0],
        "k": 0.01, "z_nom": -0.32, "x_lim": 0.08, "y_lim": 0.10},
    "5": {
        "name": "limp", "period": 0.4, "threshold": 0.5, "offset": [0.5, 0.5, 0.5, 0.0],
        "k": 0.03, "z_nom": -0.32, "x_lim": 0.12, "y_lim": 0.10},
    "3": {
        "name": "amble", "period": 0.5, "threshold": 0.625, "offset": [0.0, 0.5, 0.25, 0.75],
        "k": 0.02, "z_nom": -0.32, "x_lim": 0.14, "y_lim": 0.12},
    "2": {
        "name": "hop", "period": 0.3, "threshold": 0.5, "offset": [0.0, 0.0, 0.0, 0.0],
        "k": 0.03, "z_nom": -0.30, "x_lim": 0.15, "y_lim": 0.10},
}


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (0.0, 0.0)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.0, 4.0),
        rel_standing_envs=0.02,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.2),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(-0.5, 0.5),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.2),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.5, 0.5),
        ),
    )

    gait_id = mdp.UniformIntegerCommandCfg(
        num_commands=1,
        resampling_time_range=(8.0, 10.0),
        params={
            "range": (0, 7),  # 0, 1, 2, 3, 4, 5, 6, 7 There are 8 types of gait
            "asset_cfg": SceneEntityCfg("robot"), 
        },
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        # Explicitly list these 12 names to ensure the neural network output's first dimension always corresponds to FR_hip_joint
        joint_names=[
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        scale=0.25,
        use_default_offset=True,
    )


def _quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.norm(q, dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = torch.zeros((N, 3, 3), device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    return R


# --- Definition of observation configuration class ---
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 2. Modify state_s
        state_s = ObsTerm(
            func=mdp.robot_state_s,
            params={
                "gait_command_name": "gait_id",
                "gait_table": GAIT_CONFIGS,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"]),
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                ])
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100, 100)
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Instantiate the group
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.gait_conditioned_base_velocity, clip=(-100, 100), params={
                "command_name": "base_velocity",
                "gait_command_name": "gait_id",
                "stand_gait_id": 6,
            }
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        base_height = ObsTerm(func=mdp.base_pos_z, clip=(-100, 100))

    # privileged observations
    critic: CriticCfg = CriticCfg()
    # policy: PolicyCfg = PolicyCfg()


def alive_bonus_func(env):
    """Points are awarded as long as the person is alive"""
    return torch.ones(env.num_envs, device=env.device)


@configclass
class RewardsCfg:
    # Eq.(6)
    efficiency_penalty = RewTerm(
        func=mdp.r_eta,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "use_dt_scaling": False,
            "clamp_jerk": None,
        },
    )

    # Eq.(7)
    vcmd_tracking = RewTerm(
        func=mdp.r_vcmd,
        weight=10.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "wz_scale": 1.0,
        },
    )

    # Eq.(8) - r_f will read env.beta_* references
    gait_tracking = RewTerm(
        func=mdp.r_f,
        weight=-12.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
            )
        },
    )

    # Eq.(9) - r_stab will read env.beta_* references
    stability = RewTerm(
        func=mdp.r_stab,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
            ),
            "hip_joint_ids": [1, 0, 3, 2],  # Hip joint index in Go2
            "gait_table": GAIT_CONFIGS, 
            "gait_command_name": "gait_id",
            "desired_gravity_b": [0.0, 0.0, -1.0],
        },
    )

    # body_height = RewTerm(
    #     func=mdp.base_height_l2,
    #     weight=-1.5,
    #     params={
    #         "target_height": 0.32,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    # foot_trajectory = RewTerm(
    #     func=mdp.foot_trajectory_tracking,
    #     weight=-6.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
    #         ),
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
    #         ),
    #     },
    # )

    # gait_symmetry = RewTerm(
    #     func=mdp.gait_conditioned_symmetry,
    #     weight=-8.0,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    
    pass


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        
        # Do not use limit_ranges to override ranges
        # self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

        self.commands.base_velocity.resampling_time_range = (9999.0, 9999.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges = self.commands.base_velocity.ranges.replace(
            lin_vel_x=(1.2, 1.2), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        )