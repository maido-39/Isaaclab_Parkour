"""MDP configuration for rough terrain locomotion following Isaac Lab's standard approach."""

import math
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# Import Isaac Lab's standard MDP functions
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs.mdp.events import ( 
    randomize_rigid_body_mass,
    apply_external_force_torque,
    reset_joints_by_scale
)

from parkour_isaaclab.envs.mdp.parkour_actions import DelayedJointPositionActionCfg 
from parkour_isaaclab.envs.mdp import events
from parkour_isaaclab.envs.mdp.rough_parkour_events import SimpleParkourEventCfg
from parkour_isaaclab.envs.mdp import observations


@configclass
class RoughCommandsCfg:
    """Command specifications for rough terrain MDP following Isaac Lab standard."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), 
            lin_vel_y=(-1.0, 1.0), 
            ang_vel_z=(-1.0, 1.0), 
            heading=(-math.pi, math.pi)
        ),
    )


@configclass
class RoughObservationsCfg:
    """Observation specifications for rough terrain MDP following Isaac Lab standard."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # Standard Isaac Lab observations for rough terrain locomotion
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class DepthCameraPolicyCfg(ObsGroup):
        """Depth camera observations for student learning."""
        depth_cam = ObsTerm(
            func=observations.image_features,
            params={            
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "resize": (58, 87),
                "buffer_len": 2,
                "debug_vis": True
            },
        )

    @configclass
    class DeltaYawOkPolicyCfg(ObsGroup):
        """Delta yaw ok observations for student learning (dummy for rough terrain)."""
        deta_yaw_ok = ObsTerm(
            func=observations.obervation_delta_yaw_ok,
            params={            
                "parkour_name": 'base_parkour',  # Use same name as original
                'threshold': 0.6
            },
        )
    
    policy: PolicyCfg = PolicyCfg()
    depth_camera: DepthCameraPolicyCfg = DepthCameraPolicyCfg()
    delta_yaw_ok: DeltaYawOkPolicyCfg = DeltaYawOkPolicyCfg()


@configclass
class RoughRewardsCfg:
    """Reward terms for rough terrain MDP following Isaac Lab standard."""

    # -- task rewards (tracking velocity commands)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=1.5,  # Increased weight as in Go2 config
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=0.75,  # Increased weight as in Go2 config
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)  # Go2 specific weight
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)  # Go2 specific weight
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    
    # -- contact rewards
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.01,  # Go2 specific weight
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),  # Go2 specific naming
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    
    # -- optional penalties (disabled by default)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class RoughTerminationsCfg:
    """Termination terms for rough terrain MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # Terminate if robot base contacts ground
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), "threshold": 1.0},  # Go1 uses trunk
    )


@configclass
class RoughEventCfg:
    """Configuration for events in rough terrain following Isaac Lab standard."""
    
    # startup events
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),  # Go1 uses trunk
            "mass_distribution_params": (-1.0, 3.0),  # Go1 specific range  
            "operation": "add",
        },
    )

    randomize_rigid_body_mass = EventTerm(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),  # Go1 uses trunk
            "mass_distribution_params": (-1.0, 3.0),  # Go1 specific range
            "operation": "add",
        },
    )

    # reset events
    base_external_force_torque = EventTerm(
        func=apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),  # Go1 uses trunk
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
                "x": (0.0, 0.0),  # Go2 specific - no initial velocity
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),  # Go2 specific - no joint randomization
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval events
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RoughActionsCfg:
    """Action configuration for rough terrain."""
    
    joint_pos = DelayedJointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.25,  # Go2 specific scale
        use_default_offset=True,
        action_delay_steps=[1, 1],
        delay_update_global_steps=24 * 8000,
        history_length=1,
        use_delay=False,  # Disable delay for rough terrain
        clip={'.*': (-4.8, 4.8)}
    )


@configclass
class RoughCurriculumCfg:
    """Curriculum terms for the MDP."""
    
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class RoughParkourEventsCfg:
    """Simple parkour events configuration for rough terrain."""
    
    base_parkour = SimpleParkourEventCfg(
        asset_name='robot',
    )