from isaaclab.utils import configclass
##
# Pre-defined configs
##
# isort: skip
from parkour_isaaclab.envs import ParkourManagerBasedRLEnvCfg
from .rough_mdp_cfg import * 
from parkour_tasks.default_cfg import  CAMERA_USD_CFG, CAMERA_CFG, VIEWER
from .parkour_teacher_cfg import ParkourTeacherSceneCfg
@configclass
class ParkourStudentSceneCfg(ParkourTeacherSceneCfg):
    depth_camera = CAMERA_CFG
    depth_camera_usd = None
    
    def __post_init__(self):
        super().__post_init__()
        # Rough terrain configuration (inherited from Teacher)
        if self.terrain.terrain_generator is not None:
            self.terrain.terrain_generator.num_rows = 10
            self.terrain.terrain_generator.num_cols = 20
            self.terrain.terrain_generator.horizontal_scale = 0.1



@configclass
class UnitreeGo2StudentParkourEnvCfg(ParkourManagerBasedRLEnvCfg):
    scene: ParkourStudentSceneCfg = ParkourStudentSceneCfg(num_envs=192, env_spacing=1.)
    # Basic settings
    observations: RoughStudentObservationsCfg = RoughStudentObservationsCfg()
    actions: RoughActionsCfg = RoughActionsCfg()
    commands: RoughCommandsCfg = RoughCommandsCfg()
    # MDP settings
    rewards: RoughRewardsCfg = RoughRewardsCfg()
    terminations: RoughTerminationsCfg = RoughTerminationsCfg()
    events: RoughEventCfg = RoughEventCfg()
    # Keep parkours for compatibility but make it minimal
    parkours: RoughParkourEventsCfg = RoughParkourEventsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**18
        # update sensor update periods
        self.scene.depth_camera.update_period = self.sim.dt * self.decimation
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.scene.terrain.terrain_generator.curriculum = False  # Disable curriculum for rough terrain
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8



@configclass
class UnitreeGo2StudentParkourEnvCfg_EVAL(UnitreeGo2StudentParkourEnvCfg):
    viewer = VIEWER 
    rewards: RoughRewardsCfg = RoughRewardsCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 256
        self.episode_length_s = 20.
        self.commands.base_velocity.debug_vis = True

        self.scene.depth_camera_usd = CAMERA_USD_CFG
        self.scene.terrain.max_init_terrain_level = None

        self.observations.depth_camera.depth_cam.params['debug_vis'] = True

        self.commands.base_velocity.resampling_time_range = (60.,60.)
        self.commands.base_velocity.debug_vis = True

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.random_difficulty = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0,1.0)
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
        # Remove parkour-specific events that don't exist in rough terrain setup
        if hasattr(self.events, 'push_by_setting_velocity'):
            self.events.push_by_setting_velocity.interval_range_s = (6.,6.)
        if hasattr(self.events, 'random_camera_position'):
            self.events.random_camera_position.params['rot_noise_range'] = {'pitch':(0, 1)}

@configclass
class UnitreeGo2StudentParkourEnvCfg_PLAY(UnitreeGo2StudentParkourEnvCfg_EVAL):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.num_envs = 16
        self.episode_length_s = 60.

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.7,1.0)
        # Remove parkour-specific events that don't exist in rough terrain setup
        if hasattr(self.events, 'push_by_setting_velocity'):
            self.events.push_by_setting_velocity = None

