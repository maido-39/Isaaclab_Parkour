
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from parkour_isaaclab.envs import ParkourManagerBasedRLEnvCfg
from .rough_mdp_cfg import * 
from parkour_tasks.default_cfg import ParkourDefaultSceneCfg, VIEWER

@configclass
## Default Scene Config
class ParkourTeacherSceneCfg(ParkourDefaultSceneCfg):
    ## To Observation!! how??
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.375, 0.0, 20.0)),  # Forward offset for camera-like perspective
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=[1.65, 1.5]),  # Keep original: 12x11=132 points for student compatibility
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", 
                                      history_length=2, 
                                      track_air_time=True, 
                                      debug_vis= False,
                                      force_threshold=1.
                                      )
    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator = ROUGH_TERRAINS_CFG
    
@configclass
## 
## Espacially for Teacher Environment Config
class UnitreeGo2TeacherParkourEnvCfg(ParkourManagerBasedRLEnvCfg):
    scene: ParkourTeacherSceneCfg = ParkourTeacherSceneCfg(num_envs=64, env_spacing=1.)
    # Basic settings
    observations: RoughObservationsCfg = RoughObservationsCfg()
    actions: RoughActionsCfg = RoughActionsCfg()
    commands: RoughCommandsCfg = RoughCommandsCfg()
    # MDP settings
    rewards: RoughRewardsCfg = RoughRewardsCfg()
    terminations: RoughTerminationsCfg = RoughTerminationsCfg()
    events: RoughEventCfg = RoughEventCfg()
    curriculum: RoughCurriculumCfg = RoughCurriculumCfg()
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
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.scene.terrain.terrain_generator.curriculum = False
        self.actions.joint_pos.use_delay = False
        self.actions.joint_pos.history_length = 1

@configclass
class UnitreeGo2TeacherParkourEnvCfg_EVAL(UnitreeGo2TeacherParkourEnvCfg):
    viewer = VIEWER 

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 256
        self.episode_length_s = 20.
        self.commands.base_velocity.debug_vis = True
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.random_difficulty = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0,1.0)
        self.events.randomize_rigid_body_mass = None
        self.commands.base_velocity.resampling_time_range = (60.,60.)
                
@configclass
class UnitreeGo2TeacherParkourEnvCfg_PLAY(UnitreeGo2TeacherParkourEnvCfg_EVAL):
    viewer = VIEWER 

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 60.
        self.scene.num_envs = 16
        self.commands.base_velocity.debug_vis = True
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.7,1.0)


