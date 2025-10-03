from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # isort: skip
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporter
from parkour_tasks.extreme_parkour_task.config.go2 import agents 
from isaaclab.sensors import RayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.envs import ViewerCfg
import os, torch 
from parkour_isaaclab.actuators.parkour_actuator_cfg import ParkourDCMotorCfg, ParkourActuatorNetMLPCfg
from isaaclab_assets.robots.unitree import GO1_ACTUATOR_CFG

def quat_from_euler_xyz_tuple(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> tuple:
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    convert = torch.stack([qw, qx, qy, qz], dim=-1) * torch.tensor([1.,1.,1.,-1])
    return tuple(convert.numpy().tolist())

@configclass
class ParkourDefaultSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    terrain = TerrainImporterCfg(
        class_type= TerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=None,
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAAC_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
            texture_scale=(1.0, 1.0),
        ),
        debug_vis=False,
    )
    def __post_init__(self):
        self.robot.spawn.articulation_props.enabled_self_collisions = True
        # Use Go1's MLP-based actuator model (same as Isaac Lab's standard Go1)
        self.robot.actuators['base_legs'] = GO1_ACTUATOR_CFG

## we are now using a raycaster based camera, not a pinhole camera. see tail issue https://github.com/isaac-sim/IsaacLab/issues/719
CAMERA_CFG = RayCasterCameraCfg( 
    prim_path= '{ENV_REGEX_NS}/Robot/trunk',
    data_types=["distance_to_camera"],
    offset=RayCasterCameraCfg.OffsetCfg(
        ## Transform camera from robot frame for robot observation
        pos=(0.33, 0.0, 0.08), 
        rot=quat_from_euler_xyz_tuple(*tuple(torch.deg2rad(torch.tensor([180,70,-90])))), 
        convention="ros"
        ),
    depth_clipping_behavior = 'max',
    pattern_cfg = PinholeCameraPatternCfg(
        focal_length=11.041, 
        horizontal_aperture=20.955,
        vertical_aperture = 12.240,
        height=60,
        width=106,
    ),
    mesh_prim_paths=["/World/ground"],
    max_distance = 2.,
)

CAMERA_USD_CFG = AssetBaseCfg(
    ## Physical robot cam defenition for camera observation
    prim_path="{ENV_REGEX_NS}/Robot/trunk/d435",
    spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(agents.__path__[0],'d435.usd')),
    init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.33, 0.0, 0.08), 
            rot=quat_from_euler_xyz_tuple(*tuple(torch.deg2rad(torch.tensor([180,90,-90]))))
    )
)
VIEWER = ViewerCfg(
    ## User's viewport at IsaacSim
    eye=(-0., 2.6, 1.6),
    asset_name = "robot",
    origin_type = 'asset_root',
)
