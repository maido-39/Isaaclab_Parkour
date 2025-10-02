from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (
ParkourRslRlOnPolicyRunnerCfg,
ParkourRslRlPpoActorCriticCfg,
ParkourRslRlActorCfg,
ParkourRslRlStateHistEncoderCfg,
ParkourRslRlEstimatorCfg,
ParkourRslRlPpoAlgorithmCfg,
ParkourRslRlBaseCfg
)
from isaaclab.utils import configclass

@configclass
class RoughTerrainRslRlBaseCfg(ParkourRslRlBaseCfg):
    """Base configuration for rough terrain with correct observation dimensions."""
    num_priv_explicit: int = 0  # No privileged states in rough terrain
    num_priv_latent: int = 0    # No privileged latent states in rough terrain
    num_prop: int = 3 + 3 + 3 + 3 + 12 + 12 + 12  # 48: base_lin_vel + base_ang_vel + projected_gravity + velocity_commands + joint_pos + joint_vel + actions
    num_scan: int = 132         # height_scan (keep original for student compatibility)
    num_hist: int = 10

@configclass 
class RoughTerrainRslRlStateHistEncoderCfg(RoughTerrainRslRlBaseCfg):
    class_name: str = "StateHistoryEncoder" 
    channel_size: int = 10 

@configclass
class RoughTerrainRslRlEstimatorCfg(RoughTerrainRslRlBaseCfg):
    class_name: str = "DummyEstimator" 
    train_with_estimated_states: bool = True 
    learning_rate: float = 1.e-4 
    hidden_dims: list[int] = [128, 64]

@configclass
class RoughTerrainRslRlActorCfg(RoughTerrainRslRlBaseCfg):
    class_name: str = "Actor"
    state_history_encoder: RoughTerrainRslRlStateHistEncoderCfg = RoughTerrainRslRlStateHistEncoderCfg()

@configclass
class UnitreeGo2ParkourTeacherPPORunnerCfg(ParkourRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "unitree_go2_parkour"
    empirical_normalization = False
    policy = ParkourRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        scan_encoder_dims = [128, 64, 32],  # Keep original for student compatibility
        priv_encoder_dims = [64, 20],
        activation="elu",
        actor = RoughTerrainRslRlActorCfg(
            class_name = "Actor",
            state_history_encoder = RoughTerrainRslRlStateHistEncoderCfg(
                class_name = "StateHistoryEncoder" 
            )
        )
    )
    estimator = RoughTerrainRslRlEstimatorCfg(
        hidden_dims = [128, 64],
        class_name = "DummyEstimator"
    )
    depth_encoder = None
    algorithm = ParkourRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        desired_kl=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate = 2.e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        dagger_update_freq = 20,
        priv_reg_coef_schedual = [0.0, 0.1, 2000.0, 3000.0],
    )

