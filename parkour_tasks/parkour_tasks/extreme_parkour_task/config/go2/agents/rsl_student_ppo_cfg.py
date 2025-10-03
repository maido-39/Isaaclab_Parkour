from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (
ParkourRslRlOnPolicyRunnerCfg,
ParkourRslRlPpoActorCriticCfg,
ParkourRslRlDistillationAlgorithmCfg,
ParkourRslRlDepthEncoderCfg
)
from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_teacher_ppo_cfg import (
RoughTerrainRslRlActorCfg,
RoughTerrainRslRlStateHistEncoderCfg,
RoughTerrainRslRlEstimatorCfg
)
from isaaclab.utils import configclass

@configclass
class UnitreeGo2ParkourStudentPPORunnerCfg(ParkourRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24 
    max_iterations = 50000 
    save_interval = 100
    experiment_name = "unitree_go2_parkour"
    empirical_normalization = False
    # Enable resume to load teacher checkpoint for distillation
    resume: bool = True
    load_run: str = "*"  # Find any run directory
    load_checkpoint: str = "model_*.pt"  # Find any model checkpoint
    policy = ParkourRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        scan_encoder_dims = [128, 64, 32],
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
            hidden_dims = [128, 64]
    )
    depth_encoder = ParkourRslRlDepthEncoderCfg(
        hidden_dims = 512,
        learning_rate= 1e-3,
        num_steps_per_env = 24*5
    )

    algorithm = ParkourRslRlDistillationAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate = 2.e-4, 
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

