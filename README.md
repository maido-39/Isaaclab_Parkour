# Isaaclab_Parkour

Isaaclab based Parkour locomotion 

Base model: [Extreme-Parkour](https://extreme-parkour.github.io/)

**⚠️ MODIFIED VERSION**: This repository has been adapted for rough terrain locomotion using the Unitree Go1 robot instead of the original Go2. See [Modifications](#-modifications) section for details.

## How to install 

```
cd IsaacLab ## going to IsaacLab
```

```
git clone -b go1-implement https://github.com/maido-39/Isaaclab_Parkour.git ## cloning this repo
```

```
cd Isaaclab_Parkour && pip3 install -e .
```

```
cd parkour_tasks && pip3 install -e .
```

## How to train policies

### 1.1. Training Teacher Policy

```
python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 --seed 1 --headless
```

### 1.2. Training Student Policy

```
python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-v0 --seed 1 --headless
```

## How to play your policy 

### 2.1. Pretrained Teacher Policy 

Download Teacher Policy by this [link](https://drive.google.com/file/d/1JtGzwkBixDHUWD_npz2Codc82tsaec_w/view?usp=sharing)

### 2.2. Playing Teacher Policy 

```
python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 --num_envs 16
```

### 2.3. Evaluation Teacher Policy

```
python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Eval-v0 
```

### 3.1 Pretrained Student Policy 

Download Student Policy by this [link](https://drive.google.com/file/d/1qter_3JZgbBcpUnTmTrexKnle7sUpDVe/view?usp=sharing)

### 3.2. Playing Student Policy 

```
python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 --num_envs 16
```

### 3.3. Evaluation Student Policy

```
python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Eval-v0 
```

## How to deploy in IsaacLab

### 4.1. Deployment Teacher Policy 

```
python scripts/rsl_rl/demo.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 
```

### 4.2. Deployment Student Policy 

```
python scripts/rsl_rl/demo.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 
```

## Testing your modules

```
cd parkour_test/ ## You can test your modules in here
```

## Visualize Control (ParkourViewportCameraController)

```
press 1 or 2: Going to environment

press 8: camera forward    

press 4: camera leftward   

press 6: camera rightward   

press 5: camera backward

press 0: Use free camera (can use mouse)

press 1: Not use free camera (default)
```

## How to Deploy sim2sim or sim2real

it is a future work, i will open this repo as soon as possible

* [x] sim2sim: isaaclab to mujoco

* [ ] sim2real: isaaclab to real world

see this [repo](https://github.com/CAI23sbP/go2_parkour_deploy)

### TODO list

* [x] Opening code for training Teacher model  

* [x] Opening code for training Distillation 

* [x] Opening code for deploying policy in IsaacLab by demo: code refer [site](https://isaac-sim.github.io/IsaacLab/main/source/overview/showroom.html)  

* [x] Opening code for deploying policy by sim2sim (mujoco)

* [ ] Opening code for deploying policy in real world

## 🔧 Key Modifications

### 1. Terrain System
- **Changed from**: Custom parkour terrain with subgoals and yaw targeting
- **Changed to**: IsaacLab's standard rough terrain (`ROUGH_TERRAINS_CFG`)
- **Benefits**: More realistic locomotion, compatible with standard IsaacLab components

### 2. Robot Platform
- **Changed from**: Unitree Go2 with DC motor actuators
- **Changed to**: Unitree Go1 with MLP-based actuator model
- **Key differences**: 
  - USD asset paths (`Robot/base` → `Robot/trunk`)
  - Actuator model (`DCMotorCfg` → `ActuatorNetMLPCfg`)
  - Body names in configurations (`base` → `trunk`)

### 3. MDP Configuration
- **Observations**: Simplified to standard IsaacLab rough terrain observations
- **Rewards**: Adapted from IsaacLab's Go2 training configuration
- **Commands**: Uses `UniformVelocityCommandCfg` for forward locomotion
- **Events**: Simplified parkour events compatible with rough terrain

### 4. Student Learning
- **Depth Camera**: Integrated d435i camera for visual learning
- **Distillation**: Student learns to replicate teacher behavior using depth images
- **GRU Encoder**: Processes depth images and proprioceptive information

## 📁 Project Structure

```
Isaaclab_Parkour/
├── parkour_isaaclab/
│   ├── envs/mdp/
│   │   ├── rough_observations.py      # Simplified observations
│   │   ├── rough_rewards.py           # Rough terrain rewards
│   │   └── rough_parkour_events.py    # Compatible parkour events
│   └── actuators/
│       └── parkour_actuator_cfg.py    # Go1 actuator configuration
├── parkour_tasks/
│   └── extreme_parkour_task/config/go2/
│       ├── parkour_teacher_cfg.py     # Teacher environment config
│       ├── parkour_student_cfg.py     # Student environment config
│       ├── rough_mdp_cfg.py           # Rough terrain MDP config
│       └── agents/
│           ├── rsl_teacher_ppo_cfg.py  # Teacher training config
│           └── rsl_student_ppo_cfg.py # Student training config
└── scripts/rsl_rl/
    └── modules/
        ├── dummy_estimator.py         # Dummy estimator for rough terrain
        └── on_policy_runner_with_extractor.py # Modified runner
```

## 🎯 Training Metrics Explained

### Teacher Training Metrics
- **Computation**: Steps per second (collection + learning time)
- **Mean reward**: Overall episode reward
- **Episode length**: Average episode duration
- **Curriculum/terrain_levels**: Current terrain difficulty level

### Student Training Metrics
- **Mean depth_actor_loss**: Loss for depth encoder learning
- **Mean yaw_loss**: Loss for yaw prediction
- **Mean total_loss**: Combined learning loss

## 🔍 Troubleshooting

### Common Issues
1. **Disk Space**: Ensure sufficient disk space for Isaac Sim
2. **CUDA Memory**: Reduce `--num_envs` if running out of GPU memory
3. **Checkpoint Loading**: Verify checkpoint paths exist in logs directory

### Environment Setup
```bash
# Activate conda environment
conda activate isc-pak

# Verify Isaac Sim installation
python -c "import isaacsim; print('Isaac Sim ready')"
```

## 📚 Documentation

For detailed information about the modifications made, see:
- `MODIFICATION_GUIDE.md` - Comprehensive guide to all changes

## Citation

If you use this code for your research, you **must** cite the following paper:

```
@article{cheng2023parkour,
title={Extreme Parkour with Legged Robots},
author={Cheng, Xuxin and Shi, Kexin and Agarwal, Ananye and Pathak, Deepak},
journal={arXiv preprint arXiv:2309.14341},
year={2023}
}
```

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```

```
Copyright (c) 2025, Sangbaek Park (Original IsaacLab Parkour)
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software …

Copyright (c) 2025, REDACTED (Modified Version - Rough Terrain + Go1)



The use of this software in academic or scientific publications requires
explicit citation of the following repositories:

Original: https://github.com/CAI23sbP/Isaaclab_Parkour
Modified: https://github.com/maido-39/Isaaclab_Parkour
```

## contact us 
Copyright (c) 2025, Sangbaek Park (Original IsaacLab Parkour) : 
```
sbp0783@hanyang.ac.kr
```

Copyright (c) 2025, [Your Name] (Modified Version - Rough Terrain + Go1)
```
REDACTED
```