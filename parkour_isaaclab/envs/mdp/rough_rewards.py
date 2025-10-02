"""Simplified rewards for rough terrain locomotion."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from parkour_isaaclab.envs.parkour_manager_based_rl_env import ParkourManagerBasedRLEnv


def reward_forward_velocity(
    env: ParkourManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving forward."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, 0]  # Forward velocity in body frame


def reward_upright_orientation(
    env: ParkourManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for keeping upright orientation."""
    asset: Articulation = env.scene[asset_cfg.name]
    # Penalize roll and pitch deviations
    roll_pitch_penalty = torch.sum(torch.square(asset.data.root_quat_w[:, 1:3]), dim=1)
    return torch.exp(-roll_pitch_penalty * 3.0)


def reward_energy_efficiency(
    env: ParkourManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for energy efficiency (low torques)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return -torch.sum(torch.square(asset.data.applied_torque), dim=1) * 1e-5


def reward_smooth_actions(
    env: ParkourManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for smooth action changes."""
    action_history = env.action_manager.get_term('joint_pos').action_history_buf
    if action_history.shape[1] >= 2:
        action_diff = action_history[:, -1] - action_history[:, -2]
        return -torch.sum(torch.square(action_diff), dim=1) * 0.1
    else:
        return torch.zeros(env.num_envs, device=env.device)


def reward_feet_contact_stability(
    env: ParkourManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
) -> torch.Tensor:
    """Reward for stable foot contacts."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]  # Z forces
    
    # Get forces for feet (assuming last 4 bodies are feet)
    feet_forces = contact_forces[:, -4:]  
    
    # Reward having at least 2 feet in contact
    in_contact = (feet_forces > 1.0).float()
    num_contacts = torch.sum(in_contact, dim=1)
    return torch.where(num_contacts >= 2, torch.ones_like(num_contacts), torch.zeros_like(num_contacts))


def reward_collision_penalty(
    env: ParkourManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["base",".*_calf",".*_thigh"]),
) -> torch.Tensor:
    """Penalty for unwanted body collisions."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, 0]  # Current forces
    
    # Check for contacts on non-foot bodies
    unwanted_contacts = torch.sum(torch.norm(contact_forces, dim=-1), dim=1)
    return -torch.where(unwanted_contacts > 1.0, torch.ones_like(unwanted_contacts), torch.zeros_like(unwanted_contacts))


def reward_base_height(
    env: ParkourManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.3,
) -> torch.Tensor:
    """Reward for maintaining desired base height."""
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.abs(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-height_error * 5.0)
