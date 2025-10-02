"""Simplified observations for rough terrain locomotion."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat

if TYPE_CHECKING:
    from parkour_isaaclab.envs.parkour_manager_based_rl_env import ParkourManagerBasedRLEnv


class RoughTerrainObservations(ManagerTermBase):
    """Simplified observations for rough terrain locomotion without parkour-specific components."""

    def __init__(self, cfg: ObservationTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        self.contact_sensor: ContactSensor = env.scene.sensors['contact_forces']
        self.ray_sensor: RayCaster = env.scene.sensors['height_scanner']
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.history_length = cfg.params.get('history_length', 10)
        
        # Simplified observation buffer (no parkour-specific components)
        obs_size = 3 + 2 + 2 + 12 + 12 + 12 + 4  # ang_vel + imu + commands + joint_pos + joint_vel + actions + contacts
        self._obs_history_buffer = torch.zeros(self.num_envs, self.history_length, obs_size, device=self.device)
        self.measured_heights = torch.zeros(self.num_envs, 132, device=self.device)
        self.env = env

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._obs_history_buffer[env_ids, :, :] = 0.0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        history_length: int,
        ) -> torch.Tensor:
        
        # Get basic robot state
        roll, pitch, yaw = euler_xyz_from_quat(self.asset.data.root_quat_w)
        imu_obs = torch.stack((wrap_to_pi(roll), wrap_to_pi(pitch)), dim=1).to(self.device)
        
        # Update height measurements periodically
        if env.common_step_counter % 5 == 0:
            self.measured_heights = self._get_heights()
        
        # Get commands (simplified - just forward velocity)
        commands = env.command_manager.get_command('base_velocity')
        
        # Build observation without parkour-specific components
        obs_buf = torch.cat((
            self.asset.data.root_ang_vel_b * 0.25,   # [3] angular velocity
            imu_obs,    # [2] roll, pitch
            commands[:, 0:2],  # [2] linear velocity commands
            self.asset.data.joint_pos - self.asset.data.default_joint_pos,  # [12] joint positions
            self.asset.data.joint_vel * 0.05,  # [12] joint velocities
            env.action_manager.get_term('joint_pos').action_history_buf[:, -1],  # [12] last actions
            self._get_contact_fill(),  # [4] contact forces
        ), dim=-1)
        
        # Add to history buffer
        self._obs_history_buffer[:, 1:] = self._obs_history_buffer[:, :-1].clone()
        self._obs_history_buffer[:, 0] = obs_buf
        
        # Return flattened history
        return self._obs_history_buffer.view(self.num_envs, -1)

    def _get_heights(self) -> torch.Tensor:
        """Get height measurements from ray sensor."""
        if hasattr(self.ray_sensor, 'data') and hasattr(self.ray_sensor.data, 'ray_hits_w'):
            ray_data = self.ray_sensor.data.ray_hits_w[..., 2] - self.ray_sensor.data.pos_w[..., 2]
            return ray_data.view(self.num_envs, -1)
        else:
            return torch.zeros(self.num_envs, 132, device=self.device)

    def _get_contact_fill(self) -> torch.Tensor:
        """Get contact forces for feet."""
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, :, 2]  # Z forces
        # Get forces for 4 feet (assuming last 4 bodies are feet)
        feet_forces = contact_forces[:, -4:]  
        return (feet_forces > 1.0).float()  # Binary contact detection
