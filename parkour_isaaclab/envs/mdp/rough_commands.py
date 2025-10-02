"""Simplified commands for rough terrain locomotion."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTermCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommand

if TYPE_CHECKING:
    from parkour_isaaclab.envs.parkour_manager_based_rl_env import ParkourManagerBasedRLEnv


class RoughTerrainVelocityCommand(UniformVelocityCommand):
    """Simple velocity command for rough terrain locomotion."""

    def __init__(self, cfg: CommandTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _update_command(self):
        """Update the velocity command (simplified version)."""
        # Simple forward velocity command
        self.command[:, 0] = torch.clamp(
            self.command[:, 0], 
            self.cfg.ranges.lin_vel_x[0], 
            self.cfg.ranges.lin_vel_x[1]
        )
        # Keep angular velocity at zero for simplicity
        self.command[:, 2] = 0.0

    def _resample_command(self, env_ids):
        """Resample the velocity command for given environments."""
        # Sample forward velocity
        self.command[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * \
                                  (self.cfg.ranges.lin_vel_x[1] - self.cfg.ranges.lin_vel_x[0]) + \
                                  self.cfg.ranges.lin_vel_x[0]
        
        # Set lateral and angular velocities to zero for simplicity
        self.command[env_ids, 1] = 0.0
        self.command[env_ids, 2] = 0.0
