"""Simplified parkour events for rough terrain that don't depend on terrain-specific features."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.assets import Articulation
from parkour_isaaclab.managers.parkour_manager import ParkourTerm
from parkour_isaaclab.managers.parkour_manager_term_cfg import ParkourTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from parkour_isaaclab.envs.parkour_manager_based_rl_env import ParkourManagerBasedRLEnv


class SimpleParkourEvent(ParkourTerm):
    """Simplified parkour event manager that doesn't depend on terrain-specific features."""
    
    cfg: SimpleParkourEventCfg

    def __init__(self, cfg: SimpleParkourEventCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        
        # Initialize minimal state variables
        self.target_yaw = torch.zeros(self.num_envs, device=self.device)
        self.next_target_yaw = torch.zeros(self.num_envs, device=self.device)
        self.target_pos_rel = torch.zeros(self.num_envs, 2, device=self.device)
        self.next_target_pos_rel = torch.zeros(self.num_envs, 2, device=self.device)
        self.cur_goals = torch.zeros(self.num_envs, 2, device=self.device)
        self.next_goals = torch.zeros(self.num_envs, 2, device=self.device)
        self.cur_goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reach_goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.reached_goal_ids = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.dis_to_start_pos = torch.zeros(self.num_envs, device=self.device)
        self.env_per_terrain_name = ['rough_terrain'] * self.num_envs
        
        # Simple goal tracking
        self.reach_goal_delay = 0.1
        self.next_goal_threshold = 0.2
        self.simulation_time = env.step_dt

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the parkour event state."""
        if env_ids is None:
            env_ids = slice(None)
        
        # Reset state variables
        self.target_yaw[env_ids] = 0.0
        self.next_target_yaw[env_ids] = 0.0
        self.target_pos_rel[env_ids] = 0.0
        self.next_target_pos_rel[env_ids] = 0.0
        self.cur_goals[env_ids] = 0.0
        self.next_goals[env_ids] = 0.0
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0.0
        self.reached_goal_ids[env_ids] = False
        self.dis_to_start_pos[env_ids] = 0.0
        
        return {}  # Return empty dictionary for metrics

    def update(self, dt: float) -> None:
        """Update the parkour event state."""
        # Simple forward direction targeting
        self.target_yaw[:] = 0.0  # Always face forward
        self.next_target_yaw[:] = 0.0
        
        # Set simple forward goals
        robot_pos = self.robot.data.root_pos_w[:, :2]
        self.cur_goals = robot_pos + torch.tensor([1.0, 0.0], device=self.device)
        self.next_goals = robot_pos + torch.tensor([2.0, 0.0], device=self.device)
        
        self.target_pos_rel = self.cur_goals - robot_pos
        self.next_target_pos_rel = self.next_goals - robot_pos
    
    @property
    def command(self) -> torch.Tensor:
        """The generated command tensor."""
        return torch.zeros(self.num_envs, 3, device=self.device)  # Dummy command
    
    def _update_command(self):
        """Update the command (called at each step)."""
        self.update(self.simulation_time)
    
    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for given environments."""
        self.reset(env_ids)
    
    def _update_metrics(self):
        """Update metrics (optional)."""
        pass


@configclass
class SimpleParkourEventCfg(ParkourTermCfg):
    """Configuration for simplified parkour events."""
    class_type: type = SimpleParkourEvent
    asset_name: str = "robot"
