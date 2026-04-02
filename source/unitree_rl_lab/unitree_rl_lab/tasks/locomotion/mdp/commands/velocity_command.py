from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """Supports configuration of horizontal speed commands with limited range."""
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


class UniformIntegerCommand(CommandTerm):
    """Integer command implementation class, responsible for storing gait IDs and random resampling."""

    def __init__(self, cfg: UniformIntegerCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Pre-allocate tensor for storing commands (num_envs, 1)
        self.value_command = torch.zeros(self.num_envs, 1, device=self.device)
        # Extract sampling range from configuration
        self.low, self.high = cfg.params["range"]

    # Add the env_ids parameter
    def _resample_command(self, env_ids: torch.Tensor):
        """Implement the abstract method required by the base class: sampling logic"""
        # Randomization is performed only on environment IDs that require resampling.
        rands = torch.randint(
            low=self.low,
            high=self.high + 1,
            size=(len(env_ids), 1),
            device=self.device
        ).float()
        
        self.value_command[env_ids] = rands

    def _update_command(self):
        """Update logic at each step"""
        pass

    def _update_metrics(self):
        """Indicator statistical logic."""
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Debug visualization logic"""
        pass

    @property
    def command(self) -> torch.Tensor:
        """Return the current command tensor."""
        return self.value_command
        

@configclass
class UniformIntegerCommandCfg(CommandTermCfg):
    """Integer command configuration class."""
    class_type: type = UniformIntegerCommand
    num_commands: int = MISSING 
    params: dict = MISSING