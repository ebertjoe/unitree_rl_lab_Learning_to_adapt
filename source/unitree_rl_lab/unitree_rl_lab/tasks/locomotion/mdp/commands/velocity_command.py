from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """支持限制范围的水平速度指令配置。"""
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


class UniformIntegerCommand(CommandTerm):
    """整数指令实现类，负责步态 ID 的存储与随机重采样。"""

    def __init__(self, cfg: UniformIntegerCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # 预分配存储指令的张量 (num_envs, 1)
        self.value_command = torch.zeros(self.num_envs, 1, device=self.device)
        # 从配置中提取采样范围
        self.low, self.high = cfg.params["range"]

    # 增加 env_ids 参数
    def _resample_command(self, env_ids: torch.Tensor):
        """实现基类要求的抽象方法：采样逻辑。"""
        # 仅针对需要重采样的环境 ID 进行随机化
        rands = torch.randint(
            low=self.low, 
            high=self.high + 1, 
            size=(len(env_ids), 1), 
            device=self.device
        ).float()
        
        # 更新对应环境的指令值
        self.value_command[env_ids] = rands

    def _update_command(self):
        """每步更新逻辑。"""
        pass

    def _update_metrics(self):
        """指标统计逻辑。"""
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """调试可视化逻辑。"""
        pass

    @property
    def command(self) -> torch.Tensor:
        """返回给框架使用的指令值。"""
        return self.value_command
        

@configclass
class UniformIntegerCommandCfg(CommandTermCfg):
    """整数指令配置类。"""
    class_type: type = UniformIntegerCommand
    num_commands: int = MISSING 
    params: dict = MISSING