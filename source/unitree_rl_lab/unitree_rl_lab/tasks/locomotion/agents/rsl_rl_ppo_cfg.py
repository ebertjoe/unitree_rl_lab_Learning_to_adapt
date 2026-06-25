# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2PaperPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 1
    num_steps_per_env = 24
    # 论文中提到训练 20,000 次迭代
    max_iterations = 10000
    save_interval = 200
    experiment_name = "unitree_go2_locomotion_paper"

    # 1. 显式清除所有默认的观察项设置
    policy_observations = ["policy"] 
    critic_observations = ["critic"]

    # 2. 确保 obs_groups 字典结构完全正确
    obs_groups = {
        "policy": "policy",
        "critic": "critic",
    }
    
    # 强制覆盖任何可能的默认 key
    def __post_init__(self):
        # 确保在运行时，配置里绝对不会出现名为 'p' 的项
        self.obs_groups = {
            "policy": ["policy"],
            "critic": ["critic"],
        }

    # RaiSim 提到的 observation normalization 对齐
    empirical_normalization = True
    # ---------------------------------------------------------
    # 策略网络配置
    # ---------------------------------------------------------
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="lrelu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,

        # locomotion 常用更小 entropy
        entropy_coef=0.005,

        num_learning_epochs=5,
        num_mini_batches=8,

        # 四足更稳的 lr
        learning_rate=3.0e-4,
        schedule="adaptive",

        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
