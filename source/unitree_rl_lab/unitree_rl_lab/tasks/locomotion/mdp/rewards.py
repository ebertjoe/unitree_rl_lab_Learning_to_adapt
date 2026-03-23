from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# psi: x -> 1 - tanh(x^2)
# -----------------------------------------------------------------------------
def psi(x: torch.Tensor) -> torch.Tensor:
    """Paper psi: ψ(x) = 1 - tanh(x^2)."""
    return 1.0 - torch.tanh(x * x)


# -----------------------------------------------------------------------------
# r_eta (Eq. 6): efficiency term
# r_eta = ||q⃛||^2 + ||tau||^2 + ||q* - q*_{t-1}||   :contentReference[oaicite:2]{index=2}
# -----------------------------------------------------------------------------
def r_eta(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_dt_scaling: bool = False,
    clamp_jerk: float | None = 5.0  # 建议给一个默认 clamp
) -> torch.Tensor:
    """Efficiency penalty (aligned with paper Eq. 6, with normalization)."""
    asset: Articulation = env.scene[asset_cfg.name]
    num_dof = len(asset_cfg.joint_ids)

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]            
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]        

    # 获取 Action
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "action"):
        q_star = env.action_manager.action
    else:
        q_star = env.action_buf  # Fallback

    if q_star.shape[-1] != num_dof:
        q_star = q_star[:, asset_cfg.joint_ids]

    # 初始化 Buffers
    if not hasattr(env, "_paper_eta_qvel_prev"):
        env._paper_eta_qvel_prev = qvel.clone()
        env._paper_eta_qvel_prev2 = qvel.clone()
        env._paper_eta_action_prev = q_star.clone()

    # 处理 Resets
    if hasattr(env, "reset_buf"):
        ids = torch.where(env.reset_buf)[0]
        if len(ids) > 0:
            env._paper_eta_qvel_prev[ids] = qvel[ids]
            env._paper_eta_qvel_prev2[ids] = qvel[ids]
            env._paper_eta_action_prev[ids] = q_star[ids]

    # --- 计算分项 ---

    # 1. Jerk: 使用 mean 而不是 sum，可以抵消自由度数量带来的数值放大
    jerk = qvel - 2.0 * env._paper_eta_qvel_prev + env._paper_eta_qvel_prev2
    if use_dt_scaling:
        dt = float(env.step_dt)
        jerk = jerk / (dt * dt)
    if clamp_jerk is not None:
        jerk = torch.clamp(jerk, -clamp_jerk, clamp_jerk)
    # 使用 mean() 保证不论机器人有多少关节，数值量级都一致
    jerk_term = torch.mean(jerk * jerk, dim=-1)

    # 2. Torque: 使用 mean()。这是防止 Raw 破千的关键
    tau_term = torch.mean(qfrc * qfrc, dim=-1)

    # 3. Action rate: 动作差值
    dact = q_star - env._paper_eta_action_prev
    dact_term = torch.mean(dact * dact, dim=-1) 

    # 论文公式的总和
    total_eta = jerk_term + tau_term + dact_term

    # --- 保留打印功能 ---
    if env.common_step_counter % 200 == 0:
        print("\n" + "-" * 30)
        print(f"[ETA DETAIL | Step {env.common_step_counter}]")
        print(f"  - Jerk (mean sq):   {jerk_term[0].item():.4f}")
        print(f"  - Torque (mean sq): {tau_term[0].item():.4f}")
        print(f"  - Action (mean sq): {dact_term[0].item():.4f}")
        print(f"  >> Total Raw:       {total_eta[0].item():.4f}")
        # 这里假设权重是 -0.1，如果你依然用 -1.5，这个项依然会非常大
        print(f"  >> Weighted Score (*-0.1): {total_eta[0].item() * -0.1:.4f}")
        print("-" * 30)

    # 更新 Buffers (使用 copy_ 性能更好，不产生新内存)
    env._paper_eta_qvel_prev2.copy_(env._paper_eta_qvel_prev)
    env._paper_eta_qvel_prev.copy_(qvel)
    env._paper_eta_action_prev.copy_(q_star)

    return total_eta

# -----------------------------------------------------------------------------
# r_vcmd (Eq. 7): velocity command tracking
# r_vcmd = psi( ||vB - vB_cmd||^2 ),  vB=[vx, vy, wz]   :contentReference[oaicite:3]{index=3}
# -----------------------------------------------------------------------------


def r_vcmd(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wz_scale: float = 1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    v_cmd = env.command_manager.get_command(command_name)  # (N,3) [vx_cmd, vy_cmd, wz_cmd]

    v_xy = asset.data.root_lin_vel_b[:, :2]                # (N,2) body frame
    w_z = asset.data.root_ang_vel_b[:, 2:3] * wz_scale     # (N,1)
    v = torch.cat([v_xy, w_z], dim=-1)                     # (N,3)

    v_cmd2 = v_cmd.clone()
    v_cmd2[:, 2] = v_cmd2[:, 2] * wz_scale

    err2 = torch.sum((v - v_cmd2) ** 2, dim=-1)            # ||v - vcmd||^2
    reward = psi(err2)

    # 调试打印 (w_vcmd = 50.0)
    if env.common_step_counter % 200 == 0:
        print(f"[REWARD VCMD] Raw: {reward[0].item():.4f} | Weighted (*50): {reward[0].item() * 50.0:.4f}")

    return reward

# -----------------------------------------------------------------------------
# r_f (Eq. 8): gait reference tracking within beta_L
# r_f = |c_err| + sum_{i=1..4} ||p_i - p_i_ref||^2         :contentReference[oaicite:4]{index=4}
# -----------------------------------------------------------------------------


def r_f(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gait tracking cost (Eq. 8).

    - reference:
        env.beta_p_ref_rel_w : (N,4,3)  world-frame relative foot ref (foot_ref_w - root_w)
        env.beta_contact_ref : (N,4)    stance reference bool in LOGIC order [FR,FL,RR,RL]
    - measurement:
        real_rel_pos_w : (N,4,3) from articulation body_pos_w, LOGIC order [FR,FL,RR,RL]
        is_contact     : (N,4)   from contact sensor, LOGIC order [FR,FL,RR,RL]
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # ------------------------------------------------------------
    # 0) beta refs guard
    # ------------------------------------------------------------
    if not hasattr(env, "beta_p_ref_rel_w") or not hasattr(env, "beta_contact_ref"):
        if env.common_step_counter % 200 == 0:
            print("[REWARD GAIT] beta refs missing -> return 0.0")
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ref_rel_pos_w = env.beta_p_ref_rel_w                 # (N,4,3) LOGIC [FR,FL,RR,RL]
    contact_ref_bool = env.beta_contact_ref.bool()       # (N,4)   LOGIC [FR,FL,RR,RL]

    # ------------------------------------------------------------
    # 1) articulation foot ids (asset.data.body_names indexing)
    #    for body_pos_w ONLY
    # ------------------------------------------------------------
    logic_leg_names = ["FR", "FL", "RR", "RL"]
    if not hasattr(env, "_rf_foot_body_ids"):
        name_to_body_id = {n: i for i, n in enumerate(asset.data.body_names)}
        env._rf_foot_body_ids = [name_to_body_id[f"{leg}_foot"] for leg in logic_leg_names]
    foot_body_ids = env._rf_foot_body_ids

    foot_pos_w = asset.data.body_pos_w[:, foot_body_ids, :]      # (N,4,3) LOGIC
    root_pos_w = asset.data.root_pos_w.unsqueeze(1)              # (N,1,3)
    real_rel_pos_w = foot_pos_w - root_pos_w                     # (N,4,3) LOGIC

    # ------------------------------------------------------------
    # 2) contact ids (contact_sensor.body_names indexing)
    #    IMPORTANT: DO NOT use sensor_cfg.body_ids here (it may be FL,FR,RL,RR)
    # ------------------------------------------------------------
    if not hasattr(env, "_rf_contact_foot_ids"):
        # contact sensor 的名字列表就是它自己的索引体系
        contact_name_to_id = {n: i for i, n in enumerate(contact_sensor.body_names)}
        env._rf_contact_foot_ids = [contact_name_to_id[f"{leg}_foot"] for leg in logic_leg_names]
    contact_foot_ids = env._rf_contact_foot_ids

    is_contact = contact_sensor.data.current_contact_time[:, contact_foot_ids] > 0.0  # (N,4) LOGIC

    # contact mismatch cost
    c_err = (is_contact ^ contact_ref_bool).float()
    c_cost = c_err.mean(dim=-1)  # (N,)

    # ------------------------------------------------------------
    # 3) swing foot xy tracking cost (only swing legs)
    # ------------------------------------------------------------
    pos_dist_xy = torch.norm(real_rel_pos_w[..., :2] - ref_rel_pos_w[..., :2], dim=-1)  # (N,4)

    sigma_xy = 0.10
    pos_cost = 1.0 - torch.exp(-(pos_dist_xy ** 2) / (2.0 * sigma_xy ** 2))  # (N,4)

    swing_mask = (~contact_ref_bool).float()
    num_swing = swing_mask.sum(dim=-1)

    p_cost = torch.where(
        num_swing > 0.0,
        (pos_cost * swing_mask).sum(dim=-1) / (num_swing + 1e-6),
        torch.zeros_like(num_swing),
    )

    total_cost = c_cost + p_cost

    # ------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------
    if env.common_step_counter == 0:
        print("[r_f DEBUG] contact_sensor.body_names =", contact_sensor.body_names)
        print("[r_f DEBUG] contact_foot_ids(FR,FL,RR,RL) =", contact_foot_ids)
        print("[r_f DEBUG] contact_foot_names =", [contact_sensor.body_names[i] for i in contact_foot_ids])
        print("[r_f DEBUG] foot_body_ids (asset.data.body_names) =", foot_body_ids)

    if env.common_step_counter % 200 == 0:
        mean_xy = ((pos_dist_xy * swing_mask).sum(dim=-1) / (num_swing + 1e-6))[0].item()
        print(f"[REWARD GAIT] c_cost={c_cost[0].item():.4f} p_cost={p_cost[0].item():.4f} "
              f"total={total_cost[0].item():.4f} | mean_swing_xy_err={mean_xy:.4f}m")

    return total_cost
# -----------------------------------------------------------------------------
# r_stab (Eq. 9): stability
# r_stab = sum_{i=1..F} ||p_dot_i||^2 + ||omega_xy||^2
#          + psi(||alpha R_B - alpha R_B^des||^2)
#          - psi((zB - zNom)^2)
#          + ||q_hip||^2                                  :contentReference[oaicite:5]{index=5}
#
# We implement "alpha R_B" using gravity direction in body frame:
#   alpha=[0,0,1] selects the vertical axis; projected_gravity_b is commonly used
#   for this in simulators (it’s already a body-frame vector reflecting orientation).
# You pass desired_gravity_b for alpha R_B^des.
# -----------------------------------------------------------------------------


def r_stab(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    desired_gravity_b: list[float] | torch.Tensor,
    gait_table: dict,                # 新增：接收 GAIT_CONFIGS
    gait_command_name: str = "gait_id",
    hip_joint_ids: list[int] = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    device = env.device
    
    # -------------------------
    # A) 动态获取向量化步态参数
    # -------------------------
    # 1. 获取所有环境当前的 gait_id (N,)
    gait_ids = env.command_manager.get_command(gait_command_name).long().flatten()

    # 2. 确保 env 已经缓存了步态 Tensor (如果 beta_l 先运行，这里通常已存在)
    if not hasattr(env, "_gait_table_tensors"):
        keys = sorted([int(k) for k in gait_table.keys()])
        env._gait_znoms = torch.tensor([abs(gait_table[str(k)]["z_nom"]) for k in keys], device=device).float()
        env._gait_table_tensors = True

    # 3. 提取当前所有环境对应的标称高度 (N,)
    z_nom_t = env._gait_znoms[gait_ids]

    # stance mask：LOGIC order [FR,FL,RR,RL]
    stance = env.beta_contact_ref.bool()  # (N,4)

    # --- 用 articulation body_names 找 foot ids，保证顺序也是 [FR,FL,RR,RL] ---
    logic_leg_names = ["FR", "FL", "RR", "RL"]
    if not hasattr(env, "_rstab_foot_body_ids"):
        name_to_body_id = {n: i for i, n in enumerate(asset.data.body_names)}
        env._rstab_foot_body_ids = [name_to_body_id[f"{leg}_foot"] for leg in logic_leg_names]
    foot_body_ids = env._rstab_foot_body_ids

    # 1) 获取水平速度 (XY) - LOGIC order
    foot_vel_xy = asset.data.body_lin_vel_w[:, foot_body_ids, :2]  # (N,4,2)

    # 2) 速度平方
    slip_each = torch.sum(foot_vel_xy * foot_vel_xy, dim=-1)  # (N,4)

    # 3) 只惩罚期望 stance 的脚（LOGIC order对齐）
    slip_term = torch.sum(slip_each * stance.float(), dim=-1) * 2.0

    # 角速度 roll/pitch
    omega_xy = asset.data.root_ang_vel_b[:, :2]
    omega_term = torch.sum(omega_xy * omega_xy, dim=-1)

    # desired_gravity_b：cfg 里是 list，需要变成 (N,3) tensor
    if not isinstance(desired_gravity_b, torch.Tensor):
        desired_gravity_b = torch.tensor(desired_gravity_b, device=env.device, dtype=asset.data.projected_gravity_b.dtype)
    desired_gravity_b = desired_gravity_b.view(1, 3).repeat(env.num_envs, 1)  # (N,3)

    # 姿态项
    alphaR_B = asset.data.projected_gravity_b
    orient_err2 = torch.sum((alphaR_B - desired_gravity_b) ** 2, dim=-1)
    orient_term = psi(orient_err2)

    # 高度项
    zB = asset.data.root_pos_w[:, 2]
    height_term = 1 - psi((zB - z_nom_t) ** 2)

    # 髋关节项
    qhip = asset.data.joint_pos[:, hip_joint_ids]
    hip_term = torch.sum(qhip * qhip, dim=-1)

    # 综合返回
    total_stab_error = slip_term + omega_term + orient_term + height_term + hip_term

    # 调试打印 (w_stab = -1.0)
    if env.common_step_counter % 200 == 0:
        print(f"[STAB DETAIL | Step {env.common_step_counter}]")
        print(f"  1. Slip (足端滑行): {slip_term[0].item():.4f}")
        print(f"  2. Omega (角速度): {omega_term[0].item():.4f}")
        print(f"  3. Orient (姿态):  {orient_term[0].item():.4f}")
        print(f"  4. Height (高度):  {height_term[0].item():.4f} (zB={zB[0].item():.3f})")
        print(f"  5. Hip (髋关节):   {hip_term[0].item():.4f}")
        print(f"  >> Total Raw Error: {total_stab_error[0].item():.4f}")
        print(f"  >> Weighted Score (*-1.0): {total_stab_error[0].item() * -1.0:.4f}")
        # 额外打印高度，确认 zB 是否正常
        print(f"      -> Detail: zB={zB[0].item():.3f}, Height_Err={height_term[0].item():.3f}, Orient_Err={orient_term[0].item():.3f}")

    return total_stab_error
