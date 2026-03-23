from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def beta_l_raibert(
    env: ManagerBasedRLEnv,
    gait_table: dict,
    gait_command_name: str = "gait_id",
    command_name: str = "base_velocity",
) -> torch.Tensor:
    device = env.device
    num_envs = env.num_envs
    num_legs = 4
    robot = env.scene["robot"]

    # 1. 获取所有环境当前的 gait_id (N,)
    gait_ids = env.command_manager.get_command(gait_command_name).long().flatten()

    # 2. 将 gait_table 缓存为 Tensor (只需转换一次)
    if not hasattr(env, "_gait_table_tensors"):
        keys = sorted([int(k) for k in gait_table.keys()])
        env._gait_periods = torch.tensor([gait_table[str(k)]["period"] for k in keys], device=device).float()
        env._gait_thresholds = torch.tensor([gait_table[str(k)]["threshold"] for k in keys], device=device).float()
        env._gait_offsets = torch.tensor([gait_table[str(k)]["offset"] for k in keys], device=device).float()  # (NumGaits, 4)
        env._gait_ks = torch.tensor([gait_table[str(k)]["k"] for k in keys], device=device).float()
        env._gait_znoms = torch.tensor([gait_table[str(k)]["z_nom"] for k in keys], device=device).float()
        env._gait_xlims = torch.tensor([gait_table[str(k)]["x_lim"] for k in keys], device=device).float()
        env._gait_ylims = torch.tensor([gait_table[str(k)]["y_lim"] for k in keys], device=device).float()
        env._gait_table_tensors = True

    # 3. 批量提取当前环境对应的参数
    # 使用 (N, 1) 以便在 (N, 4) 维度上正确广播
    period = env._gait_periods[gait_ids].unsqueeze(1)      # (N, 1)
    threshold = env._gait_thresholds[gait_ids].unsqueeze(1)  # (N, 1)
    offset = env._gait_offsets[gait_ids]                    # (N, 4)
    kx = ky = env._gait_ks[gait_ids].unsqueeze(1)           # (N, 1)
    z_nominal = env._gait_znoms[gait_ids].unsqueeze(1)      # (N, 1)
    x_limit = env._gait_xlims[gait_ids].unsqueeze(1)        # (N, 1)
    y_limit = env._gait_ylims[gait_ids].unsqueeze(1)        # (N, 1)

    # 1) 获取步数统计
    episode_length_buf = getattr(env, "episode_length_buf", env.episode_length if hasattr(env, "episode_length") else None)
    if episode_length_buf is None:
        raise AttributeError("无法从 env 获取步数统计，请确认环境类继承自 ManagerBasedRLEnv")
    # -------------------------
    # 2) 计算相位与步态周期
    # -------------------------
    t_exec = episode_length_buf.unsqueeze(1) * env.step_dt  # (N,1)
    global_phase = (t_exec % period) / period  # (N,1) 每个环境的全局相位，范围 [0,1)

    # off = torch.tensor(offset, device=device, dtype=torch.float32).view(1, num_legs)  # (1,4)
    # leg_phase = (global_phase + off) % 1.0  # (N,4)
    leg_phase = (global_phase + offset) % 1.0  # (N,4) 每条腿的相位都加上对应的偏移，形成独立的周期

    # stance if leg_phase < threshold
    c_ref = (leg_phase < threshold).to(torch.float32)  # (N,4)

    # R_WB: World-from-Body rotation matrix
    quat_WB = robot.data.root_quat_w
    R_WB = quat_wxyz_to_rotmat(quat_WB)      # (N,3,3)

    # -------------------------
    # 3) cache: prev contact + stored landing refs
    # -------------------------
    logic_leg_names = ["FR", "FL", "RR", "RL"]

    # 用 body_names 建映射（不要写死 16/15/18/17）
    if not hasattr(env, "_name_to_body_id"):
        env._name_to_body_id = {n: i for i, n in enumerate(robot.data.body_names)}
    name_to_body_id = env._name_to_body_id

    foot_body_ids = [name_to_body_id[f"{leg}_foot"] for leg in logic_leg_names]
    hip_body_ids = [name_to_body_id[f"{leg}_hip"] for leg in logic_leg_names]

    if not hasattr(env, "_raibert_prev_c"):
        env._raibert_prev_c = c_ref.clone()

    # 初始化 hip_static / p_ref（第一次进入时）
    if not hasattr(env, "_raibert_hip_pos_B_static"):
        hip_pos_w = robot.data.body_pos_w[:, hip_body_ids, :]          # (N,4,3)
        root_pos_w = robot.data.root_pos_w.unsqueeze(1)                # (N,1,3)
        hip_pos_B = torch.bmm(R_WB.transpose(1, 2), (hip_pos_w - root_pos_w).transpose(1, 2)).transpose(1, 2)
        env._raibert_hip_pos_B_static = hip_pos_B.clone()
        env._raibert_p_ref_B = torch.zeros((num_envs, 4, 3), device=device)

        env._raibert_p_ref_B[..., 2] = z_nominal.expand(-1, 4)  # (N,4,3) 初始时脚点参考位置就在髋部正下方，z 方向是标称高度

    # 取静态 Hip 基准
    hip = env._raibert_hip_pos_B_static  # (N,4,3)

    # reset 时：同步更新 hip_static + p_ref（只更新 reset env）
    if hasattr(env, "reset_buf"):
        reset_ids = torch.where(env.reset_buf)[0]
        if reset_ids.numel() > 0:
            # 更新 hip_static（避免上一回合残留）
            hip_pos_w_res = robot.data.body_pos_w[reset_ids][:, hip_body_ids, :]
            root_pos_w_res = robot.data.root_pos_w[reset_ids].unsqueeze(1)
            hip_pos_B_res = torch.bmm(R_WB[reset_ids].transpose(1, 2), (hip_pos_w_res - root_pos_w_res).transpose(1, 2)).transpose(1, 2)
            env._raibert_hip_pos_B_static[reset_ids] = hip_pos_B_res

            # p_ref 用“当前真实脚点(转到B) + 固定z_nominal”初始化
            foot_pos_w_res = robot.data.body_pos_w[reset_ids][:, foot_body_ids, :]
            p_rel_w_res = foot_pos_w_res - root_pos_w_res
            p_ref_B_res = torch.bmm(R_WB[reset_ids].transpose(1, 2), p_rel_w_res.transpose(1, 2)).transpose(1, 2)
            p_ref_B_res[..., 2] = z_nominal[reset_ids].expand(-1, 4)  # 修正这里，确保 reset 时脚点参考高度正确

            # prev_c 也重置一下更稳（避免 liftoff 误触发）
            env._raibert_prev_c[reset_ids] = c_ref[reset_ids]
            env._raibert_p_ref_B[reset_ids] = p_ref_B_res

    # -------------------------
    # 4) base velocity (world->body) + cmd velocity
    # -------------------------

    v_W = robot.data.root_lin_vel_w          # (N,3)
    v_B = torch.bmm(R_WB.transpose(1, 2), v_W.unsqueeze(-1)).squeeze(-1)  # (N,3)

    vxN, vyN = v_B[:, 0:1], v_B[:, 1:2]

    cmd = env.command_manager.get_command(command_name)  # (N, C)
    vcmd_xN, vcmd_yN = cmd[:, 0:1], cmd[:, 1:2]

    # 5) Raibert 启发式计算
    Tst = threshold * period
    Tswing = (1.0 - threshold) * period

    # swing phase in [0, 1], only meaningful during swing (leg_phase >= threshold)
    # leg_phase: (N,4) in [0,1)
    swing_phase = (leg_phase - threshold) / (1.0 - threshold)
    swing_phase = torch.clamp(swing_phase, 0.0, 1.0)   # (N,4)

    # textbook-like Raibert:
    # p_ref = p_hip + (1-phi)*Tswing*v + 0.5*Tst*v + K*(v-v_cmd)
    dx4 = (1.0 - swing_phase) * Tswing * vxN + 0.5 * Tst * vxN + kx * (vxN - vcmd_xN)  # (N,4)
    dy4 = (1.0 - swing_phase) * Tswing * vyN + 0.5 * Tst * vyN + ky * (vyN - vcmd_yN)  # (N,4)
    # dx4 = 0.5 * Tst * vxN + kx * (vxN - vcmd_xN)
    # dy4 = 0.5 * Tst * vyN + ky * (vyN - vcmd_yN)

    new_p = hip.clone()
    new_p[..., 0] += dx4
    new_p[..., 1] += dy4
    new_p[..., 2] = z_nominal.expand(-1, 4)

    # clamp around hip
    # new_p[..., 0] = torch.clamp(new_p[..., 0], hip[..., 0] + x_min, hip[..., 0] + x_max)
    # new_p[..., 1] = torch.clamp(new_p[..., 1], hip[..., 1] + y_min, hip[..., 1] + y_max)
    new_p[..., 0] = torch.clamp(new_p[..., 0], hip[..., 0] - x_limit, hip[..., 0] + x_limit)
    new_p[..., 1] = torch.clamp(new_p[..., 1], hip[..., 1] - y_limit, hip[..., 1] + y_limit)

    # 仅在离地瞬间更新
    liftoff = (env._raibert_prev_c > 0.5) & (c_ref < 0.5)
    env._raibert_p_ref_B = torch.where(liftoff.unsqueeze(-1), new_p, env._raibert_p_ref_B)
    env._raibert_prev_c = c_ref.clone()

    # 7) 叠加摆动高度（严格跟 swing/stance 对齐）
    swing_phase = (leg_phase - threshold) / (1.0 - threshold)
    swing_phase = torch.clamp(swing_phase, 0.0, 1.0)  # (N,4)
    swing_mask = (leg_phase >= threshold).float()

    z_swing = 0.10 * swing_mask * torch.sin(torch.pi * swing_phase)  # 0->1->0, only in swing

    p_ref_B_dynamic = env._raibert_p_ref_B.clone()
    p_ref_B_dynamic[..., 2] = z_nominal + z_swing  # (N,4,3) 这是动态的脚点参考位置，包含了 Raibert 调整和摆动高度
    # -------------------------
    # 8) 转换到世界坐标系 (计算奖励用)
    # -------------------------
    # 使用完整的旋转矩阵将 Body 系参考投影到 World 系方向
    p_ref_rel_w = torch.bmm(R_WB, p_ref_B_dynamic.transpose(1, 2)).transpose(1, 2)
    
    # 供奖励函数使用的绝对世界坐标
    p_ref_W = robot.data.root_pos_w.unsqueeze(1) + p_ref_rel_w

    # -------------------------
    # 9) 缓存与返回
    # -------------------------
    env.beta_contact_ref = (c_ref > 0.5)
    env.beta_foot_pos_ref_w = p_ref_W  # 供 r_f 使用
    
    # 将相对于质心的位移存入 env，这在计算奖励时比绝对坐标更稳定
    env.beta_p_ref_rel_w = p_ref_rel_w 

    # 关键修改：返回给神经网络的坐标是“相对”的,输入值永远在 [-1, 1] 左右
    beta = torch.zeros((num_envs, num_legs, 4), device=device)
    beta[..., 0] = c_ref
    beta[..., 1:] = p_ref_rel_w  # 改成相对位移

    return beta.view(num_envs, -1)


def quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Quaternion (w,x,y,z) -> rotation matrix R_WB."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = torch.zeros((N, 3, 3), device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)

    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)

    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def robot_state_s(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gait_command_name: str = "gait_id",
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    
    # 强制定义关节的逻辑顺序
    logic_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
    ]

    # 重新排序 asset_cfg.joint_ids 以匹配逻辑顺序
    name_to_joint_id = {n: i for i, n in enumerate(robot.data.joint_names)}
    asset_cfg.joint_ids = [name_to_joint_id[n] for n in logic_joint_names]

    # =========================
    # 2) 关节状态 (12)
    # =========================
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]              # (N,12)
    joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]              # (N,12)
    joint_torques = robot.data.applied_torque[:, asset_cfg.joint_ids]      # (N,12)

    # =========================
    # 3) 基础观测
    # =========================
    projected_gravity = robot.data.projected_gravity_b                     # (N,3)
    ang_vel = robot.data.root_ang_vel_b                                    # (N,3)
    lin_vel = robot.data.root_lin_vel_b                                    # (N,3)
    base_height = robot.data.root_pos_w[:, 2:3]                            # (N,1)

    # =========================
    # 4) Body/Foot/Hip 索引（用 body_names 自动查，不写死 15/16/17/18）
    # =========================
    logic_leg_names = ["FR", "FL", "RR", "RL"]
    name_to_body_id = {n: i for i, n in enumerate(robot.data.body_names)}
    foot_body_ids = [name_to_body_id[f"{leg}_foot"] for leg in logic_leg_names]
    hip_body_ids = [name_to_body_id[f"{leg}_hip"] for leg in logic_leg_names]

    # =========================
    # 5) 接触：按 contact_sensor.body_names 映射到逻辑腿顺序 FR,FL,RR,RL
    # =========================
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    logic_leg_names = ["FR", "FL", "RR", "RL"]

    # cache：contact sensor 的 body name -> id（注意：用 contact_sensor.body_names，不是 robot.data.body_names）
    if not hasattr(env, "_contact_name_to_id"):
        env._contact_name_to_id = {n: i for i, n in enumerate(contact_sensor.body_names)}

    contact_name_to_id = env._contact_name_to_id
    contact_foot_ids = [contact_name_to_id[f"{leg}_foot"] for leg in logic_leg_names]  # FR,FL,RR,RL

    foot_contact = (contact_sensor.data.current_contact_time[:, contact_foot_ids] > 0.0).float()  # (N,4)
    
    # 8 维 One-Hot (相当于给网络一个明确的“剧本标签”)
    current_gait_command = env.command_manager.get_command(gait_command_name)  # (N, 1)
    current_gait_id = current_gait_command[:, 0].long()  # 转为长整型索引
    gait_obs = torch.nn.functional.one_hot(current_gait_id, num_classes=8).float()  # (N, 8)

    # =========================
    # 6) Debug print every 200 steps
    # =========================
    if env.common_step_counter % 200 == 0:
        print("\n" + "🏠" * 10 + f" [DEBUG STEP: {env.common_step_counter}] 机身系校验 " + "🏠" * 10)

        # 基本状态
        print(f"Base Height: {base_height[0].item():.3f} | Gravity_B: {projected_gravity[0].cpu().numpy().round(2)}")
        print(f"Lin Vel (v_B):     {lin_vel[0].cpu().numpy().round(3)}")

        print("[contact_foot_ids (FR,FL,RR,RL)]", contact_foot_ids)
        print("[contact_foot_names]", [contact_sensor.body_names[i] for i in contact_foot_ids])

        # 名称列表
        print(f"Joint Names in Sim: {robot.data.joint_names}")
        print(f"Body Name Order:   {robot.data.body_names}")

        # 真实接触 bool（来自 contact sensor）
        # is_contact = contact_sensor.data.current_contact_time[:, env._contact_body_ids] > 0.0
        is_contact = contact_sensor.data.current_contact_time[:, contact_foot_ids] > 0.0
        print("[is_contact bool]", is_contact[0].tolist())
        print("[sensor_cfg.body_ids]", sensor_cfg.body_ids)
        print("[foot_body_ids(body_names)]", foot_body_ids)

        # 参考接触 bool（来自 beta）
        if hasattr(env, "beta_contact_ref"):
            print("[ref_contact bool]", env.beta_contact_ref[0].tolist())
        
        contact_sensor = env.scene.sensors[sensor_cfg.name]

        print("\n[CONTACT SENSOR INFO]")
        print("contact_sensor.body_names =", contact_sensor.body_names)
        print("len(body_names) =", len(contact_sensor.body_names))
        print("contact_time.shape =", tuple(contact_sensor.data.current_contact_time.shape))

        # 打印解析出来的 joint_ids 对应名字（确认顺序）
        print("\n[Resolved Joint Order]")
        print([robot.data.joint_names[i] for i in asset_cfg.joint_ids])

        # 打印解析出来的 foot/hip body ids
        print("\n[Resolved Body IDs]")
        print("Hip body ids:", list(zip(logic_leg_names, hip_body_ids)))
        print("Foot body ids:", list(zip(logic_leg_names, foot_body_ids)))

        # ========== A) 世界系足端 + Δ ==========
        real_foot_pos_w = robot.data.body_pos_w[:, foot_body_ids, :]       # (N,4,3)

        if not hasattr(env, "_debug_prev_foot_pos_w"):
            env._debug_prev_foot_pos_w = real_foot_pos_w.clone()
        delta_foot_w = real_foot_pos_w - env._debug_prev_foot_pos_w        # (N,4,3)
        env._debug_prev_foot_pos_w = real_foot_pos_w.clone()

        # ========== B) body系足端 ==========
        root_pos_w = robot.data.root_pos_w.unsqueeze(1)                    # (N,1,3)
        rel_pos_w = real_foot_pos_w - root_pos_w                           # (N,4,3)

        quat_WB = robot.data.root_quat_w
        R_WB = quat_wxyz_to_rotmat(quat_WB)                                # (N,3,3) body->world
        real_foot_pos_b = torch.bmm(
            R_WB.transpose(1, 2),
            rel_pos_w.transpose(1, 2)
        ).transpose(1, 2)                                                  # (N,4,3)

        # ΔGravity
        if not hasattr(env, "_debug_prev_gravity_b"):
            env._debug_prev_gravity_b = projected_gravity.clone()
        delta_g = (projected_gravity - env._debug_prev_gravity_b)[0].cpu().numpy().round(4)
        env._debug_prev_gravity_b = projected_gravity.clone()

        # 世界系足端打印
        print("\n[Foot Position (World Frame)]")
        print(f"{'Leg':<5} | {'Foot_W (X,Y,Z)':<28} | {'ΔFoot_W (X,Y,Z)':<28} | {'Con'}")
        print("-" * 90)
        for i, leg in enumerate(logic_leg_names):
            fw = real_foot_pos_w[0, i, :3].cpu().numpy().round(3)
            dfw = delta_foot_w[0, i, :3].cpu().numpy().round(4)
            con = foot_contact[0, i].item()
            print(f"{leg:<5} | {str(fw):<28} | {str(dfw):<28} | {con}")
        print(f"\nΔGravity_B: {delta_g}")

        # RR腿深度检查：现在 RR 起始索引仍是 6（因为逻辑顺序固定）
        rr_joint_pos = joint_pos[0, 6:9]
        rr_joint_vel = joint_vel[0, 6:9]
        print(f"\n[RR Leg Deep Check]")
        print(f"RR_Hip   | Pos: {rr_joint_pos[0]:.4f} | Vel: {rr_joint_vel[0]:.4f}")
        print(f"RR_Thigh | Pos: {rr_joint_pos[1]:.4f} | Vel: {rr_joint_vel[1]:.4f}")
        print(f"RR_Calf  | Pos: {rr_joint_pos[2]:.4f} | Vel: {rr_joint_vel[2]:.4f}")

        # 关键：验证 index 6 在“逻辑关节数组”里到底对应什么真实名字
        print(f"Index 6 in logic is actually: {robot.data.joint_names[asset_cfg.joint_ids[6]]}")

        # Hip 静态参考、脚点参考、实际脚点（Body Frame）
        print("\n[Leg Ordering Verification (Body Frame)]")
        print(f"{'Leg':<5} | {'Hip_Base (X,Y)':<18} | {'Ref_Foot (X,Y)':<18} | {'Real_Foot_B (X,Y,Z)':<25} | {'Con'}")
        print("-" * 110)

        for i, leg in enumerate(logic_leg_names):
            h_base = "N/A"
            if hasattr(env, "_raibert_hip_pos_B_static"):
                h_base = env._raibert_hip_pos_B_static[0, i, :2].cpu().numpy().round(3)

            f_ref_b = "N/A"
            if hasattr(env, "_raibert_p_ref_B"):
                f_ref_b = env._raibert_p_ref_B[0, i, :2].cpu().numpy().round(3)

            f_real_b = real_foot_pos_b[0, i, :3].cpu().numpy().round(3)
            con = foot_contact[0, i].item()
            print(f"{leg:<5} | {str(h_base):<18} | {str(f_ref_b):<18} | {str(f_real_b):<25} | {con}")

        # 世界系跟踪误差（如存在）
        if hasattr(env, "beta_foot_pos_ref_w"):
            total_err = torch.norm(real_foot_pos_w[0] - env.beta_foot_pos_ref_w[0], dim=-1).mean().item()
            print(f"\nMean Tracking Error (World): {total_err:.4f}")

        # 全 body dump (World + Body)
        print("\n[Full Body Position Dump]")
        print(f"{'ID':<4} | {'Body Name':<15} | {'World (X,Y,Z)':<25} | {'Body (X,Y,Z)':<25}")
        print("-" * 95)

        all_body_pos_w = robot.data.body_pos_w                             # (N, Nbodies, 3)
        root_pos_w_full = robot.data.root_pos_w.unsqueeze(1)               # (N,1,3)
        rel_pos_w_full = all_body_pos_w - root_pos_w_full                  # (N,Nbodies,3)
        all_body_pos_b = torch.bmm(
            R_WB.transpose(1, 2),
            rel_pos_w_full.transpose(1, 2)
        ).transpose(1, 2)                                                  # (N,Nbodies,3)

        for idx, bname in enumerate(robot.data.body_names):
            w_pos = all_body_pos_w[0, idx].cpu().numpy().round(3)
            b_pos = all_body_pos_b[0, idx].cpu().numpy().round(3)
            print(f"{idx:<4} | {bname:<15} | {str(w_pos):<25} | {str(b_pos):<25}")

        print("🏠" * 35 + "\n")

    # =========================
    # 7) 拼接观测向量(50+8)
    # =========================
    return torch.cat(
        [
            projected_gravity,
            joint_pos,
            ang_vel,
            joint_vel,
            lin_vel,
            base_height,
            joint_torques,
            foot_contact,
            gait_obs           # +8 (额外标签)
        ],
        dim=-1,
    )