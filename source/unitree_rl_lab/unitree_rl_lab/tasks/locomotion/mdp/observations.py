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


def gait_conditioned_base_velocity(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    gait_command_name: str = "gait_id",
    stand_gait_id: int = 6,
) -> torch.Tensor:
    v_cmd = env.command_manager.get_command(command_name).clone()
    gait_ids = env.command_manager.get_command(gait_command_name).long().flatten()
    stand_mask = (gait_ids == stand_gait_id).unsqueeze(1)
    return torch.where(stand_mask, torch.zeros_like(v_cmd), v_cmd)


def beta_l_raibert(
    env: ManagerBasedRLEnv,
    gait_table: dict,
    gait_command_name: str = "gait_id",
    command_name: str = "base_velocity",
) -> torch.Tensor:
    # -------------------------
    # Each simulation step is only truly updated once
    # -------------------------
    cur_step = int(env.common_step_counter)
    if (
        hasattr(env, "_beta_last_step")
        and env._beta_last_step == cur_step
        and hasattr(env, "_beta_last_value")
    ):
        return env._beta_last_value

    device = env.device
    num_envs = env.num_envs
    num_legs = 4
    robot = env.scene["robot"]

    # 1) commands
    gait_ids = env.command_manager.get_command(gait_command_name).long().flatten()   # (N,)
    # v_cmd = env.command_manager.get_command(command_name)                           # (N,3) [vx, vy, wz]
    v_cmd = gait_conditioned_base_velocity(
        env,
        command_name=command_name,
        gait_command_name=gait_command_name,
        stand_gait_id=6,
    )

    # 2) cache gait table
    if not hasattr(env, "_gait_table_tensors"):
        keys = sorted([int(k) for k in gait_table.keys()])
        env._gait_periods = torch.tensor(
            [gait_table[str(k)]["period"] for k in keys], device=device
        ).float()
        env._gait_thresholds = torch.tensor(
            [gait_table[str(k)]["threshold"] for k in keys], device=device
        ).float()
        env._gait_offsets = torch.tensor(
            [gait_table[str(k)]["offset"] for k in keys], device=device
        ).float()  # (G,4)
        env._gait_ks = torch.tensor(
            [gait_table[str(k)]["k"] for k in keys], device=device
        ).float()
        env._gait_znoms = torch.tensor(
            [gait_table[str(k)]["z_nom"] for k in keys], device=device
        ).float()
        env._gait_xlims = torch.tensor(
            [gait_table[str(k)]["x_lim"] for k in keys], device=device
        ).float()
        env._gait_ylims = torch.tensor(
            [gait_table[str(k)]["y_lim"] for k in keys], device=device
        ).float()

        env._prev_gait_ids = gait_ids.clone()
        env._phase_compensation = torch.zeros((num_envs, 1), device=device)
        env._current_period_blended = env._gait_periods[gait_ids].unsqueeze(1)
        env._current_znoms_blended = env._gait_znoms[gait_ids].unsqueeze(1)
        env._gait_table_tensors = True

    # 3) target gait params
    target_period = env._gait_periods[gait_ids].unsqueeze(1)       # (N,1)
    threshold = env._gait_thresholds[gait_ids].unsqueeze(1)        # (N,1)
    offset = env._gait_offsets[gait_ids]                           # (N,4)
    kx = ky = env._gait_ks[gait_ids].unsqueeze(1)                  # (N,1)
    z_nominal_target = env._gait_znoms[gait_ids].unsqueeze(1)      # (N,1)
    x_limit = env._gait_xlims[gait_ids].unsqueeze(1)               # (N,1)
    y_limit = env._gait_ylims[gait_ids].unsqueeze(1)               # (N,1)

    # 4) smooth transition of period / z_nom
    blend_alpha = 0.1
    env._current_period_blended = (
        blend_alpha * target_period + (1.0 - blend_alpha) * env._current_period_blended
    )
    env._current_znoms_blended = (
        blend_alpha * z_nominal_target + (1.0 - blend_alpha) * env._current_znoms_blended
    )

    period = env._current_period_blended
    z_nominal = env._current_znoms_blended

    # 5) body ids / rotation
    logic_leg_names = ["FR", "FL", "RR", "RL"]
    if not hasattr(env, "_name_to_body_id"):
        env._name_to_body_id = {n: i for i, n in enumerate(robot.data.body_names)}

    quat_WB = robot.data.root_quat_w
    R_WB = quat_wxyz_to_rotmat(quat_WB)                  # body -> world (full)
    R_WB_yaw = quat_wxyz_to_yaw_rotmat(quat_WB)         # body -> world (yaw only)

    # 6) init static hip + ref cache
    if not hasattr(env, "_raibert_hip_pos_B_static"):
        hip_pos_B_single = get_go2_hip_positions_B(
            device=device,
            dtype=robot.data.root_pos_w.dtype,
        )  # (4,3)

        hip_pos_B = hip_pos_B_single.unsqueeze(0).repeat(num_envs, 1, 1)  # (N,4,3)

        env._raibert_hip_pos_B_static = hip_pos_B.clone()
        env._raibert_p_ref_B = hip_pos_B.clone()
        env._raibert_p_ref_B[..., 2] = z_nominal.expand(-1, 4)

    # 7) reset handling
    if hasattr(env, "reset_buf"):
        reset_ids = torch.where(env.reset_buf)[0]
        if reset_ids.numel() > 0:
            hip_pos_B_single = get_go2_hip_positions_B(
                device=device,
                dtype=robot.data.root_pos_w.dtype,
            )

            hip_pos_B = hip_pos_B_single.unsqueeze(0).repeat(reset_ids.numel(), 1, 1)

            env._raibert_hip_pos_B_static[reset_ids] = hip_pos_B
            env._raibert_p_ref_B[reset_ids] = hip_pos_B
            env._raibert_p_ref_B[reset_ids, :, 2] = z_nominal[reset_ids].expand(-1, 4)

            if hasattr(env, "_raibert_prev_c"):
                env._raibert_prev_c[reset_ids] = 1.0

    # 8) phase continuity
    episode_length_buf = getattr(
        env,
        "episode_length_buf",
        env.episode_length if hasattr(env, "episode_length") else None,
    )
    if episode_length_buf is None:
        raise AttributeError("env missing episode_length_buf")

    t_exec = episode_length_buf.unsqueeze(1) * env.step_dt  # (N,1)

    gait_changed = (gait_ids != env._prev_gait_ids).unsqueeze(1)
    if gait_changed.any():
        old_period = env._gait_periods[env._prev_gait_ids].unsqueeze(1)
        new_comp = t_exec - (t_exec - env._phase_compensation) * (period / old_period)
        env._phase_compensation = torch.where(gait_changed, new_comp, env._phase_compensation)

    env._prev_gait_ids = gait_ids.clone()

    global_phase = ((t_exec - env._phase_compensation) % period) / period   # (N,1)
    leg_phase = (global_phase + offset) % 1.0                               # (N,4)
    c_ref = (leg_phase < threshold).float()                                 # (N,4)

    # 9) body velocities
    # full rotation
    v_B = torch.bmm(
        R_WB.transpose(1, 2),
        robot.data.root_lin_vel_w.unsqueeze(-1)
    ).squeeze(-1)                                                           # (N,3)

    yaw_vel = robot.data.root_ang_vel_b[:, 2:3]                             # (N,1)

    # 10) hip base
    hip_base = env._raibert_hip_pos_B_static.clone()                        # (N,4,3)
    is_bound_1d = (gait_ids == 0)                                           # (N,)

    # 11) Raibert foot placement
    twist_x = torch.zeros_like(v_B[:, 0:1])
    twist_y = torch.zeros_like(v_B[:, 0:1])

    Tst = threshold * period
    dx4 = 0.5 * Tst * v_B[:, 0:1] + kx * (v_B[:, 0:1] - v_cmd[:, 0:1]) + twist_x
    dy4 = 0.5 * Tst * v_B[:, 1:2] + ky * (v_B[:, 1:2] - v_cmd[:, 1:2]) - twist_y

    new_p = hip_base.clone()
    new_p[..., 0] += dx4
    new_p[..., 1] += dy4
    new_p[..., 2] = z_nominal.expand(-1, 4)

    # 12) clamp around hip_base
    new_p[..., 0] = torch.clamp(
        new_p[..., 0],
        hip_base[..., 0] - x_limit,
        hip_base[..., 0] + x_limit,
    )
    new_p[..., 1] = torch.clamp(
        new_p[..., 1],
        hip_base[..., 1] - y_limit,
        hip_base[..., 1] + y_limit,
    )

    # 13) Bound-specific geometry constraint
    # if is_bound_1d.any():
    #     # Use the current dx4, but restrict the hind legs from being further forward than the forelegs.
    #     front_dx = dx4[is_bound_1d, 0]
    #     rear_dx = dx4[is_bound_1d, 0]

    #     # Geometric constraint: Hind leg forward movement <= Foreleg forward movement
    #     rear_dx = torch.minimum(rear_dx, front_dx)

    #     # forelegs
    #     new_p[is_bound_1d, 0, 0] = hip_base[is_bound_1d, 0, 0] + front_dx
    #     new_p[is_bound_1d, 1, 0] = hip_base[is_bound_1d, 1, 0] + front_dx

    #     # hind legs
    #     new_p[is_bound_1d, 2, 0] = hip_base[is_bound_1d, 2, 0] + rear_dx
    #     new_p[is_bound_1d, 3, 0] = hip_base[is_bound_1d, 3, 0] + rear_dx

    #     # y maintains left-right symmetry
    #     new_p[is_bound_1d, 0, 1] = hip_base[is_bound_1d, 0, 1] + dy4[is_bound_1d, 0]
    #     new_p[is_bound_1d, 1, 1] = hip_base[is_bound_1d, 1, 1] + dy4[is_bound_1d, 0]
    #     new_p[is_bound_1d, 2, 1] = hip_base[is_bound_1d, 2, 1] + dy4[is_bound_1d, 0]
    #     new_p[is_bound_1d, 3, 1] = hip_base[is_bound_1d, 3, 1] + dy4[is_bound_1d, 0]

    # 14) liftoff detection
    if not hasattr(env, "_raibert_prev_c"):
        env._raibert_prev_c = c_ref.clone()

    prev_c = env._raibert_prev_c
    liftoff = (prev_c > 0.5) & (c_ref < 0.5)

    # Bound: pair liftoff
    # if is_bound_1d.any():
    #     front_liftoff = (
    #         ((prev_c[:, 0] > 0.5) | (prev_c[:, 1] > 0.5))
    #         & ((c_ref[:, 0] < 0.5) & (c_ref[:, 1] < 0.5))
    #     )
    #     rear_liftoff = (
    #         ((prev_c[:, 2] > 0.5) | (prev_c[:, 3] > 0.5))
    #         & ((c_ref[:, 2] < 0.5) & (c_ref[:, 3] < 0.5))
    #     )

    #     liftoff[is_bound_1d, 0] = front_liftoff[is_bound_1d]
    #     liftoff[is_bound_1d, 1] = front_liftoff[is_bound_1d]
    #     liftoff[is_bound_1d, 2] = rear_liftoff[is_bound_1d]
    #     liftoff[is_bound_1d, 3] = rear_liftoff[is_bound_1d]

    env._raibert_p_ref_B = torch.where(
        liftoff.unsqueeze(-1),
        new_p,
        env._raibert_p_ref_B,
    )
    env._raibert_prev_c = c_ref.clone()

    # 15) swing uses locked target, stance uses dynamic target
    # stance_mask = (c_ref > 0.5).unsqueeze(-1)
    # p_ref_B_final = torch.where(
    #     stance_mask,
    #     new_p,                   # stance: dynamic support reference
    #     env._raibert_p_ref_B,    # swing: locked target
    # )
    p_ref_B_final = env._raibert_p_ref_B.clone()

    # 16) swing height
    swing_mask = (c_ref < 0.5).float()
    x = torch.clamp((leg_phase - threshold) / (1.0 - threshold), 0.0, 1.0)
    step_height = 0.10
    # z_swing = step_height * torch.sin(torch.pi * x) * swing_mask
    z_swing = 0.5 * step_height * (1.0 - torch.cos(2.0 * torch.pi * x)) * swing_mask
    p_ref_B_final[..., 2] = z_nominal.expand(-1, 4) + z_swing

    # 17) to world
    # full rotation
    p_ref_rel_w = torch.bmm(
        R_WB,
        p_ref_B_final.transpose(1, 2)
    ).transpose(1, 2)                                                      # (N,4,3)

    p_ref_W = robot.data.root_pos_w.unsqueeze(1) + p_ref_rel_w

    # 18) cache for rewards / debug
    env.beta_contact_ref = (c_ref > 0.5)
    env.beta_foot_pos_ref_w = p_ref_W
    env.beta_p_ref_rel_w = p_ref_rel_w

    env.beta_p_ref_B = p_ref_B_final.clone()
    env.beta_hip_base_B = hip_base.clone()

    beta = torch.zeros((num_envs, num_legs, 4), device=device)
    beta[..., 0] = c_ref
    beta[..., 1:] = p_ref_rel_w
    beta = beta.view(num_envs, -1)

    # step cache
    env._beta_last_step = cur_step
    env._beta_last_value = beta
    return beta


def quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Quaternion (w,x,y,z) -> full rotation matrix R_WB."""
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


def quat_wxyz_to_yaw_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    Quaternion (w,x,y,z) -> yaw-only rotation matrix.
    Only the rotation around the z-axis is retained, and the effect of roll/pitch on the foot reference is removed
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    yaw = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z)
    )  # (N,)

    cy = torch.cos(yaw)
    sy = torch.sin(yaw)

    N = q.shape[0]
    R = torch.zeros((N, 3, 3), device=q.device, dtype=q.dtype)

    R[:, 0, 0] = cy
    R[:, 0, 1] = -sy
    R[:, 0, 2] = 0.0

    R[:, 1, 0] = sy
    R[:, 1, 1] = cy
    R[:, 1, 2] = 0.0

    R[:, 2, 0] = 0.0
    R[:, 2, 1] = 0.0
    R[:, 2, 2] = 1.0

    return R


def get_go2_hip_positions_B(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Get the fixed hip joint reference positions (Body frame), order: FR, FL, RR, RL."""
    return torch.tensor(
        [
            [ 0.183, -0.122, 0.0],   # FR
            [ 0.183,  0.122, 0.0],   # FL
            [-0.183, -0.122, 0.0],   # RR
            [-0.183,  0.122, 0.0],   # RL
        ],
        device=device,
        dtype=dtype,
    )


def robot_state_s(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gait_command_name: str = "gait_id",
    gait_table: dict | None = None,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]

    logic_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    logic_leg_names = ["FR", "FL", "RR", "RL"]

    # =========================================================
    # 1) joint ids: Forced rearrangement into logical order [FR, FL, RR, RL]
    # =========================================================
    name_to_joint_id = {n: i for i, n in enumerate(robot.data.joint_names)}
    asset_cfg.joint_ids = [name_to_joint_id[n] for n in logic_joint_names]

    # joint states
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]          # (N,12)
    joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]          # (N,12)
    joint_torques = robot.data.applied_torque[:, asset_cfg.joint_ids]  # (N,12)

    # base states
    projected_gravity = robot.data.projected_gravity_b                # (N,3)
    ang_vel = robot.data.root_ang_vel_b                               # (N,3)
    lin_vel = robot.data.root_lin_vel_b                               # (N,3)
    base_height = robot.data.root_pos_w[:, 2:3]                       # (N,1)

    # =========================================================
    # 2) body ids
    # =========================================================
    name_to_body_id = {n: i for i, n in enumerate(robot.data.body_names)}
    foot_body_ids = [name_to_body_id[f"{leg}_foot"] for leg in logic_leg_names]
    hip_body_ids = [name_to_body_id[f"{leg}_hip"] for leg in logic_leg_names]

    # =========================================================
    # 3) contact sensor ids
    # =========================================================
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    if not hasattr(env, "_contact_name_to_id"):
        env._contact_name_to_id = {n: i for i, n in enumerate(contact_sensor.body_names)}
    contact_name_to_id = env._contact_name_to_id

    contact_foot_ids = [contact_name_to_id[f"{leg}_foot"] for leg in logic_leg_names]

    foot_contact = (
        contact_sensor.data.current_contact_time[:, contact_foot_ids] > 0.0
    ).float()  # (N,4)

    # calf contact ids
    if not hasattr(env, "_contact_calf_ids"):
        env._contact_calf_ids = {
            "FR": contact_name_to_id["FR_calf"],
            "FL": contact_name_to_id["FL_calf"],
            "RR": contact_name_to_id["RR_calf"],
            "RL": contact_name_to_id["RL_calf"],
        }

    # gait one-hot
    current_gait_command = env.command_manager.get_command(gait_command_name)
    current_gait_id = current_gait_command[:, 0].long()
    gait_obs = torch.nn.functional.one_hot(current_gait_id, num_classes=8).float()

    # =========================================================
    # 4) Read the beta cache; if this step hasn't been performed yet, perform the calculation again.
    # =========================================================
    if (
        not hasattr(env, "_beta_last_step")
        or env._beta_last_step != int(env.common_step_counter)
        or not hasattr(env, "_beta_last_value")
    ):
        gait_info = beta_l_raibert(
            env,
            gait_table=gait_table,
            gait_command_name=gait_command_name,
        )
    else:
        gait_info = env._beta_last_value

    # =========================================================
    # 5) Unified pre-calculation
    # =========================================================
    root_pos_w = robot.data.root_pos_w.unsqueeze(1)              # (N,1,3)
    real_foot_pos_w = robot.data.body_pos_w[:, foot_body_ids, :]  # (N,4,3)
    rel_pos_w = real_foot_pos_w - root_pos_w                     # (N,4,3)

    quat_WB = robot.data.root_quat_w
    R_WB = quat_wxyz_to_rotmat(quat_WB)
    real_foot_pos_b = torch.bmm(
        R_WB.transpose(1, 2),
        rel_pos_w.transpose(1, 2)
    ).transpose(1, 2)                                            # (N,4,3)

    forces = contact_sensor.data.net_forces_w                    # (N, n_contact_bodies, 3)
    force_norm = torch.norm(forces, dim=-1)                      # (N, n_contact_bodies)

    # =========================================================
    # First 1000 steps: Print each of the four legs individually for z-tracking.
    # =========================================================
    if env.common_step_counter < 1000 and hasattr(env, "beta_p_ref_B") and hasattr(env, "beta_contact_ref"):
        print(f"\n[STEP {env.common_step_counter}] FOOT Z (BODY + WORLD)")
        print(f"{'Leg':<5} | {'real_z_B':<8} | {'ref_z_B':<8} | {'real_z_W':<8} | {'dz_B':<8} | {'ref_c':<5} | {'real_c':<6}")
        print("-" * 80)

        # --- world frame foot pos ---
        foot_pos_w = robot.data.body_pos_w[:, foot_body_ids, :]   # (N,4,3)

        for i, leg in enumerate(["FR", "FL", "RR", "RL"]):
            # --- BODY frame ---
            real_z_B = real_foot_pos_b[0, i, 2].item()
            ref_z_B = env.beta_p_ref_B[0, i, 2].item()
            dz_B = real_z_B - ref_z_B

            # --- WORLD frame ---
            real_z_W = foot_pos_w[0, i, 2].item()

            ref_c = int(env.beta_contact_ref[0, i].item())
            real_c = int(foot_contact[0, i].item())

            print(f"{leg:<5} | {real_z_B:+.3f}  | {ref_z_B:+.3f}  | {real_z_W:+.3f}  | {dz_B:+.3f}  | {ref_c:<5} | {real_c:<6}")

        # -------------------------
        # REAR JOINT DEBUG（只打印一次，不要放在for里面）
        # -------------------------
        print("\n[REAR JOINT CHECK]")
        rr_pos = joint_pos[0, 6:9].detach().cpu().numpy().round(3)
        rl_pos = joint_pos[0, 9:12].detach().cpu().numpy().round(3)
        rr_vel = joint_vel[0, 6:9].detach().cpu().numpy().round(3)
        rl_vel = joint_vel[0, 9:12].detach().cpu().numpy().round(3)

        print("RR joint_pos [hip, thigh, calf]:", rr_pos)
        print("RL joint_pos [hip, thigh, calf]:", rl_pos)
        print("RR joint_vel [hip, thigh, calf]:", rr_vel)
        print("RL joint_vel [hip, thigh, calf]:", rl_vel)

        # -------------------------
        # 姿态 & 后腿高度
        # -------------------------
        roll_indicator = projected_gravity[0, 1].item()

        print(
            f"[STEP {env.common_step_counter}] "
            f"grav_y={roll_indicator:+.3f} | "
            f"RR_z_B={real_foot_pos_b[0,2,2].item():+.3f} ref={env.beta_p_ref_B[0,2,2].item():+.3f} | "
            f"RR_z_W={foot_pos_w[0,2,2].item():+.3f} | "
            f"RL_z_B={real_foot_pos_b[0,3,2].item():+.3f} ref={env.beta_p_ref_B[0,3,2].item():+.3f} | "
            f"RL_z_W={foot_pos_w[0,3,2].item():+.3f}"
        )

    # =========================================================
    # 6) debug：every 200 steps
    # =========================================================
    if env.common_step_counter % 200 == 0:
        print("\n" + "🏠" * 10 + f" [DEBUG STEP: {env.common_step_counter}] 机身系校验 " + "🏠" * 10)
        print(f"Base Height: {base_height[0].item():.3f} | Gravity_B: {projected_gravity[0].cpu().numpy().round(2)}")
        print(f"Lin Vel (v_B):     {lin_vel[0].cpu().numpy().round(3)}")
        print("[contact_foot_ids (FR,FL,RR,RL)]", contact_foot_ids)
        print("[contact_foot_names]", [contact_sensor.body_names[i] for i in contact_foot_ids])
        print(f"Joint Names in Sim: {robot.data.joint_names}")
        print(f"Body Name Order:   {robot.data.body_names}")

        is_contact = contact_sensor.data.current_contact_time[:, contact_foot_ids] > 0.0
        print("[is_contact bool]", is_contact[0].tolist())
        print("[sensor_cfg.body_ids]", sensor_cfg.body_ids)
        print("[foot_body_ids(body_names)]", foot_body_ids)

        if hasattr(env, "beta_contact_ref"):
            print("[ref_contact bool]", env.beta_contact_ref[0].tolist())

        print("\n[Resolved Joint Order]")
        print([robot.data.joint_names[i] for i in asset_cfg.joint_ids])

        print("\n[Resolved Body IDs]")
        print("Hip body ids:", list(zip(logic_leg_names, hip_body_ids)))
        print("Foot body ids:", list(zip(logic_leg_names, foot_body_ids)))

        # -----------------------------------------------------
        # Four-legged Z-tracking
        # -----------------------------------------------------
        print("\n[FOOT Z TRACKING DEBUG]")
        print(f"{'Leg':<5} | {'real_z':<8} | {'ref_z':<8} | {'dz':<8} | {'ref_c':<5} | {'real_c':<6}")
        print("-" * 60)
        for i, leg in enumerate(logic_leg_names):
            real_z = real_foot_pos_b[0, i, 2].item()
            ref_z = env.beta_p_ref_B[0, i, 2].item() if hasattr(env, "beta_p_ref_B") else float("nan")
            dz = real_z - ref_z
            ref_c = int(env.beta_contact_ref[0, i].item()) if hasattr(env, "beta_contact_ref") else -1
            real_c = int(foot_contact[0, i].item())
            print(f"{leg:<5} | {real_z:+.3f}  | {ref_z:+.3f}  | {dz:+.3f}  | {ref_c:<5} | {real_c:<6}")

        # -----------------------------------------------------
        # Machine system pin details
        # -----------------------------------------------------
        print("\n[Body-Frame Foot Debug]")
        print(f"{'Leg':<5} | {'Hip_Base_B (x,y,z)':<24} | {'Ref_Foot_B (x,y,z)':<24} | {'Real_Foot_B (x,y,z)':<24} | {'Err_B (x,y,z)':<24}")
        print("-" * 130)

        for i, leg in enumerate(logic_leg_names):
            hip_b = "N/A"
            ref_b = "N/A"
            err_b = "N/A"

            if hasattr(env, "beta_hip_base_B"):
                hip_b = env.beta_hip_base_B[0, i].detach().cpu().numpy().round(3)

            if hasattr(env, "beta_p_ref_B"):
                ref_b = env.beta_p_ref_B[0, i].detach().cpu().numpy().round(3)
                err_b = (real_foot_pos_b[0, i] - env.beta_p_ref_B[0, i]).detach().cpu().numpy().round(3)

            real_b = real_foot_pos_b[0, i].detach().cpu().numpy().round(3)

            print(f"{leg:<5} | {str(hip_b):<24} | {str(ref_b):<24} | {str(real_b):<24} | {str(err_b):<24}")

        # -----------------------------------------------------
        # calf contact debug
        # -----------------------------------------------------
        print("\n[CALF CONTACT DEBUG]")
        for leg in logic_leg_names:
            cid = env._contact_calf_ids[leg]
            contact_flag = (force_norm[0, cid] > 1.0).item()
            force_val = force_norm[0, cid].item()
            print(f"{leg} calf contact: {contact_flag} | force: {force_val:.2f}")

        # -----------------------------------------------------
        # rear leg height check
        # -----------------------------------------------------
        print("\n[REAR LEG HEIGHT CHECK]")
        for i, leg in enumerate(logic_leg_names):
            if leg in ["RR", "RL"]:
                real_z = real_foot_pos_b[0, i, 2].item()
                ref_z = env.beta_p_ref_B[0, i, 2].item() if hasattr(env, "beta_p_ref_B") else float("nan")
                dz = real_z - ref_z
                print(f"{leg}: real_z={real_z:.3f} | ref_z={ref_z:.3f} | dz={dz:.3f}")

        # -----------------------------------------------------
        # RL touchdown delay check
        # -----------------------------------------------------
        print("\n[RL TOUCHDOWN DELAY CHECK]")
        i = 3  # RL
        rl_ref_c = bool(env.beta_contact_ref[0, i].item()) if hasattr(env, "beta_contact_ref") else False
        rl_real_c = bool(foot_contact[0, i].item())
        rl_real_b = real_foot_pos_b[0, i].detach().cpu().numpy().round(3)
        rl_ref_b = env.beta_p_ref_B[0, i].detach().cpu().numpy().round(3) if hasattr(env, "beta_p_ref_B") else "N/A"
        print("RL ref_contact:", rl_ref_c)
        print("RL real_contact:", rl_real_c)
        print("RL ref_foot_B:", rl_ref_b)
        print("RL real_foot_B:", rl_real_b)
        print("RL joint_pos [hip, thigh, calf]:", joint_pos[0, 9:12].detach().cpu().numpy().round(3))
        print("RL joint_vel [hip, thigh, calf]:", joint_vel[0, 9:12].detach().cpu().numpy().round(3))
        print("RL has _raibert_prev_c:", hasattr(env, "_raibert_prev_c"))

    # =========================================================
    # 7) Real-time capture of critical anomalies: those that should have been written but weren't.
    # =========================================================
    if current_gait_id[0].item() == 0 and hasattr(env, "beta_contact_ref"):
        for leg, i in zip(logic_leg_names, [0, 1, 2, 3]):
            ref_c = bool(env.beta_contact_ref[0, i].item())
            real_c = bool(foot_contact[0, i].item())

            if ref_c and not real_c:
                real_z = real_foot_pos_b[0, i, 2].item()
                ref_z = env.beta_p_ref_B[0, i, 2].item() if hasattr(env, "beta_p_ref_B") else float("nan")
                print(f"\n⚠️ [{leg} SHOULD TOUCH BUT NOT TOUCHING]")
                print(
                    f"{leg} | real_z={real_z:+.3f} | ref_z={ref_z:+.3f} | dz={real_z-ref_z:+.3f} "
                    f"| ref_c=1 real_c=0"
                )

    # =========================================================
    # 8) Bound knee-to-ground moment capture: checked every 20 steps
    # =========================================================
    if env.common_step_counter % 20 == 0 and current_gait_id[0].item() == 0:
        for leg in ["RR", "RL"]:
            cid = env._contact_calf_ids[leg]
            if force_norm[0, cid] > 5.0:
                print("\n🔥 [KNEE CONTACT DETECTED] ")
                print(f"Leg: {leg}")
                print(f"Calf force: {force_norm[0, cid].item():.2f}")

                i = logic_leg_names.index(leg)
                foot_contact_cid = contact_foot_ids[i]
                foot_force = force_norm[0, foot_contact_cid].item()
                print(f"Foot force: {foot_force:.2f}")
                print(f"Foot height (z): {real_foot_pos_b[0, i, 2].item():.3f}")

                if hasattr(env, "beta_p_ref_B"):
                    print(f"Ref foot z: {env.beta_p_ref_B[0, i, 2].item():.3f}")
                    print(f"dz(real-ref): {(real_foot_pos_b[0, i, 2] - env.beta_p_ref_B[0, i, 2]).item():.3f}")

    # =========================================================
    # 9) unpack gait_info
    # =========================================================
    g_reshaped = gait_info.view(-1, 4, 4)
    # vel_cmd = env.command_manager.get_command("base_velocity")
    vel_cmd = gait_conditioned_base_velocity(
        env,
        command_name="base_velocity",
        gait_command_name=gait_command_name,
        stand_gait_id=6,
    )

    desFeetContact = g_reshaped[:, :, 0]
    refFootX = g_reshaped[:, :, 1]
    refFootY = g_reshaped[:, :, 2]
    refFootZ = g_reshaped[:, :, 3]

    # =========================================================
    # 10) final obs
    # =========================================================
    obs_69 = torch.cat([
        projected_gravity,  # 3
        joint_pos,          # 12
        ang_vel,            # 3
        joint_vel,          # 12
        lin_vel,            # 3
        vel_cmd,            # 3
        joint_torques,      # 12
        foot_contact,       # 4
        base_height,        # 1
        desFeetContact,     # 4
        refFootZ,           # 4
        refFootX,           # 4
        refFootY            # 4
    ], dim=-1)

    return obs_69
