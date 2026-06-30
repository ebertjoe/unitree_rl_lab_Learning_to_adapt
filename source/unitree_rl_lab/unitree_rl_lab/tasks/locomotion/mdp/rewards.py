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
from .observations import gait_conditioned_base_velocity

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
    clamp_jerk: float | None = 5.0
) -> torch.Tensor:
    """Efficiency penalty (aligned with paper Eq. 6, with normalization)."""
    asset: Articulation = env.scene[asset_cfg.name]
    num_dof = len(asset_cfg.joint_ids)

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]

    # get Action
    q_star = None
    term = env.action_manager.get_term("JointPositionAction")
    q_star = term.processed_actions

    if q_star.shape[-1] != num_dof:
        q_star = q_star[:, asset_cfg.joint_ids]

    # Initialize Buffers
    if not hasattr(env, "_paper_eta_qvel_prev"):
        env._paper_eta_qvel_prev = qvel.clone()
        env._paper_eta_qvel_prev2 = qvel.clone()
        env._paper_eta_action_prev = q_star.clone()

    # Handling Resets
    if hasattr(env, "reset_buf"):
        ids = torch.where(env.reset_buf)[0]
        if len(ids) > 0:
            env._paper_eta_qvel_prev[ids] = qvel[ids]
            env._paper_eta_qvel_prev2[ids] = qvel[ids]
            env._paper_eta_action_prev[ids] = q_star[ids]

    # --- Calculate ---

    # 1. Jerk: Using mean instead of sum can offset the numerical amplification caused by the increase in degrees of freedom.
    jerk = qvel - 2.0 * env._paper_eta_qvel_prev + env._paper_eta_qvel_prev2
    if use_dt_scaling:
        dt = float(env.step_dt)
        jerk = jerk / (dt * dt)
    if clamp_jerk is not None:
        jerk = torch.clamp(jerk, -clamp_jerk, clamp_jerk)
    # Using mean() ensures that the numerical magnitude remains consistent regardless of the number of joints in the robot.
    jerk_term = torch.mean(jerk * jerk, dim=-1)

    # 2. Torque: Using mean()
    tau_term = torch.mean(qfrc * qfrc, dim=-1)

    # 3. Action rate: Action difference
    dact = q_star - env._paper_eta_action_prev
    # dact_term = torch.mean(dact * dact, dim=-1)
    dact_term = torch.sum(dact * dact, dim=-1)

    # The sum of formulas in the paper
    total_eta = 0.01 * jerk_term + 0.0001 * tau_term + dact_term
    # total_eta = dact_term

    # --- debug ---
    if env.common_step_counter % 200 == 0:
        print("\n" + "-" * 30)
        print(f"[ETA DETAIL | Step {env.common_step_counter}]")
        print(f"  - Jerk (mean sq):   {jerk_term[0].item():.4f}")
        print(f"  - Torque (mean sq): {tau_term[0].item():.4f}")
        print(f"  - Action (mean sq): {dact_term[0].item():.4f}")
        print(f"  >> Total Raw:       {total_eta[0].item():.4f}")
        print(f"  >> Weighted Score (*-1.5): {total_eta[0].item() * -1.5:.4f}")
        print("Using processed_actions:", hasattr(term, "processed_actions"))
        print("q_star shape:", q_star.shape)
        print("q_star sample:", q_star[0])
        print("active_terms:", env.action_manager.active_terms)
        print("-" * 30)

    # Update Buffers
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

    # v_cmd = env.command_manager.get_command(command_name)  # (N,3) [vx_cmd, vy_cmd, wz_cmd]
    v_cmd = gait_conditioned_base_velocity(
        env,
        command_name=command_name,
        gait_command_name="gait_id",
        stand_gait_id=6,
    )

    v_xy = asset.data.root_lin_vel_b[:, :2]                # (N,2) body frame
    w_z = asset.data.root_ang_vel_b[:, 2:3] * wz_scale     # (N,1)
    v = torch.cat([v_xy, w_z], dim=-1)                     # (N,3)

    v_cmd2 = v_cmd.clone()
    v_cmd2[:, 2] = v_cmd2[:, 2] * wz_scale

    err2 = torch.sum((v - v_cmd2) ** 2, dim=-1)            # ||v - vcmd||^2
    reward = psi(err2 * 4.0)

    # Debug Printing (w_vcmd = 50.0)
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
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # 0) Read-only cache, does not actively advance to beta state.
    if not hasattr(env, "beta_p_ref_rel_w") or not hasattr(env, "beta_contact_ref"):
        return torch.zeros(env.num_envs, device=env.device)

    ref_rel_pos_w = env.beta_p_ref_rel_w                  # (N,4,3)
    contact_ref_bool = env.beta_contact_ref.bool()        # (N,4)

    # 1) articulation foot ids in LOGIC order [FR, FL, RR, RL]
    logic_leg_names = ["FR", "FL", "RR", "RL"]
    if not hasattr(env, "_rf_foot_body_ids"):
        name_to_body_id = {n: i for i, n in enumerate(asset.data.body_names)}
        env._rf_foot_body_ids = [name_to_body_id[f"{leg}_foot"] for leg in logic_leg_names]
    foot_body_ids = env._rf_foot_body_ids

    foot_pos_w = asset.data.body_pos_w[:, foot_body_ids, :]   # (N,4,3)
    root_pos_w = asset.data.root_pos_w.unsqueeze(1)           # (N,1,3)
    real_rel_pos_w = foot_pos_w - root_pos_w                  # (N,4,3)

    # 2) contact ids from contact sensor indexing
    if not hasattr(env, "_rf_contact_foot_ids"):
        contact_name_to_id = {n: i for i, n in enumerate(contact_sensor.body_names)}
        env._rf_contact_foot_ids = [contact_name_to_id[f"{leg}_foot"] for leg in logic_leg_names]
    contact_foot_ids = env._rf_contact_foot_ids

    is_contact = contact_sensor.data.current_contact_time[:, contact_foot_ids] > 0.0  # (N,4)

    # 3) contact mismatch cost
    c_err = (is_contact ^ contact_ref_bool).float()
    c_cost = c_err.mean(dim=-1) * 1.2

    # 4) only penalize swing legs for foot placement tracking
    swing_mask = (~contact_ref_bool).float()   # (N,4)
    num_swing = swing_mask.sum(dim=-1)         # (N,)

    # XY distance
    pos_dist_xy = torch.norm(real_rel_pos_w[..., :2] - ref_rel_pos_w[..., :2], dim=-1)  # (N,4)
    sigma_xy = 0.10
    pos_cost_xy = 1.0 - torch.exp(-(pos_dist_xy ** 2) / (2.0 * sigma_xy ** 2))          # (N,4)

    # Z distance
    pos_dist_z = torch.abs(real_rel_pos_w[..., 2] - ref_rel_pos_w[..., 2])               # (N,4)
    sigma_z = 0.08
    pos_cost_z = 1.0 - torch.exp(-(pos_dist_z ** 2) / (2.0 * sigma_z ** 2))              # (N,4)

    # combine XY + Z with different weights (you can tune these)
    foot_cost = pos_cost_xy + 1.0 * pos_cost_z                                            # (N,4)

    p_cost = torch.where(
        num_swing > 0.0,
        (foot_cost * swing_mask).sum(dim=-1) / (num_swing + 1e-6),
        torch.zeros_like(num_swing),
    )

    total_cost = c_cost + p_cost

    if env.common_step_counter == 0:
        print("[r_f DEBUG] contact_sensor.body_names =", contact_sensor.body_names)
        print("[r_f DEBUG] contact_foot_ids(FR,FL,RR,RL) =", contact_foot_ids)
        print("[r_f DEBUG] contact_foot_names =", [contact_sensor.body_names[i] for i in contact_foot_ids])
        print("[r_f DEBUG] foot_body_ids (asset.data.body_names) =", foot_body_ids)

    if env.common_step_counter % 200 == 0:
        mean_xy = ((pos_dist_xy * swing_mask).sum(dim=-1) / (num_swing + 1e-6))[0].item()
        mean_z = ((pos_dist_z * swing_mask).sum(dim=-1) / (num_swing + 1e-6))[0].item()
        print(
            f"[REWARD GAIT] c_cost={c_cost[0].item():.4f} "
            f"p_cost={p_cost[0].item():.4f} "
            f"total={total_cost[0].item():.4f} | "
            f"mean_swing_xy_err={mean_xy:.4f}m | "
            f"mean_swing_z_err={mean_z:.4f}m"
        )

    return total_cost

# --------------------------------------------------------
# r_stab (Eq. 9): stability
# -----------------------------------------------------------------------------


def r_stab(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    desired_gravity_b: list[float] | torch.Tensor,
    gait_table: dict,
    gait_command_name: str = "gait_id",
    hip_joint_ids: list[int] = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    device = env.device

    # -------------------------
    # A) Dynamically obtain vectorized gait parameters
    # -------------------------
    # 1. Get the current gait_id (N,) for all environments
    gait_ids = env.command_manager.get_command(gait_command_name).long().flatten()

    # 2.  Ensure that the gait tensor is cached in the environment
    if not hasattr(env, "_gait_table_tensors"):
        keys = sorted([int(k) for k in gait_table.keys()])
        env._gait_znoms = torch.tensor([abs(gait_table[str(k)]["z_nom"]) for k in keys], device=device).float()
        env._gait_table_tensors = True

    # 3. Extract the nominal height (N,) corresponding to all current environments
    z_nom_t = env._gait_znoms[gait_ids]

    # stance mask：LOGIC order [FR,FL,RR,RL]
    stance = env.beta_contact_ref.bool()  # (N,4)

    # --- use articulation body_names to find foot ids，make sure [FR,FL,RR,RL] ---
    logic_leg_names = ["FR", "FL", "RR", "RL"]
    if not hasattr(env, "_rstab_foot_body_ids"):
        name_to_body_id = {n: i for i, n in enumerate(asset.data.body_names)}
        env._rstab_foot_body_ids = [name_to_body_id[f"{leg}_foot"] for leg in logic_leg_names]
    foot_body_ids = env._rstab_foot_body_ids

    # 1) Get horizontal speed (XY) - LOGIC order
    foot_vel_xy = asset.data.body_lin_vel_w[:, foot_body_ids, :2]  # (N,4,2)

    # 2) velocity squared
    slip_each = torch.sum(foot_vel_xy * foot_vel_xy, dim=-1)  # (N,4)

    # 3) Only penalize feet in the expected stance (LOGIC order aligned)
    slip_term = torch.sum(slip_each * stance.float(), dim=-1) * 2.0

    # angular velocity: roll/pitch
    omega_xy = asset.data.root_ang_vel_b[:, :2]
    omega_term = torch.sum(omega_xy * omega_xy, dim=-1)

    # desired_gravity_b
    if not isinstance(desired_gravity_b, torch.Tensor):
        desired_gravity_b = torch.tensor(desired_gravity_b, device=env.device, dtype=asset.data.projected_gravity_b.dtype)
    desired_gravity_b = desired_gravity_b.view(1, 3).repeat(env.num_envs, 1)  # (N,3)

    # Posture Item
    alphaR_B = asset.data.projected_gravity_b
    orient_err2 = torch.sum((alphaR_B - desired_gravity_b) ** 2, dim=-1)
    # orient_term = psi(orient_err2)
    orient_term = 1.0 - psi(orient_err2)

    # height item
    zB = asset.data.root_pos_w[:, 2]
    height_term = 1 - psi((zB - z_nom_t) ** 2)

    # -------------------------------------------------
    # Flight-phase gating (NEW)
    # -------------------------------------------------
    # When the Raibert reference schedules ALL FOUR legs in swing
    # simultaneously (a genuine flight/aerial phase, e.g. mid-cycle in
    # bound/pronk/hop), the body is expected to be in ballistic motion:
    # height and orientation will naturally deviate from the stance
    # nominal values. Without gating, height_term/orient_term spike during
    # exactly the window the policy is supposed to risk, making flight
    # phases reward-negative and teaching the policy to keep a foot down
    # at all times (suppressing aerial gaits like bound entirely).
    #
    # slip_term already does NOT need this gating: it's masked by `stance`
    # and naturally goes to zero when no legs are in stance.
    in_flight_ref = (~stance).all(dim=-1).float()  # (N,) 1.0 if all 4 legs scheduled swing

    height_term = (1.0 - in_flight_ref) * height_term
    orient_term = (1.0 - in_flight_ref) * orient_term

    # Hip joint
    qhip = asset.data.joint_pos[:, hip_joint_ids]
    hip_term = torch.sum(qhip * qhip, dim=-1)

    # -------------------------------------------------
    # stand-only extra height penalty
    # only active when gait_id == 6
    # -------------------------------------------------
    stand_mask = (gait_ids == 6).float()
    stand_target_height = 0.32

    # stand_height_err = (zB - stand_target_height) ** 2
    stand_height_err = torch.abs(zB - stand_target_height)
    stand_height_term = stand_mask * stand_height_err * 3.0

    # return
    total_stab_error = slip_term + omega_term + orient_term + height_term + hip_term + stand_height_term

    # debug (w_stab = -1.0)
    if env.common_step_counter % 200 == 0:
        print(f"[STAB DETAIL | Step {env.common_step_counter}]")
        print(f"  1. Slip: {slip_term[0].item():.4f}")
        print(f"  2. Omega: {omega_term[0].item():.4f}")
        print(f"  3. Orient:  {orient_term[0].item():.4f}")
        print(f"  4. Height:  {height_term[0].item():.4f} (zB={zB[0].item():.3f})")
        print(f"  5. Hip:   {hip_term[0].item():.4f}")
        print(f"  6. In-flight (ref, env 0): {bool(in_flight_ref[0].item())}")
        print(f"  >> Total Raw Error: {total_stab_error[0].item():.4f}")
        print(f"  >> Weighted Score (*-1.0): {total_stab_error[0].item() * -1.0:.4f}")
        print(f"      -> Detail: zB={zB[0].item():.3f}, Height_Err={height_term[0].item():.3f}, Orient_Err={orient_term[0].item():.3f}")

    return total_stab_error

def foot_trajectory_tracking(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize deviation of actual foot positions from Raibert reference
    positions, but only during swing phase (desFeetContact == 0).
    """
    # get actual foot positions in world frame
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene[sensor_cfg.name]

    # foot positions: get from body state
    foot_body_ids = asset_cfg.body_ids  
    foot_pos_w = asset.data.body_pos_w[:, foot_body_ids, :] 

    # reference positions from Raibert (cached by observations.py)
    ref_pos = env.raibert_ref_foot_pos  
    des_contact = env.raibert_des_contact  

    # only penalize during swing phase
    swing_mask = (des_contact < 0.5).float() 

    # L2 distance between actual and reference foot position
    dist = torch.norm(foot_pos_w - ref_pos, dim=-1) 

    # apply swing mask and sum over legs
    dist = torch.clamp(dist, max=0.5)
    return (dist * swing_mask).sum(dim=-1)

def gait_conditioned_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize left-right joint asymmetry for symmetric gaits only."""
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # only enforce for symmetric gaits: trot(1), bound(0), hop(2), pronk(4), run(7)
    gait_id = env.command_manager.get_command("gait_id").squeeze(-1)
    symmetric_mask = torch.isin(
        gait_id,
        torch.tensor([0, 2, 4], device=env.device)
    ).float() 

    joint_pos = asset.data.joint_pos  
    right = joint_pos[:, [0, 1, 2, 6, 7,  8]]  
    left  = joint_pos[:, [3, 4, 5, 9, 10, 11]] 

    symmetry_error = torch.sum((left - right) ** 2, dim=-1)
    if torch.any(torch.isnan(symmetry_error)) or torch.any(torch.isinf(symmetry_error)):
        print("NaN/Inf in gait_conditioned_symmetry!")
        return torch.zeros(env.num_envs, device=env.device)
    return torch.clamp(symmetry_error * symmetric_mask, max=2.0)