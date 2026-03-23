# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL.

Launch Isaac Sim Simulator first.
"""

import argparse
import os
import time
from importlib.metadata import version

import gymnasium as gym
import matplotlib.pyplot as plt
import torch

from isaaclab.app import AppLauncher
from isaaclab.utils.dict import print_dict

# local imports
import cli_args  # isort: skip


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# ---- custom logging args ----
parser.add_argument("--env_id", type=int, default=0, help="Which env index to log (default: 0).")
parser.add_argument("--leg_id", type=int, default=0, help="0:FL, 1:FR, 2:RL, 3:RR (default: 0).")
parser.add_argument("--log_steps", type=int, default=60, help="How many events to log (default: 60).")
parser.add_argument(
    "--max_sim_steps",
    type=int,
    default=6000,
    help="Max simulation steps before forced stop (default: 6000). Prevents running forever.",
)
parser.add_argument(
    "--plot_swing_traj",
    action="store_true",
    default=False,
    help="Also log and plot real foot trajectory in x-z during swing (textbook-like).",
)

# choose event type for p_ref logging
parser.add_argument(
    "--p_ref_event",
    type=str,
    default="touchdown",
    choices=["liftoff", "touchdown"],
    help="Log p_ref at liftoff (stance->swing) or touchdown (swing->stance). Touchdown is more footprint-like.",
)

# ground height for footprint plots
parser.add_argument(
    "--ground_z",
    type=float,
    default=0.0,
    help="Ground height used for footprint z in world frame (default: 0.0).",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# --------------------------------------------------------------------------------
# Imports that require SimulationApp
# --------------------------------------------------------------------------------
import omni.ui as ui
import carb
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------
def _auto_find_foot_indices(robot) -> list[int] | None:
    """Try to find 4 foot link indices by name heuristics (Articulation body indices)."""
    if not hasattr(robot.data, "body_names"):
        return None
    names = list(robot.data.body_names)

    cand = [i for i, n in enumerate(names) if "foot" in n.lower()]
    if len(cand) >= 4:
        return cand[:4]

    cand = [i for i, n in enumerate(names) if "ankle" in n.lower()]
    if len(cand) >= 4:
        return cand[:4]

    return None


def _yaw_from_quat_wxyz(q_wxyz: torch.Tensor) -> torch.Tensor:
    """Return yaw (rad) from quaternion wxyz (supports batched or single)."""
    w, x, y, z = q_wxyz[..., 0], q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate_xy_by_yaw(px: torch.Tensor, py: torch.Tensor, yaw: torch.Tensor):
    """Rotate (px, py) by yaw (about z)."""
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    wx = cy * px - sy * py
    wy = sy * px + cy * py
    return wx, wy


def _find_contact_sensor(env_unwrapped):
    """Try to find a contact sensor in env.scene.sensors.

    Returns: (sensor_obj, sensor_name) or (None, None)
    """
    try:
        if hasattr(env_unwrapped, "scene") and hasattr(env_unwrapped.scene, "sensors"):
            for name, s in env_unwrapped.scene.sensors.items():
                if not hasattr(s, "data"):
                    continue
                if hasattr(s.data, "current_contact_time") or hasattr(s.data, "net_forces_w"):
                    return s, name
    except Exception:
        pass
    return None, None


def _map_foot_to_contact_index(contact_sensor, foot_body_index: int) -> int | None:
    """Map articulation body index -> contact sensor column index if possible."""
    body_ids = None
    for attr in ("body_ids", "body_ids_w", "body_ids_env", "body_ids_local"):
        if hasattr(contact_sensor, attr):
            body_ids = getattr(contact_sensor, attr)
            break
    if body_ids is None and hasattr(contact_sensor, "cfg") and hasattr(contact_sensor.cfg, "body_ids"):
        body_ids = contact_sensor.cfg.body_ids

    if body_ids is None:
        return None

    if isinstance(body_ids, torch.Tensor):
        ids = body_ids.detach().cpu().tolist()
    else:
        ids = list(body_ids)

    try:
        return ids.index(int(foot_body_index))
    except ValueError:
        return None


def _default_hip_offsets_B(device, dtype):
    """Fallback hip offsets in body frame for Go2-like quadruped (approx).
    Order: FL, FR, RL, RR
    Units: meters.
    """
    return torch.tensor(
        [
            [0.19, 0.11, 0.0],   # FL
            [0.19, -0.11, 0.0],  # FR
            [-0.19, 0.11, 0.0],  # RL
            [-0.19, -0.11, 0.0], # RR
        ],
        device=device,
        dtype=dtype,
    )


def _debug_list_command_terms(cm):
    """Best-effort print of command terms across IsaacLab versions."""
    names = None
    for attr in ("_terms", "terms", "command_terms"):
        if hasattr(cm, attr):
            try:
                obj = getattr(cm, attr)
                if isinstance(obj, dict):
                    names = list(obj.keys())
                elif hasattr(obj, "keys"):
                    names = list(obj.keys())
                break
            except Exception:
                pass
    if names is None:
        names = sorted({n for n in dir(cm) if ("vel" in n.lower() or "command" in n.lower() or "term" in n.lower())})
    print("[DEBUG] Command terms (best-effort):", names)


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
def main():
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ================= DEBUG: print command cfg =================
    cm0 = env.unwrapped.command_manager
    try:
        print("[DEBUG] command terms:", cm0._terms.keys())
        base_term = cm0._terms["base_velocity"]
        print("[DEBUG] base_velocity cfg:", base_term.cfg)
        cmd0 = cm0.get_command("base_velocity")
        print("[DEBUG] initial base_velocity cmd (env0):", cmd0[0].detach().cpu().numpy())
    except Exception as e:
        print("[WARN] command cfg debug failed:", e)
    # ============================================================

    # important: import after SimulationApp started
    try:
        from unitree_rl_lab.tasks.locomotion.mdp.observations import beta_l_raibert, get_go2_hip_positions_B
        HAS_GO2_HIP = True
    except Exception as e:
        print("[WARN] Cannot import get_go2_hip_positions_B from observations. Will use fallback hips.", e)
        from unitree_rl_lab.tasks.locomotion.mdp.observations import beta_l_raibert
        HAS_GO2_HIP = False
        get_go2_hip_positions_B = None  # type: ignore

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # -------------------------
    # DEBUG: list command terms once
    # -------------------------
    cm = env.unwrapped.command_manager
    _debug_list_command_terms(cm)
    try:
        cmd0 = cm.get_command("base_velocity")
        print("[DEBUG] initial base_velocity cmd (env0):", cmd0[0].detach().cpu().numpy())
    except Exception as e:
        print("[WARN] cannot read base_velocity cmd:", e)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    # -------------------------
    # logging setup
    # -------------------------
    env_id = int(args_cli.env_id)
    leg_id = int(args_cli.leg_id)

    # event-based p_ref in world (yaw-only)
    t_log, x_log, y_log = [], [], []
    # also log body-frame relative dx/dy (textbook-like)
    t_rel, dx_rel_log, dy_rel_log = [], [], []

    # -------------------------
    # Raibert params (MUST be explicit here)
    # -------------------------
    raibert_period = 0.5
    raibert_offset = [0.0, 0.5, 0.5, 0.0]  # FL, FR, RL, RR
    raibert_threshold = 0.5
    raibert_kx = 0.03
    raibert_ky = 0.03

    prev_c = None
    event_count = 0
    sim_step_count = 0
    timestep_video = 0
    _printed_events = 0

    # robot handles
    robot = env.unwrapped.scene["robot"]
    foot_indices = _auto_find_foot_indices(robot)

    # contact sensor (preferred for swing segmentation)
    contact_sensor, contact_sensor_name = _find_contact_sensor(env.unwrapped)
    if contact_sensor is not None:
        print(f"[INFO] Using contact sensor: {contact_sensor_name}")
    else:
        print("[WARN] No ContactSensor found. Swing segmentation will fallback to height-based contact.")

    # swing trajectory segments (world x-z)
    swing_segments_xz = []
    _cur_x, _cur_z = [], []
    prev_contact_for_seg = None

    if args_cli.plot_swing_traj:
        if foot_indices is None:
            print("[WARN] Cannot auto-find foot indices. plot_swing_traj will be disabled.")
            args_cli.plot_swing_traj = False
        else:
            print(f"[INFO] Auto foot indices: {foot_indices} (leg_id uses this order)")

    # hip offsets for dx/dy (fallback; will override per-event if import works)
    hip_offsets_B_fallback = _default_hip_offsets_B(device=robot.data.root_pos_w.device, dtype=robot.data.root_pos_w.dtype)

    robot = env.unwrapped.scene["robot"]
    p0 = robot.data.root_pos_w[env_id]
    q0 = robot.data.root_quat_w[env_id]
    yaw0 = _yaw_from_quat_wxyz(q0)
    print(f"[INIT] base_pos_w=({p0[0].item():.3f},{p0[1].item():.3f},{p0[2].item():.3f}) yaw={yaw0.item():.3f} rad")

    # -------------------------
    # simulate environment
    # -------------------------
    # 定义 key_map (使用 carb 的键码)
    # 数字键 1-8 对应的 ASCII 码通常是 0x31-0x38
    # 修改后的 key_map 映射
    # 数字键 1-8 对应的常见十六进制偏移
    # ki = carb.input.KeyboardInput
    # key_map = {
    #   ki.KEY_1 if hasattr(ki, 'KEY_1') else 0x31: 0,
    #  ki.KEY_2 if hasattr(ki, 'KEY_2') else 0x32: 1,
    #  ki.KEY_3 if hasattr(ki, 'KEY_3') else 0x33: 2,
    #  ki.KEY_4 if hasattr(ki, 'KEY_4') else 0x34: 3,
    # ki.KEY_5 if hasattr(ki, 'KEY_5') else 0x35: 4,
    #  ki.KEY_6 if hasattr(ki, 'KEY_6') else 0x36: 5,
    #  ki.KEY_7 if hasattr(ki, 'KEY_7') else 0x37: 6,
    # ki.KEY_8 if hasattr(ki, 'KEY_8') else 0x38: 7,
    # }

    # --- [新增] UI Gait Controller 窗口 ---
    gait_names = ["Bound (0)", "Trot (1)", "Run (2)", "Stand (3)", "Pronk (4)", "Limp (5)", "Amble (6)", "Hop (7)"]
    
    # 创建 UI 窗口
    _gait_window = ui.Window("Gait Controller", width=300, height=200)
    with _gait_window.frame:
        with ui.VStack(spacing=8, padding=10):
            ui.Label("Click to Switch Gait", alignment=ui.Alignment.CENTER, style={"font_size": 18, "color": 0xFF00FFFF})
            
            def on_ui_click(gait_id):
                # 强制修改所有环境的指令值
                env.unwrapped.command_manager._terms["gait_id"].value_command[:] = float(gait_id)
                print(f"\033[93m[UI 切换] 已强制设置全场步态为: {gait_id} ({gait_names[gait_id]})\033[0m")

            # 布局按钮（两列分布）
            for i in range(0, 8, 2):
                with ui.HStack(spacing=5):
                    ui.Button(gait_names[i], clicked_fn=lambda idx=i: on_ui_click(idx), height=40)
                    ui.Button(gait_names[i+1], clicked_fn=lambda idx=i+1: on_ui_click(idx), height=40)
            
            ui.Spacer(height=10)
            ui.Label("Note: All envs will switch simultaneously.", style={"font_size": 10, "color": 0x99FFFFFF})

    gait_table = env_cfg.observations.policy.gait_beta.params["gait_table"]
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            sim_step_count += 1
            gait_params = env.unwrapped.cfg.observations.policy.gait_beta.params

            # --- 新增：获取并打印步态 ID ---
            # 从 CommandManager 获取当前所有环境的 gait_id
            gait_ids = env.unwrapped.command_manager.get_command("gait_id")
            
            if sim_step_count % 100 == 0:   # 每100步打印一次
                unique_ids = torch.unique(gait_ids)
                print("\033[93m当前所有 env 的 gait:", unique_ids.cpu().numpy(), "\033[0m")
           
            if sim_step_count % 50 == 0:
                current_id = int(gait_ids[env_id].item())
                # 步态名称映射（根据你的 GAIT_CONFIGS）
                gait_map = {
                    "0": "Bound", "1": "Trot", "2": "Run", "3": "Stand", 
                    "4": "Pronk", "5": "Limp", "6": "Amble", "7": "Hop"
                }
                name = gait_map.get(str(current_id), "Unknown")
                print(f"\033[92m>>> [ENV {env_id}] 当前步态 ID: {current_id} ({name}) \033[0m")

            # --------- compute beta_l ----------
            beta = beta_l_raibert(
                env.unwrapped,
                gait_command_name="gait_id",
                gait_table=gait_table,
                command_name="base_velocity",
            )
            beta_ = beta.view(env.unwrapped.num_envs, 4, 4)

            c = float(beta_[env_id, leg_id, 0].item())
            px = float(beta_[env_id, leg_id, 1].item())
            py = float(beta_[env_id, leg_id, 2].item())

            # --------- event detection ----------
            if prev_c is None:
                event = False  # first sample: don't trigger
            else:
                if args_cli.p_ref_event == "liftoff":
                    event = (prev_c > 0.5) and (c < 0.5)      # stance -> swing
                else:
                    event = (prev_c < 0.5) and (c > 0.5)      # swing -> stance
            prev_c = c

            # time
            t = float(env.unwrapped.episode_length_buf[env_id].item()) * float(env.unwrapped.step_dt)

            # -------------------------
            # Event-based βL verification + printing
            # -------------------------
            if event:
                # 获取当前环境真实的 gait_id
                current_gait_id_idx = str(int(env.unwrapped.command_manager.get_command("gait_id")[env_id].item()))
                current_cfg = gait_table[current_gait_id_idx]
        
                # 动态赋值给验证逻辑
                raibert_period = current_cfg["period"]
                raibert_threshold = current_cfg["threshold"]
                raibert_kx = current_cfg["k"]
                raibert_ky = current_cfg["k"]
                raibert_offset = current_cfg["offset"]
                # --- handles ---
                robot = env.unwrapped.scene["robot"]

                # --- hip offsets (use fallback) ---
                hip_offsets_B = hip_offsets_B_fallback.to(device=env.unwrapped.device, dtype=torch.float32)  # (4,3)
                hip_x = float(hip_offsets_B[leg_id, 0].item())
                hip_y = float(hip_offsets_B[leg_id, 1].item())

                # --- from beta (this is what you want to verify) ---
                dx_from_beta = px - hip_x
                dy_from_beta = py - hip_y

                # --- measured base velocity in body frame ---
                v_B = robot.data.root_lin_vel_b[env_id]
                vx = float(v_B[0].item())
                vy = float(v_B[1].item())

                # --- command velocity (body frame command) ---
                cmd = env.unwrapped.command_manager.get_command("base_velocity")[env_id]
                vcmd_x = float(cmd[0].item())
                vcmd_y = float(cmd[1].item())
                vcmd_wz = float(cmd[2].item())

                # --- durations ---
                Tst = float(raibert_threshold) * float(raibert_period)
                Tswing = (1.0 - float(raibert_threshold)) * float(raibert_period)

                # --- compute leg_phase & swing_phase (same as beta_l_raibert) ---
                global_phase = ((env.unwrapped.episode_length_buf * env.unwrapped.step_dt) % raibert_period) / raibert_period
                leg_phase = (float(global_phase[env_id].item()) + float(raibert_offset[leg_id])) % 1.0
                swing_phase = (leg_phase - float(raibert_threshold)) / (1.0 - float(raibert_threshold))
                swing_phase = max(0.0, min(1.0, swing_phase))

                # --- SAME formula as beta_l_raibert ---
                dx_calc = (1.0 - swing_phase) * Tswing * vx + 0.5 * Tst * vx + float(raibert_kx) * (vx - vcmd_x)
                dy_calc = (1.0 - swing_phase) * Tswing * vy + 0.5 * Tst * vy + float(raibert_ky) * (vy - vcmd_y)

                # --- action stats ---
                a = actions[env_id]
                act_norm = float(torch.linalg.norm(a).item())
                act_max = float(a.abs().max().item())

                # --- other state ---
                wz = float(robot.data.root_ang_vel_w[env_id, 2].item())
                base_z = float(robot.data.root_pos_w[env_id, 2].item())
                t = float(env.unwrapped.episode_length_buf[env_id].item()) * float(env.unwrapped.step_dt)

                print(
                    f"[{args_cli.p_ref_event}] #{_printed_events:03d} step={sim_step_count} t={t:.2f}s | "
                    f"vcmd_B=({vcmd_x:.3f},{vcmd_y:.3f},{vcmd_wz:.3f}) | "
                    f"v_B=({vx:.3f},{vy:.3f}) wz_W={wz:.3f} z={base_z:.3f} | "
                    f"act_norm={act_norm:.4f} act_max={act_max:.4f} | "
                    f"dx_err={(dx_from_beta - dx_calc):+.3e} dy_err={(dy_from_beta - dy_calc):+.3e} | "
                    f"(leg_phase={leg_phase:.3f}, swing_phase={swing_phase:.3f})"
                )

                _printed_events += 1


                # ------- p_ref log (yaw-only world) -------
                base_pos_w = robot.data.root_pos_w[env_id]
                base_quat_w = robot.data.root_quat_w[env_id]
                yaw_e = _yaw_from_quat_wxyz(base_quat_w)

                px_t = torch.tensor(px, device=base_pos_w.device, dtype=base_pos_w.dtype)
                py_t = torch.tensor(py, device=base_pos_w.device, dtype=base_pos_w.dtype)

                wx, wy = _rotate_xy_by_yaw(px_t, py_t, yaw_e)
                p_ref_wx = base_pos_w[0] + wx
                p_ref_wy = base_pos_w[1] + wy

                t_log.append(t)
                x_log.append(float(p_ref_wx.item()))
                y_log.append(float(p_ref_wy.item()))

                t_rel.append(t)
                dx_rel_log.append(dx_from_beta)
                dy_rel_log.append(dy_from_beta)

                event_count += 1

            # -------------------------
            # Optional: real foot swing trajectory (world x-z)
            # -------------------------
            if args_cli.plot_swing_traj and foot_indices is not None:
                foot_body_idx = foot_indices[leg_id]
                p_foot_w = robot.data.body_pos_w[env_id, foot_body_idx, :]  # (3,)
                fx = float(p_foot_w[0].item())
                fz = float(p_foot_w[2].item())

                contact = None
                if contact_sensor is not None:
                    ci = _map_foot_to_contact_index(contact_sensor, foot_body_idx)
                    if ci is not None:
                        try:
                            if hasattr(contact_sensor.data, "current_contact_time"):
                                ct = contact_sensor.data.current_contact_time
                                contact = 1.0 if float(ct[env_id, ci].item()) > 0.0 else 0.0
                            elif hasattr(contact_sensor.data, "net_forces_w"):
                                fzw = float(contact_sensor.data.net_forces_w[env_id, ci, 2].item())
                                contact = 1.0 if abs(fzw) > 5.0 else 0.0
                        except Exception:
                            contact = None

                if contact is None:
                    contact = 1.0 if (fz <= float(args_cli.ground_z) + 0.02) else 0.0

                if prev_contact_for_seg is None:
                    prev_contact_for_seg = contact

                if (prev_contact_for_seg > 0.5) and (contact < 0.5):
                    _cur_x, _cur_z = [fx], [fz]
                elif contact < 0.5:
                    _cur_x.append(fx)
                    _cur_z.append(fz)
                elif (prev_contact_for_seg < 0.5) and (contact > 0.5):
                    if len(_cur_x) >= 5:
                        swing_segments_xz.append((_cur_x, _cur_z))
                    _cur_x, _cur_z = [], []

                prev_contact_for_seg = contact

            # stop conditions
            if event_count >= int(args_cli.log_steps):
                print(f"[INFO] Reached event samples: {event_count}/{args_cli.log_steps}")
                break
            if sim_step_count >= int(args_cli.max_sim_steps):
                print(f"[WARN] Reached max_sim_steps={args_cli.max_sim_steps}. Stop to avoid long run.")
                break

        # video stop condition
        if args_cli.video:
            timestep_video += 1
            if timestep_video >= args_cli.video_length:
                break

        # real-time sleep
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # -------------------------
    # Plot / Save
    # -------------------------
    out_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(out_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"env{env_id}_leg{leg_id}_{ts}"

    if len(t_log) > 2:
        plt.figure()
        plt.scatter(t_log, x_log)
        plt.xlabel("t (s)")
        plt.ylabel("p_ref_w_x")
        plt.title(f"Raibert p_ref ({args_cli.p_ref_event}) - world x vs t (yaw-only, scatter)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_pRef_wx_vs_t_scatter.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.scatter(t_log, y_log)
        plt.xlabel("t (s)")
        plt.ylabel("p_ref_w_y")
        plt.title(f"Raibert p_ref ({args_cli.p_ref_event}) - world y vs t (yaw-only, scatter)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_pRef_wy_vs_t_scatter.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.scatter(x_log, y_log)
        plt.xlabel("p_ref_w_x")
        plt.ylabel("p_ref_w_y")
        plt.title(f"Raibert p_ref ({args_cli.p_ref_event}) - world x-y scatter (yaw-only)")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_pRef_wx_wy.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.scatter(t_rel, dx_rel_log)
        plt.xlabel("t (s)")
        plt.ylabel("dx_rel = p_ref_x - hip_x (m)")
        plt.title(f"Raibert dx_rel ({args_cli.p_ref_event}) - body frame (scatter)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_dx_rel_vs_t_scatter.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.scatter(t_rel, dy_rel_log)
        plt.xlabel("t (s)")
        plt.ylabel("dy_rel = p_ref_y - hip_y (m)")
        plt.title(f"Raibert dy_rel ({args_cli.p_ref_event}) - body frame (scatter)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_dy_rel_vs_t_scatter.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.scatter(dx_rel_log, dy_rel_log)
        plt.xlabel("dx_rel (m)")
        plt.ylabel("dy_rel (m)")
        plt.title("Raibert (dx_rel, dy_rel) scatter - body frame")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_dxdy_rel_scatter.png"), dpi=200)
        plt.close()

        print(f"[INFO] Saved p_ref and dx/dy plots to: {out_dir}")
    else:
        print("[WARN] Not enough event samples logged to plot p_ref.")

    if args_cli.plot_swing_traj and len(swing_segments_xz) > 0:
        plt.figure()
        for xs, zs in swing_segments_xz[-10:]:
            plt.plot(xs, zs, linestyle="--")
            plt.scatter([xs[0]], [zs[0]], marker="o")   # liftoff
            plt.scatter([xs[-1]], [zs[-1]], marker="x") # touchdown
        plt.xlabel("x (world)")
        plt.ylabel("z (world)")
        plt.title("Foot swing trajectories (world x-z)")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_swing_xz.png"), dpi=200)
        plt.close()
        print(f"[INFO] Saved swing trajectory plot to: {out_dir}")
    elif args_cli.plot_swing_traj:
        print("[WARN] plot_swing_traj enabled but no swing segments were collected.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
