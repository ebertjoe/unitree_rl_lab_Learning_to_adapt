"""
isaac_switch_sweep.py — Isaac Lab gait-switch robustness sweep.

Same design as run_switch_sweep.py — all 56 gait pairs × 4 vel cmds × 5 seeds.
Outputs the same CSV schema so analyse_switch_sweep.py works on both.

Usage:
    python isaac_switch_sweep.py \
        --task      Unitree-Go2-Velocity \
        --checkpoint /path/to/model.pt \
        --out       sweep/results/isaac_switch_results_raw.csv \
        --headless
"""

import argparse
import csv
import json
import time
import traceback
from pathlib import Path

# ── Isaac Lab launcher ────────────────────────────────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab gait-switch sweep")
parser.add_argument("--task",       required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out",        default="sweep/results/isaac_switch_results_raw.csv")
parser.add_argument("--n_runs",     type=int, default=5)
parser.add_argument("--save_timeseries", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli          = parser.parse_args()
args_cli.headless = True

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac imports ─────────────────────────────────────────────────────────────
import gymnasium as gym
import numpy as np
import torch
from importlib.metadata import version

import isaaclab_tasks       # noqa
import unitree_rl_lab.tasks # noqa

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

try:
    import cli_args
    def _parse_agent_cfg(task, args):
        return cli_args.parse_rsl_rl_cfg(task, args)
except ImportError:
    def _parse_agent_cfg(task, args):
        from isaaclab_tasks.utils import load_cfg_from_registry
        return load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Constants — identical to run_switch_sweep.py ─────────────────────────────

GAIT_NAMES = {
    0:"bound",1:"trot",2:"hop",3:"amble",
    4:"pronk",5:"limp",6:"stand",7:"run",
}
ALL_GAITS     = list(GAIT_NAMES.keys())
STAND_GAIT_ID = 6

SWITCH_VEL_CMDS = [
    (0.0, 0.0, 0.0),
    (0.6, 0.0, 0.0),
    (1.2, 0.0, 0.0),
    (0.6, 0.2, 0.0),
]

PHASE1_STEPS        = 300
PHASE2_STEPS        = 700
MAX_EP_STEPS        = PHASE1_STEPS + PHASE2_STEPS
PRE_WINDOW          = 100
TRANS_WINDOW        = 100
POST_WINDOW         = 200
RECOVERY_HEIGHT_TOL = 0.02
RECOVERY_VEL_TOL    = 0.10
FALL_HEIGHT         = 0.15
FALL_HEIGHT_GRACE   = 10
DEFAULT_SETTLE_STEPS= 100
GAIT_SETTLE_STEPS   = {2: 150, 4: 200}
STEP_DT             = 0.01

CSV_FIELDS = [
    "run_id","seed",
    "gait_from","gait_from_name","gait_to","gait_to_name",
    "vx_cmd","vy_cmd","wz_cmd",
    "survived","survival_steps","survival_time_s","termination_reason",
    "pre_mean_height","pre_mean_vx_error","pre_mean_contact_acc",
    "trans_min_height","trans_max_vx_error","trans_max_roll",
    "height_recovery_steps","vel_recovery_steps",
    "post_mean_height","post_mean_vx_error",
    "post_mean_contact_acc","post_mean_torque_norm",
    "wall_time_s",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _force_command(env_unwrapped, gait_id, vx, vy, wz):
    cm = env_unwrapped.command_manager
    cm._terms["gait_id"].value_command[:] = float(gait_id)
    vel = cm._terms["base_velocity"]
    vel.vel_command_b[:, 0] = vx
    vel.vel_command_b[:, 1] = vy
    vel.vel_command_b[:, 2] = wz
    if gait_id == STAND_GAIT_ID:
        vel.vel_command_b[:] = 0.0


def _get_contact_ids(env_unwrapped):
    cs = env_unwrapped.scene.sensors["contact_forces"]
    n2i = {n: i for i, n in enumerate(cs.body_names)}
    return cs, [n2i[f"{l}_foot"] for l in ["FR","FL","RR","RL"]]


def _rpy(q):
    w,x,y,z = q[:,0],q[:,1],q[:,2],q[:,3]
    roll  = torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = torch.asin(torch.clamp(2*(w*y-z*x),-1.,1.))
    return roll, pitch


def _des_contact(env_unwrapped, n, device):
    if hasattr(env_unwrapped, "beta_contact_ref"):
        return env_unwrapped.beta_contact_ref.float()
    return torch.ones(n, 4, device=device)


def _m(lst): return float(np.mean(lst)) if lst else float("nan")


def build_jobs(n_runs):
    jobs = []
    for gf in ALL_GAITS:
        for gt in ALL_GAITS:
            if gf == gt: continue
            for vel in SWITCH_VEL_CMDS:
                jobs.append({"gait_from": gf, "gait_to": gt, "vel_cmd": vel})
    return jobs


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep(args):
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ts_dir = out_path.parent / "isaac_switch_timeseries"
    if args.save_timeseries:
        ts_dir.mkdir(parents=True, exist_ok=True)

    n_runs = args.n_runs

    # ── Build env ─────────────────────────────────────────────────────────
    print("\n[sweep] Building environment...")
    env_cfg = parse_env_cfg(
        args.task, device=args_cli.device, num_envs=n_runs,
        use_fabric=True, entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.observations.policy.enable_corruption             = False
    env_cfg.commands.base_velocity.resampling_time_range      = (9999., 9999.)
    env_cfg.commands.base_velocity.rel_standing_envs          = 0.0
    env_cfg.commands.gait_id.resampling_time_range            = (9999., 9999.)
    env_cfg.events.push_robot                                  = None
    env_cfg.events.add_base_mass                               = None
    env_cfg.events.reset_base.params["pose_range"]            = {
        "x":(-0.5,0.5), "y":(-0.5,0.5), "yaw":(0.,0.)}
    env_cfg.events.reset_robot_joints.params["velocity_range"]= (0., 0.)
    max_settle = max(GAIT_SETTLE_STEPS.values())
    env_cfg.episode_length_s = (MAX_EP_STEPS + max_settle + 50) * STEP_DT

    env = gym.make(args.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    device = env.unwrapped.device

    print("[sweep] Loading policy...")
    agent_cfg   = _parse_agent_cfg(args.task, args_cli)
    resume_path = retrieve_file_path(args.checkpoint)
    runner      = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=device)
    print("[sweep] Ready.")

    cs, foot_ids = _get_contact_ids(env.unwrapped)

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_exists = out_path.exists()
    csv_fh     = open(out_path, "a", newline="")
    writer     = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    if not csv_exists:
        writer.writeheader(); csv_fh.flush()
    existing = 0
    if csv_exists:
        with open(out_path) as f:
            existing = sum(1 for _ in f) - 1
        print(f"[resume] {existing} existing rows.")

    jobs       = build_jobs(n_runs)
    total      = len(jobs)
    pbar       = tqdm(total=total, unit="ep") if HAS_TQDM else None
    run_id     = existing
    n_survived = 0
    n_done     = 0

    print(f"\n{'='*60}")
    print(f"  Isaac switch sweep — {total} episodes")
    print(f"  56 pairs × {len(SWITCH_VEL_CMDS)} vel cmds × {n_runs} seeds")
    print(f"  Output: {out_path}")
    print(f"{'='*60}\n")

    for job in jobs:
        gf  = job["gait_from"]
        gt  = job["gait_to"]
        vel = job["vel_cmd"]
        sd  = job["seed"]
        vx, vy, wz = vel
        ep_settle = GAIT_SETTLE_STEPS.get(gf, DEFAULT_SETTLE_STEPS)

        t0 = time.time()
        try:
            # ── Reset ─────────────────────────────────────────────────────
            if version("rsl-rl-lib").startswith("2.3."):
                obs, _ = env.get_observations()
            else:
                obs = env.get_observations()
            _force_command(env.unwrapped, gf, vx, vy, wz)

            # Per-env state
            terminated       = torch.zeros(n_runs, dtype=torch.bool, device=device)
            survival_steps_t = torch.zeros(n_runs, dtype=torch.long, device=device)
            term_reason      = ["running"] * n_runs
            low_h_count      = torch.zeros(n_runs, dtype=torch.long, device=device)
            switch_done      = False

            # Metric buffers (per env, lists of tensors)
            pre_h=[]; pre_vxe=[]; pre_cacc=[]
            trans_h=[]; trans_vxe=[]; trans_roll=[]
            post_h=[]; post_vxe=[]; post_cacc=[]; post_tnorm=[]
            ts_h=[]; ts_vxe=[]; ts_step=[]

            for step in range(MAX_EP_STEPS + ep_settle):
                policy_step = step - ep_settle

                # ── Switch ────────────────────────────────────────────────
                if policy_step == PHASE1_STEPS and not switch_done:
                    _force_command(env.unwrapped, gt, vx, vy, wz)
                    switch_done = True

                with torch.inference_mode():
                    actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

                # Re-assert command
                current_gait = gt if switch_done else gf
                _force_command(env.unwrapped, current_gait, vx, vy, wz)

                # Suppress Isaac termination during settle
                if step < ep_settle:
                    env.unwrapped.reset_buf[:] = 0

                # ── Read state ────────────────────────────────────────────
                robot   = env.unwrapped.scene["robot"]
                lv_b    = robot.data.root_lin_vel_b
                av_b    = robot.data.root_ang_vel_b
                height  = robot.data.root_pos_w[:, 2]
                q_wxyz  = robot.data.root_quat_w
                roll, _ = _rpy(q_wxyz)
                ct      = cs.data.current_contact_time
                fc      = (ct[:, foot_ids] > 0.0).float()
                dc      = _des_contact(env.unwrapped, n_runs, device)
                tnorm   = torch.norm(robot.data.applied_torque, dim=-1)
                vxe     = (lv_b[:, 0] - vx).abs()

                # ── Termination ───────────────────────────────────────────
                if policy_step >= 0:
                    below = height < FALL_HEIGHT
                    low_h_count = torch.where(
                        below, low_h_count+1,
                        torch.zeros_like(low_h_count))
                    fell = (~terminated) & (low_h_count >= FALL_HEIGHT_GRACE)
                    for i in range(n_runs):
                        if fell[i]:
                            survival_steps_t[i] = policy_step
                            if policy_step < PHASE1_STEPS:
                                term_reason[i] = "fall_phase1"
                            elif policy_step < PHASE1_STEPS + TRANS_WINDOW:
                                term_reason[i] = "fall_transition"
                            else:
                                term_reason[i] = "fall_phase2"
                    terminated |= fell

                # ── Accumulate metrics ────────────────────────────────────
                if policy_step >= 0:
                    cacc = ((fc>0.5)==(dc>0.5)).float().mean(dim=-1)

                    if PHASE1_STEPS-PRE_WINDOW <= policy_step < PHASE1_STEPS:
                        pre_h.append(height.clone()); pre_vxe.append(vxe.clone())
                        pre_cacc.append(cacc.clone())

                    if PHASE1_STEPS <= policy_step < PHASE1_STEPS+TRANS_WINDOW:
                        trans_h.append(height.clone()); trans_vxe.append(vxe.clone())
                        trans_roll.append(roll.abs().clone())

                    post_start = MAX_EP_STEPS - POST_WINDOW
                    if policy_step >= post_start:
                        post_h.append(height.clone()); post_vxe.append(vxe.clone())
                        post_cacc.append(cacc.clone()); post_tnorm.append(tnorm.clone())

                    if PHASE1_STEPS-PRE_WINDOW <= policy_step < PHASE1_STEPS+TRANS_WINDOW:
                        ts_h.append(height.clone()); ts_vxe.append(vxe.clone())
                        ts_step.append(policy_step - PHASE1_STEPS)

                # Timeout
                if policy_step == MAX_EP_STEPS - 1:
                    for i in range(n_runs):
                        if not terminated[i]:
                            survival_steps_t[i] = MAX_EP_STEPS
                            term_reason[i] = "timeout"
                            terminated[i]  = True

                if terminated.all():
                    break

            # ── Compute metrics ───────────────────────────────────────────
            wall_time = time.time() - t0

            def _sm(lst):
                if not lst: return torch.full((n_runs,), float("nan"), device=device)
                return torch.stack(lst).mean(dim=0)
            def _smin(lst):
                if not lst: return torch.full((n_runs,), float("nan"), device=device)
                return torch.stack(lst).min(dim=0).values
            def _smax(lst):
                if not lst: return torch.full((n_runs,), float("nan"), device=device)
                return torch.stack(lst).max(dim=0).values

            pre_h_mean  = _sm(pre_h)
            pre_vxe_mean= _sm(pre_vxe)

            # Recovery: per env
            def _hrec(env_idx):
                if not trans_h: return MAX_EP_STEPS
                ref = float(pre_h_mean[env_idx].item())
                all_h = [float(t[env_idx]) for t in trans_h] + \
                        [float(t[env_idx]) for t in post_h]
                for i, h in enumerate(all_h):
                    if abs(h - ref) <= RECOVERY_HEIGHT_TOL: return i
                return MAX_EP_STEPS

            def _vrec(env_idx):
                all_vxe = [float(t[env_idx]) for t in trans_vxe] + \
                          [float(t[env_idx]) for t in post_vxe]
                for i, e in enumerate(all_vxe):
                    if e < RECOVERY_VEL_TOL: return i
                return MAX_EP_STEPS

            # ── Write rows ────────────────────────────────────────────────
            for i in range(n_runs):
                survived = term_reason[i] == "timeout"
                n_survived += int(survived)
                n_done     += 1

                row = {
                    "run_id": run_id, "seed": i,
                    "gait_from": gf, "gait_from_name": GAIT_NAMES[gf],
                    "gait_to":   gt, "gait_to_name":   GAIT_NAMES[gt],
                    "vx_cmd": round(vx,2), "vy_cmd": round(vy,2), "wz_cmd": round(wz,2),
                    "survived":           survived,
                    "survival_steps":     int(survival_steps_t[i].item()),
                    "survival_time_s":    round(int(survival_steps_t[i].item())*STEP_DT, 3),
                    "termination_reason": term_reason[i],
                    "pre_mean_height":    round(float(pre_h_mean[i].item()), 4),
                    "pre_mean_vx_error":  round(float(pre_vxe_mean[i].item()), 4),
                    "pre_mean_contact_acc": round(float(_sm(pre_cacc)[i].item()), 4),
                    "trans_min_height":   round(float(_smin(trans_h)[i].item()), 4),
                    "trans_max_vx_error": round(float(_smax(trans_vxe)[i].item()), 4),
                    "trans_max_roll":     round(float(_smax(trans_roll)[i].item()), 4),
                    "height_recovery_steps": _hrec(i),
                    "vel_recovery_steps":    _vrec(i),
                    "post_mean_height":   round(float(_sm(post_h)[i].item()), 4),
                    "post_mean_vx_error": round(float(_sm(post_vxe)[i].item()), 4),
                    "post_mean_contact_acc": round(float(_sm(post_cacc)[i].item()), 4),
                    "post_mean_torque_norm": round(float(_sm(post_tnorm)[i].item()), 4),
                    "wall_time_s": round(wall_time/n_runs, 3),
                }
                writer.writerow(row)
                run_id += 1

            csv_fh.flush()

            if args.save_timeseries and ts_h:
                for i in range(n_runs):
                    fname = (f"ts_gf{gf}_gt{gt}_vx{vx:.1f}"
                             f"_vy{vy:.1f}_s{i}.npz")
                    np.savez_compressed(
                        ts_dir / fname,
                        step   = np.array(ts_step, dtype=np.int16),
                        height = np.array([float(t[i]) for t in ts_h], dtype=np.float32),
                        vx_error=np.array([float(t[i]) for t in ts_vxe], dtype=np.float32),
                    )

        except Exception as e:
            wall_time = time.time() - t0
            print(f"\n[ERROR] gf={GAIT_NAMES[gf]} gt={GAIT_NAMES[gt]} vel={vel} seed={sd}")
            traceback.print_exc()
            for i in range(n_runs):
                writer.writerow({
                    "run_id":run_id,"seed":i,
                    "gait_from":gf,"gait_from_name":GAIT_NAMES[gf],
                    "gait_to":gt,"gait_to_name":GAIT_NAMES[gt],
                    "vx_cmd":round(vx,2),"vy_cmd":round(vy,2),"wz_cmd":round(wz,2),
                    "survived":False,"survival_steps":-1,"survival_time_s":-1,
                    "termination_reason":f"error:{type(e).__name__}",
                    "wall_time_s":round(wall_time/n_runs,3),
                })
                run_id += 1
            csv_fh.flush()
            n_done += n_runs

        if pbar:
            pbar.update(1)
            pbar.set_postfix_str(
                f"{GAIT_NAMES[gf]}→{GAIT_NAMES[gt]} "
                f"sr={100*n_survived/max(n_done,1):.0f}%"
            )
        if n_done % 200 == 0:
            print(f"\n[{n_done}/{total}] survival={100*n_survived/max(n_done,1):.1f}%")

    if pbar: pbar.close()
    csv_fh.close()
    env.close()

    print(f"\n{'='*60}")
    print("  ISAAC SWITCH SWEEP COMPLETE")
    print(f"  Total rows : {run_id - existing}")
    print(f"  Survived   : {n_survived}  ({100*n_survived/max(n_done,1):.1f}%)")
    print(f"  Output     : {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_sweep(args_cli)
    simulation_app.close()