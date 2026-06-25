"""
isaac_sweep.py — Isaac Lab gait robustness sweep, parallel across seeds.

Consistency with MuJoCo sweep (run_sweep.py):
  - Same velocity grid: vx [0,0.3,0.6,0.9,1.2], vy [-0.4..0.4], wz [-0.5..0.5]
  - Same episode length: 1000 policy steps = 10s
  - Same settle window: 100 steps excluded from metrics
  - Same termination thresholds: height<0.20m, |roll|>0.8, |pitch|>0.8
  - Same CSV schema so analyse_sweep.py works on both outputs

Isaac-specific consistency fixes vs first version:
  - gait_id resampling disabled (was firing at step 9, causing cascading falls)
  - base_velocity resampling disabled
  - push_robot disturbance disabled (not present in MuJoCo)
  - Termination suppressed during settle window (grace period matches MuJoCo PD settle)
  - Zero yaw randomisation on reset (MuJoCo always resets facing same direction)
  - Zero joint velocity on reset (MuJoCo resets to settled static pose)
  - Commands re-asserted every step after env.step() to prevent internal override

Usage:
    python isaac_sweep.py \
        --task      Unitree-Go2-Velocity \
        --checkpoint /path/to/model.pt \
        --out       sweep/results/isaac_results_raw.csv \
        --n_runs    5 \
        --headless
"""

import argparse
import csv
import time
import traceback
from itertools import product
from pathlib import Path

# ── Isaac Lab launcher MUST be set up before any other Isaac imports ──────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab gait sweep")
parser.add_argument("--task",         required=True,  help="Isaac Lab task name")
parser.add_argument("--checkpoint",   required=True,  help="Path to policy checkpoint (.pt)")
parser.add_argument("--out",          default="sweep/results/isaac_results_raw.csv")
parser.add_argument("--n_runs",       type=int, default=5,
                    help="Parallel envs per combo = seeds (default: 5)")
parser.add_argument("--max_steps",    type=int, default=1000,
                    help="Max policy steps per episode (default: 1000 = 10s)")
parser.add_argument("--settle_steps", type=int, default=100,
                    help="Steps excluded from metrics + termination at start (default: 100)")
AppLauncher.add_app_launcher_args(parser)
args_cli        = parser.parse_args()
args_cli.headless = True

app_launcher    = AppLauncher(args_cli)
simulation_app  = app_launcher.app

# ── Now safe to import Isaac ──────────────────────────────────────────────────
import gymnasium as gym
import torch
from importlib.metadata import version

import isaaclab_tasks        # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
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

# ── Sweep grid — identical to run_sweep.py ────────────────────────────────────

GAIT_NAMES = {
    0: "bound", 1: "trot",  2: "hop",   3: "amble",
    4: "pronk", 5: "limp",  6: "stand", 7: "run",
}
ALL_GAITS     = list(GAIT_NAMES.keys())
STAND_GAIT_ID = 6

VX_CMDS = [0.0, 0.3, 0.6, 0.9, 1.2]
VY_CMDS = [-0.4, -0.2, 0.0, 0.2, 0.4]
WZ_CMDS = [-0.5, -0.25, 0.0, 0.25, 0.5]

# ── Termination thresholds — identical to sim_runner.py ──────────────────────

FALL_HEIGHT        = 0.15  # m
FALL_HEIGHT_GRACE  = 10    # consecutive steps below threshold before fall declared

# Gait-aware settle window — identical to sim_runner.py
DEFAULT_SETTLE_STEPS = 100
GAIT_SETTLE_STEPS = {
    2: 150,   # hop
    4: 200,   # pronk
}

STEP_DT = 0.01           # policy dt = 10 ms

# ── CSV schema — identical to run_sweep.py ────────────────────────────────────

CSV_FIELDS = [
    "run_id", "seed",
    "gait_id", "gait_name", "vx_cmd", "vy_cmd", "wz_cmd",
    "survived", "survival_steps", "survival_time_s", "termination_reason",
    "mean_vx_actual", "mean_vy_actual", "mean_wz_actual",
    "mean_vx_error",  "mean_vy_error",  "mean_wz_error",
    "mean_height", "std_height", "mean_roll", "mean_pitch",
    "mean_contact_frac", "mean_contact_acc",
    "mean_torque_norm",
    "xy_drift_m",
    "wall_time_s",
]


# ── Command override ──────────────────────────────────────────────────────────

def _force_command(env_unwrapped, gait_id: int, vx: float, vy: float, wz: float):
    """
    Write fixed gait + velocity into all env command buffers.
    Called every step to prevent Isaac's internal command manager from
    overriding our fixed commands via resampling or heading control.
    """
    cm = env_unwrapped.command_manager

    # Gait: UniformIntegerCommand stores live tensor in .value_command
    cm._terms["gait_id"].value_command[:] = float(gait_id)

    # Velocity: UniformVelocityCommand stores live tensor in .vel_command_b
    vel_term = cm._terms["base_velocity"]
    vel_term.vel_command_b[:, 0] = vx
    vel_term.vel_command_b[:, 1] = vy
    vel_term.vel_command_b[:, 2] = wz

    # Stand gait always gets zero velocity (matches gait_conditioned_base_velocity)
    if gait_id == STAND_GAIT_ID:
        vel_term.vel_command_b[:] = 0.0


# ── Sensor helpers ────────────────────────────────────────────────────────────

def _get_contact_ids(env_unwrapped):
    """Return (contact_sensor, foot_col_ids) for FR/FL/RR/RL feet."""
    contact_sensor = env_unwrapped.scene.sensors["contact_forces"]
    name_to_id     = {n: i for i, n in enumerate(contact_sensor.body_names)}
    foot_ids       = [name_to_id[f"{leg}_foot"] for leg in ["FR", "FL", "RR", "RL"]]
    return contact_sensor, foot_ids


def _get_base_contact_id(env_unwrapped, contact_sensor):
    """Cache and return the contact sensor column index for the base link."""
    if not hasattr(env_unwrapped, "_sweep_base_contact_id"):
        cn2id = {n: i for i, n in enumerate(contact_sensor.body_names)}
        env_unwrapped._sweep_base_contact_id = cn2id.get("base", None)
    return env_unwrapped._sweep_base_contact_id


def _get_des_contact(env_unwrapped, n_runs, device) -> torch.Tensor:
    """Read desFeetContact (N,4) from beta cache written by robot_state_s."""
    if hasattr(env_unwrapped, "beta_contact_ref"):
        return env_unwrapped.beta_contact_ref.float()
    return torch.ones(n_runs, 4, device=device)


def _rpy_from_quat_wxyz(q: torch.Tensor):
    """Return (roll, pitch) tensors from (N,4) wxyz quaternion batch."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll  = torch.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = torch.asin(torch.clamp(2*(w*y - z*x), -1.0, 1.0))
    return roll, pitch


# ── Per-episode metric accumulator ───────────────────────────────────────────

class EpisodeAccumulator:
    """Accumulates per-step tensors for N parallel envs."""

    def __init__(self, n: int, device):
        self.n      = n
        self.device = device
        self.reset()

    def reset(self):
        self._vx    = []
        self._vy    = []
        self._wz    = []
        self._exvx  = []
        self._exvy  = []
        self._exwz  = []
        self._h     = []
        self._roll  = []
        self._pitch = []
        self._cfrac = []
        self._cacc  = []
        self._tnorm = []

    def record(self, lin_vel_b, ang_vel_b, height, roll, pitch,
               contact, des_contact, torques_norm, vel_cmd):
        self._vx.append(lin_vel_b[:, 0].clone())
        self._vy.append(lin_vel_b[:, 1].clone())
        self._wz.append(ang_vel_b[:, 2].clone())
        self._exvx.append((lin_vel_b[:, 0] - vel_cmd[0]).abs())
        self._exvy.append((lin_vel_b[:, 1] - vel_cmd[1]).abs())
        self._exwz.append((ang_vel_b[:, 2] - vel_cmd[2]).abs())
        self._h.append(height.clone())
        self._roll.append(roll.abs())
        self._pitch.append(pitch.abs())
        self._cfrac.append(contact.mean(dim=-1))
        acc = ((contact > 0.5) == (des_contact > 0.5)).float().mean(dim=-1)
        self._cacc.append(acc)
        self._tnorm.append(torques_norm.clone())

    def _mean(self, lst):
        if not lst:
            return torch.full((self.n,), float("nan"), device=self.device)
        return torch.stack(lst, dim=0).mean(dim=0)

    def _std(self, lst):
        if not lst:
            return torch.full((self.n,), float("nan"), device=self.device)
        t = torch.stack(lst, dim=0)
        return t.std(dim=0) if t.shape[0] > 1 else torch.zeros(self.n, device=self.device)

    def get(self) -> dict:
        return {
            "mean_vx_actual":    self._mean(self._vx),
            "mean_vy_actual":    self._mean(self._vy),
            "mean_wz_actual":    self._mean(self._wz),
            "mean_vx_error":     self._mean(self._exvx),
            "mean_vy_error":     self._mean(self._exvy),
            "mean_wz_error":     self._mean(self._exwz),
            "mean_height":       self._mean(self._h),
            "std_height":        self._std(self._h),
            "mean_roll":         self._mean(self._roll),
            "mean_pitch":        self._mean(self._pitch),
            "mean_contact_frac": self._mean(self._cfrac),
            "mean_contact_acc":  self._mean(self._cacc),
            "mean_torque_norm":  self._mean(self._tnorm),
        }


# ── Job builder ───────────────────────────────────────────────────────────────

def build_jobs() -> list:
    jobs = []
    for gait_id in ALL_GAITS:
        if gait_id == STAND_GAIT_ID:
            jobs.append({"gait_id": gait_id, "vel_cmd": (0.0, 0.0, 0.0)})
        else:
            for vx, vy, wz in product(VX_CMDS, VY_CMDS, WZ_CMDS):
                jobs.append({"gait_id": gait_id, "vel_cmd": (vx, vy, wz)})
    return jobs


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep(args):
    out_path    = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_runs      = args.n_runs
    max_steps   = args.max_steps
    settle_steps = args.settle_steps

    # ── Build env ─────────────────────────────────────────────────────────
    print("\n[sweep] Building environment...")
    env_cfg = parse_env_cfg(
        args.task,
        device=args_cli.device,
        num_envs=n_runs,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )

    # ── Consistency fixes ─────────────────────────────────────────────────
    # Disable observation noise — MuJoCo sweep also uses clean observations
    env_cfg.observations.policy.enable_corruption = False

    # Disable ALL command resampling — we drive commands manually every step.
    # Root cause of the step-9 cascading failures in v1: gait_id was being
    # resampled by Isaac's command manager, causing a sudden gait switch
    # mid-stride that destabilised the robot.
    env_cfg.commands.base_velocity.resampling_time_range = (9999.0, 9999.0)
    env_cfg.commands.base_velocity.rel_standing_envs     = 0.0
    env_cfg.commands.gait_id.resampling_time_range       = (9999.0, 9999.0)

    # Disable push_robot — not present in MuJoCo, artificially deflates
    # Isaac survival rates by introducing disturbances every 5-10s
    env_cfg.events.push_robot   = None
    env_cfg.events.add_base_mass = None

    # Zero yaw randomisation on reset — MuJoCo always resets facing forward.
    # Random yaw combined with directional velocity commands (vx>0 only)
    # means some resets face backwards, making tracking trivially impossible.
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "yaw": (0.0, 0.0),
    }

    # Zero joint velocity on reset — MuJoCo resets to a PD-settled static pose.
    # Keeping position_range=(1.0,1.0) preserves the default joint positions
    # (same as MuJoCo default), but zero velocity prevents initial momentum.
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    # Set episode length long enough that Isaac never triggers time_out
    # before our max_steps counter does
    env_cfg.episode_length_s = (max_steps + max(GAIT_SETTLE_STEPS.values()) + 50) * STEP_DT

    env = gym.make(args.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # clip_actions must be float|None — False/True cause gymnasium Box errors
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    device = env.unwrapped.device

    # ── Load policy ───────────────────────────────────────────────────────
    print("[sweep] Loading policy...")
    agent_cfg = _parse_agent_cfg(args.task, args_cli)
    resume_path = retrieve_file_path(args.checkpoint)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=device)
    print("[sweep] Policy loaded.")

    # ── Sensor setup ──────────────────────────────────────────────────────
    contact_sensor, foot_col_ids = _get_contact_ids(env.unwrapped)

    # ── CSV setup — append mode for crash recovery ────────────────────────
    csv_exists = out_path.exists()
    csv_fh     = open(out_path, "a", newline="")
    writer     = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    if not csv_exists:
        writer.writeheader()
        csv_fh.flush()
    existing = 0
    if csv_exists:
        with open(out_path) as f:
            existing = sum(1 for _ in f) - 1
        print(f"[sweep] Resuming — {existing} existing rows.")

    # ── Job loop ──────────────────────────────────────────────────────────
    jobs  = build_jobs()
    total = len(jobs)

    n_loco_combos = len([g for g in ALL_GAITS if g != STAND_GAIT_ID]) * \
                    len(VX_CMDS) * len(VY_CMDS) * len(WZ_CMDS)
    n_episodes    = (n_loco_combos + 1) * n_runs

    print(f"\n{'='*60}")
    print(f"  Isaac sweep — {total} command combos × {n_runs} envs")
    print(f"  = {n_episodes} episodes total")
    print(f"  Episode : {max_steps * STEP_DT:.0f}s max  "
          f"settle: {settle_steps * STEP_DT:.0f}s grace")
    print(f"  Fixes   : gait resampling OFF, push_robot OFF, "
          f"yaw reset=0, vel reset=0")
    print(f"  Output  : {out_path}")
    print(f"{'='*60}\n")

    pbar       = tqdm(total=total, unit="combo") if HAS_TQDM else None
    run_id     = existing
    n_survived = 0
    n_done     = 0

    for job_idx, job in enumerate(jobs):
        gait_id        = job["gait_id"]
        vx, vy, wz     = job["vel_cmd"]
        vel_cmd_tensor = (vx, vy, wz)

        t0 = time.time()

        try:
            # ── Reset all envs ────────────────────────────────────────────
            if version("rsl-rl-lib").startswith("2.3."):
                obs, _ = env.get_observations()
            else:
                obs = env.get_observations()

            # Force command immediately after reset, before first step
            _force_command(env.unwrapped, gait_id, vx, vy, wz)

            # Record start XY for drift metric
            start_xy = env.unwrapped.scene["robot"].data.root_pos_w[:, :2].clone()

            # Per-env termination tracking
            terminated        = torch.zeros(n_runs, dtype=torch.bool, device=device)
            survival_steps    = torch.zeros(n_runs, dtype=torch.long, device=device)
            term_reason       = ["running"] * n_runs
            low_height_count  = torch.zeros(n_runs, dtype=torch.long, device=device)

            # Gait-aware settle: pronk/hop need longer startup window
            ep_settle = GAIT_SETTLE_STEPS.get(gait_id, DEFAULT_SETTLE_STEPS)

            accum = EpisodeAccumulator(n_runs, device)

            # ── Episode loop ──────────────────────────────────────────────
            for step in range(max_steps + ep_settle):

                with torch.inference_mode():
                    actions = policy(obs)

                obs, _, dones, infos = env.step(actions)

                # Re-assert command every step — Isaac's command manager
                # runs _update_command() internally during env.step() and
                # can override vel_command_b via heading control logic
                _force_command(env.unwrapped, gait_id, vx, vy, wz)

                # Suppress Isaac termination during settle window —
                # equivalent to MuJoCo's PD settle phase where the robot
                # is not yet running the policy and can't "fail"
                if step < ep_settle:
                    env.unwrapped.reset_buf[:] = 0

                # ── Read state ────────────────────────────────────────────
                robot      = env.unwrapped.scene["robot"]
                lin_vel_b  = robot.data.root_lin_vel_b       # (N,3)
                ang_vel_b  = robot.data.root_ang_vel_b       # (N,3)
                height     = robot.data.root_pos_w[:, 2]     # (N,)
                q_wxyz     = robot.data.root_quat_w          # (N,4)
                roll, pitch = _rpy_from_quat_wxyz(q_wxyz)

                contact_time = contact_sensor.data.current_contact_time
                foot_contact = (contact_time[:, foot_col_ids] > 0.0).float()
                des_contact  = _get_des_contact(env.unwrapped, n_runs, device)
                torques_norm = torch.norm(robot.data.applied_torque, dim=-1)

                # ── Termination — same logic as sim_runner.py ─────────────
                # Only terminate on sustained low height (physical fall).
                # Orientation, base contact, velocity divergence are metrics
                # only — never termination conditions. Consistent with MuJoCo.
                if step >= ep_settle:
                    below = height < FALL_HEIGHT
                    low_height_count = torch.where(
                        below,
                        low_height_count + 1,
                        torch.zeros_like(low_height_count),
                    )
                    fell_h     = low_height_count >= FALL_HEIGHT_GRACE
                    newly_done = ~terminated & fell_h

                    for i in range(n_runs):
                        if newly_done[i] and not terminated[i]:
                            survival_steps[i] = step - ep_settle
                            term_reason[i]    = "fall_height"
                    terminated |= newly_done

                # ── Accumulate metrics after settle window ─────────────────
                if step >= ep_settle:
                    accum.record(
                        lin_vel_b, ang_vel_b, height, roll, pitch,
                        foot_contact, des_contact, torques_norm,
                        vel_cmd_tensor,
                    )

                # Timeout: envs still alive at end of episode
                if step == max_steps + ep_settle - 1:
                    for i in range(n_runs):
                        if not terminated[i]:
                            survival_steps[i] = max_steps
                            term_reason[i]    = "timeout"
                            terminated[i]     = True

                if terminated.all():
                    break

            # ── Compute metrics ───────────────────────────────────────────
            wall_time = time.time() - t0
            metrics   = accum.get()
            end_xy    = env.unwrapped.scene["robot"].data.root_pos_w[:, :2]
            xy_drift  = torch.norm(end_xy - start_xy, dim=-1)

            # ── Write one CSV row per env ─────────────────────────────────
            for i in range(n_runs):
                survived    = term_reason[i] == "timeout"
                n_survived += int(survived)
                n_done     += 1

                row = {
                    "run_id":             run_id,
                    "seed":               i,
                    "gait_id":            gait_id,
                    "gait_name":          GAIT_NAMES[gait_id],
                    "vx_cmd":             round(vx, 2),
                    "vy_cmd":             round(vy, 2),
                    "wz_cmd":             round(wz, 2),
                    "survived":           survived,
                    "survival_steps":     int(survival_steps[i].item()),
                    "survival_time_s":    round(int(survival_steps[i].item()) * STEP_DT, 3),
                    "termination_reason": term_reason[i],
                    "mean_vx_actual":     round(float(metrics["mean_vx_actual"][i].item()),  4),
                    "mean_vy_actual":     round(float(metrics["mean_vy_actual"][i].item()),  4),
                    "mean_wz_actual":     round(float(metrics["mean_wz_actual"][i].item()),  4),
                    "mean_vx_error":      round(float(metrics["mean_vx_error"][i].item()),   4),
                    "mean_vy_error":      round(float(metrics["mean_vy_error"][i].item()),   4),
                    "mean_wz_error":      round(float(metrics["mean_wz_error"][i].item()),   4),
                    "mean_height":        round(float(metrics["mean_height"][i].item()),     4),
                    "std_height":         round(float(metrics["std_height"][i].item()),      4),
                    "mean_roll":          round(float(metrics["mean_roll"][i].item()),       4),
                    "mean_pitch":         round(float(metrics["mean_pitch"][i].item()),      4),
                    "mean_contact_frac":  round(float(metrics["mean_contact_frac"][i].item()), 4),
                    "mean_contact_acc":   round(float(metrics["mean_contact_acc"][i].item()),  4),
                    "mean_torque_norm":   round(float(metrics["mean_torque_norm"][i].item()),  4),
                    "xy_drift_m":         round(float(xy_drift[i].item()),                  4),
                    "wall_time_s":        round(wall_time / n_runs, 3),
                }
                writer.writerow(row)
                run_id += 1

            csv_fh.flush()

        except Exception as e:
            wall_time = time.time() - t0
            print(f"\n[ERROR] job={job_idx} gait={GAIT_NAMES[gait_id]} "
                  f"vel=({vx:.2f},{vy:.2f},{wz:.2f})")
            traceback.print_exc()
            for i in range(n_runs):
                writer.writerow({
                    "run_id": run_id, "seed": i,
                    "gait_id": gait_id, "gait_name": GAIT_NAMES[gait_id],
                    "vx_cmd": round(vx,2), "vy_cmd": round(vy,2), "wz_cmd": round(wz,2),
                    "survived": False, "survival_steps": -1, "survival_time_s": -1,
                    "termination_reason": f"error: {type(e).__name__}",
                    "wall_time_s": round(wall_time / n_runs, 3),
                })
                run_id += 1
            csv_fh.flush()

        if pbar:
            pbar.update(1)
            pbar.set_postfix_str(
                f"gait={GAIT_NAMES[gait_id]} "
                f"vx={vx:.1f} "
                f"sr={100*n_survived/max(n_done,1):.0f}%"
            )

        if (job_idx + 1) % 50 == 0:
            print(f"\n[{job_idx+1}/{total}]  "
                  f"survival={100*n_survived/max(n_done,1):.1f}%  "
                  f"rows={run_id - existing}")

    if pbar:
        pbar.close()
    csv_fh.close()
    env.close()

    print(f"\n{'='*60}")
    print("  ISAAC SWEEP COMPLETE")
    print(f"  Total rows : {run_id - existing}")
    print(f"  Survived   : {n_survived}  ({100*n_survived/max(n_done,1):.1f}%)")
    print(f"  Output     : {out_path}")
    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_sweep(args_cli)
    simulation_app.close()