"""Isaac Lab backend of the §4.1 robustness sweep.

This module must only be imported *after* ``isaaclab.app.AppLauncher`` has
started the simulation app - see ``run_robustness_sweep_isaac.py``.

Protocol notes
--------------
* Both command terms are frozen: ``base_velocity`` and ``gait_id`` never resample
  and are rewritten into the command buffers after every ``env.step`` so the
  command manager cannot override the swept cell.
* ``base_contact`` and ``bad_orientation`` are removed from the environment's
  termination manager and re-implemented in the harness. Isaac Lab resets an
  environment inside ``step()`` as soon as a termination fires, which would both
  corrupt the metrics of that episode and make the per-gait settle window
  impossible to honour. With the terms removed the harness owns every episode
  boundary and calls ``env.reset()`` itself.
* The Raibert planner state (phase compensation, blended period/nominal height,
  reference foot positions) is primed at the start of every batch so each episode
  starts at phase 0 with the parameters of the gait under test, instead of
  inheriting the blend from the previous batch.
* The scheduled contact state used for the gait contact accuracy is read straight
  from the planner (``env.beta_contact_ref``), i.e. exactly the signal the policy
  is conditioned on.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from robustness_sweep import grid as G
from robustness_sweep.metrics import (
    CSV_FIELDS,
    BatchAccumulator,
    make_error_row,
    make_row,
    orientation_angle_from_projected_gravity,
)
from robustness_sweep.video import GaitBalancedSelector, SweepVideoWriter

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _muted(enabled: bool):
    """Swallow the environment's periodic debug prints during the rollout."""
    if not enabled:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _get_obs(env):
    """rsl-rl 2.3 returns ``(obs, extras)``, rsl-rl 3.x returns a TensorDict.

    A TorchScript policy takes a plain tensor, so the policy group is unwrapped
    here rather than at every call site.
    """
    return _as_policy_obs(env.get_observations())


def _as_policy_obs(out):
    """Unwrap a plain policy-observation tensor from whatever rsl-rl handed back."""
    out = out[0] if isinstance(out, tuple) else out
    if isinstance(out, torch.Tensor):
        return out
    # TensorDict (or any mapping): the actor is fed the "policy" group.
    try:
        return out["policy"]
    except (KeyError, TypeError):
        pass
    for key in ("obs", "observations", "actor"):
        try:
            return out[key]
        except (KeyError, TypeError):
            continue
    raise RuntimeError(f"cannot extract the policy observation from {type(out)!r}")


def _observations_blob() -> str | None:
    """git object hash of the observations.py the sweep is running against."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "hash-object",
             "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/observations.py"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return (x + torch.pi) % (2 * torch.pi) - torch.pi


def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield i // size, seq[i : i + size]


# ---------------------------------------------------------------------------
# environment configuration
# ---------------------------------------------------------------------------


def make_env_cfg(task: str, num_envs: int, device: str, args) -> object:
    """Build the play env cfg and apply every protocol override of §4.1."""
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    cfg = parse_env_cfg(
        task,
        device=device,
        num_envs=num_envs,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )

    # -- observations ------------------------------------------------------
    cfg.observations.policy.enable_corruption = bool(args.obs_noise)

    # -- commands: frozen, driven by the harness ---------------------------
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.gait_id.resampling_time_range = (1.0e9, 1.0e9)

    # -- terminations: owned by the harness --------------------------------
    # ``time_out`` stays so the termination manager is never empty, but the
    # episode length is set past the longest rollout so it can never fire.
    cfg.terminations.base_contact = None
    cfg.terminations.bad_orientation = None
    cfg.episode_length_s = (G.EPISODE_STEPS + G.MAX_SETTLE_STEPS + 100) * G.POLICY_DT

    # -- events ------------------------------------------------------------
    # Pushes are an extra disturbance that the §4.1 protocol does not apply.
    if not args.push_robot:
        cfg.events.push_robot = None
    if not args.domain_rand:
        cfg.events.add_base_mass = None
        cfg.events.physics_material = None
    # A random reset yaw combined with a body-frame velocity command makes the
    # command unreachable for part of the episode, so the heading is fixed.
    cfg.events.reset_base.params["pose_range"] = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "yaw": (0.0, 0.0),
    }
    if args.deterministic_reset:
        cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    # -- viewer / video ----------------------------------------------------
    if args.video:
        cfg.viewer.resolution = tuple(args.video_resolution)
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name = "robot"
        cfg.viewer.env_index = 0
        cfg.viewer.eye = tuple(args.video_eye)
        cfg.viewer.lookat = (0.0, 0.0, 0.0)

    return cfg


def check_gait_table(task: str) -> None:
    """Fail loudly if the mirrored gait table has drifted from the environment."""
    try:
        from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import GAIT_CONFIGS as ENV_TABLE
    except Exception as exc:  # pragma: no cover - non-go2 task
        print(f"[sweep] gait table not verified ({exc})")
        return
    mismatches = []
    for key, ref in ENV_TABLE.items():
        mine = G.GAIT_CONFIGS.get(key)
        if mine is None or any(mine.get(k) != v for k, v in ref.items()):
            mismatches.append(key)
    if mismatches:
        raise RuntimeError(
            "robustness_sweep.grid.GAIT_CONFIGS is out of sync with the environment "
            f"for gait id(s) {mismatches}. Update the mirrored table before sweeping."
        )
    print("[sweep] gait table verified against the environment configuration.")


# ---------------------------------------------------------------------------
# command forcing and planner priming
# ---------------------------------------------------------------------------


def force_commands(u, gait_ids: torch.Tensor, vel_cmd: torch.Tensor) -> None:
    """Write the swept cell into both command buffers of every environment."""
    cm = u.command_manager
    cm._terms["gait_id"].value_command[:, 0] = gait_ids.to(torch.float32)
    vel_term = cm._terms["base_velocity"]
    vel_term.vel_command_b[:] = vel_cmd
    # ``stand`` is always evaluated at zero velocity; mirrors the policy's own
    # gait_conditioned_base_velocity observation.
    stand = gait_ids == G.STAND_GAIT_ID
    if stand.any():
        vel_term.vel_command_b[stand] = 0.0
    if hasattr(vel_term, "is_standing_env"):
        vel_term.is_standing_env[:] = False


def prime_planner(u, gait_ids: torch.Tensor) -> None:
    """Restart the Raibert planner at phase 0 with the target gait's parameters."""
    # The buffers only exist once the observation has been computed at least once.
    if not hasattr(u, "_gait_table_tensors"):
        return
    u._phase_compensation.zero_()
    u._prev_gait_ids = gait_ids.clone()
    u._current_period_blended = u._gait_periods[gait_ids].unsqueeze(1).clone()
    u._current_znoms_blended = u._gait_znoms[gait_ids].unsqueeze(1).clone()
    if hasattr(u, "_raibert_hip_pos_B_static"):
        u._raibert_p_ref_B = u._raibert_hip_pos_B_static.clone()
        u._raibert_p_ref_B[..., 2] = u._current_znoms_blended.expand(-1, 4)
    if hasattr(u, "_raibert_prev_c"):
        u._raibert_prev_c[:] = 1.0
    # Invalidate the per-step planner cache so the next observation is rebuilt
    # with the commands we have just written.
    u._beta_last_step = -1


# ---------------------------------------------------------------------------
# sensor accessors
# ---------------------------------------------------------------------------


def _quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Rotation matrix from a (n,4) wxyz quaternion; matches the env's convention."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=1)


class SceneRefs:
    """Cached body / sensor indices for the Go2 scene."""

    def __init__(self, u):
        self.robot = u.scene["robot"]
        self.contact = u.scene.sensors["contact_forces"]
        name_to_id = {n: i for i, n in enumerate(self.contact.body_names)}
        self.foot_ids = [name_to_id[f"{leg}_foot"] for leg in G.FOOT_ORDER]
        # Body indices of the same feet on the articulation (distinct from the
        # contact-sensor indices above) so the measured foot position can be
        # compared against the Raibert reference. Same FR, FL, RR, RL order as
        # ``get_go2_hip_positions_B`` in mdp/observations.py.
        name_to_body = {n: i for i, n in enumerate(self.robot.data.body_names)}
        self.foot_body_ids = [name_to_body[f"{leg}_foot"] for leg in G.FOOT_ORDER]
        self.base_id = name_to_id.get("base")
        if self.base_id is None:
            raise RuntimeError(
                "contact sensor has no 'base' body; cannot evaluate the base_contact termination"
            )

    def foot_contact(self) -> torch.Tensor:
        return (self.contact.data.current_contact_time[:, self.foot_ids] > 0.0).float()

    def foot_pos_b(self) -> torch.Tensor:
        """Measured foot positions in the base frame, (n,4,3).

        Mirrors ``real_foot_pos_b`` in ``mdp.observations.robot_state_s`` so the
        comparison against ``env.beta_p_ref_B`` is like-for-like.
        """
        d = self.robot.data
        rel_w = d.body_pos_w[:, self.foot_body_ids, :] - d.root_pos_w.unsqueeze(1)
        R_WB = _quat_wxyz_to_rotmat(d.root_quat_w)
        return torch.bmm(R_WB.transpose(1, 2), rel_w.transpose(1, 2)).transpose(1, 2)

    def base_contact_force(self) -> torch.Tensor:
        hist = self.contact.data.net_forces_w_history
        if hist is not None:
            return torch.max(torch.norm(hist[:, :, self.base_id], dim=-1), dim=1)[0]
        return torch.norm(self.contact.data.net_forces_w[:, self.base_id], dim=-1)


def scheduled_contact(u, n: int, device) -> torch.Tensor:
    ref = getattr(u, "beta_contact_ref", None)
    if ref is None:
        raise RuntimeError(
            "env.beta_contact_ref is missing - the Raibert planner has not run. "
            "The gait contact accuracy cannot be computed without it."
        )
    return ref.float()


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------


class IsaacRobustnessSweep:
    def __init__(self, args, policy_ref, env, runner_policy, device):
        self.args = args
        self.policy_ref = policy_ref
        self.env = env
        self.u = env.unwrapped
        self.policy = runner_policy
        self.device = device
        self.refs = SceneRefs(self.u)
        self.n = self.u.num_envs

        self.provenance = {
            "simulator": "isaac",
            "experiment": policy_ref.experiment,
            "run": policy_ref.run,
            "checkpoint": policy_ref.checkpoint.name,
        }

        self.out_dir = Path(args.out_dir) if args.out_dir else policy_ref.output_dir("isaac")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "episodes.csv"

        self.video: SweepVideoWriter | None = None
        self.selector = GaitBalancedSelector()
        if args.video:
            self.video = SweepVideoWriter(
                self.out_dir / "video" / "sweep.mp4",
                fps=args.video_fps,
                manifest_path=self.out_dir / "video" / "manifest.csv",
            )

    # -- bookkeeping -------------------------------------------------------

    def _open_csv(self):
        existing = 0
        if self.csv_path.exists() and self.args.resume:
            with open(self.csv_path) as f:
                existing = max(sum(1 for _ in f) - 1, 0)
            fh = open(self.csv_path, "a", newline="")
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        else:
            fh = open(self.csv_path, "w", newline="")
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()
            fh.flush()
        return fh, writer, existing

    def _write_config(self, episodes):
        cfg = {
            "simulator": "isaac",
            "task": self.args.task,
            "policy": self.policy_ref.as_dict(),
            "num_envs": self.n,
            # Provenance: these sweeps roll each policy out against its own
            # training-time observations.py, so the layout and the exact source
            # blob are recorded rather than left implicit.
            "policy_obs_dim": getattr(self.args, "obs_dim", None),
            "observations_py_blob": _observations_blob(),
            "episodes": len(episodes),
            "nominal_full_grid_episodes": G.nominal_episode_count(self.args.seeds),
            "seeds": self.args.seeds,
            "base_seed": self.args.base_seed,
            "episode_steps": G.EPISODE_STEPS,
            "policy_dt": G.POLICY_DT,
            "settle_steps": {G.GAIT_NAMES[g]: G.GAIT_SETTLE_STEPS.get(g, G.DEFAULT_SETTLE_STEPS)
                             for g in G.GAIT_IDS},
            "grid": {"vx": G.VX_CMDS, "vy": G.VY_CMDS, "wz": G.WZ_CMDS},
            "fall_height_m": G.FALL_HEIGHT,
            "fall_height_grace_steps": G.FALL_HEIGHT_GRACE,
            "bad_orientation_limit_rad": G.BAD_ORIENTATION_LIMIT,
            "base_contact_force_n": G.BASE_CONTACT_FORCE,
            "domain_randomisation": bool(self.args.domain_rand),
            "push_robot": bool(self.args.push_robot),
            "observation_noise": bool(self.args.obs_noise),
            "deterministic_reset": bool(self.args.deterministic_reset),
            "video": bool(self.args.video),
        }
        (self.out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # -- one batch ---------------------------------------------------------

    def _prepare_batch(self, batch: list[G.Episode]):
        n_active = len(batch)
        gait_ids = torch.full((self.n,), G.STAND_GAIT_ID, dtype=torch.long, device=self.device)
        vel_cmd = torch.zeros(self.n, 3, device=self.device)
        settle = torch.full((self.n,), G.DEFAULT_SETTLE_STEPS, dtype=torch.long, device=self.device)
        for i, ep in enumerate(batch):
            gait_ids[i] = ep.gait_id
            vel_cmd[i, 0] = ep.vx
            vel_cmd[i, 1] = ep.vy
            vel_cmd[i, 2] = ep.wz
            settle[i] = ep.settle_steps
        active = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        active[:n_active] = True
        return gait_ids, vel_cmd, settle, active

    def _run_batch(self, batch: list[G.Episode], batch_idx: int, n_batches: int):
        n_active = len(batch)
        gait_ids, vel_cmd, settle, active = self._prepare_batch(batch)
        max_settle = int(settle[:n_active].max().item())
        total_steps = G.EPISODE_STEPS + max_settle

        # -- video subject --------------------------------------------------
        film_slot = -1
        if self.video is not None and (batch_idx % self.args.video_stride == 0):
            film_slot = self.selector.pick(batch)
            if self.u.viewport_camera_controller is not None:
                self.u.viewport_camera_controller.set_view_env_index(film_slot)
            self.video.begin_segment()

        # -- reset and pin the commands -------------------------------------
        with _muted(self.args.quiet_env):
            self.env.reset()
            force_commands(self.u, gait_ids, vel_cmd)
            prime_planner(self.u, gait_ids)
            obs = _get_obs(self.env)
            # The planner buffers are created lazily by the first observation;
            # prime again so episode 1 of the process starts as cleanly as the rest.
            prime_planner(self.u, gait_ids)
            obs = _get_obs(self.env)

        robot = self.refs.robot
        # Drift is measured over the scored window, so the origin is re-latched
        # at the first scored step of each episode rather than at the reset pose.
        start_xy = robot.data.root_pos_w[:, :2].clone()
        start_yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w).clone()
        last_xy = start_xy.clone()
        last_yaw = start_yaw.clone()

        alive = active.clone()
        survival_steps = torch.zeros(self.n, dtype=torch.long, device=self.device)
        low_height = torch.zeros(self.n, dtype=torch.long, device=self.device)
        reason = ["not_run"] * self.n
        for i in range(n_active):
            reason[i] = "running"

        accum = BatchAccumulator(self.n, self.device)

        for step in range(total_steps):
            with torch.inference_mode():
                actions = self.policy(obs)
            with _muted(self.args.quiet_env):
                obs, _, _, _ = self.env.step(actions)
                obs = _as_policy_obs(obs)
                # the command manager runs inside step(); re-pin afterwards
                force_commands(self.u, gait_ids, vel_cmd)

            lin_vel_b = robot.data.root_lin_vel_b
            ang_vel_b = robot.data.root_ang_vel_b
            height = robot.data.root_pos_w[:, 2]
            grav_b = robot.data.projected_gravity_b
            foot_contact = self.refs.foot_contact()
            des_contact = scheduled_contact(self.u, self.n, self.device)
            torque_norm = torch.norm(robot.data.applied_torque, dim=-1)

            scored = alive & (step >= settle)

            first_scored = active & (step == settle)
            if first_scored.any():
                start_xy[first_scored] = robot.data.root_pos_w[first_scored][:, :2]
                start_yaw[first_scored] = _yaw_from_quat_wxyz(robot.data.root_quat_w)[first_scored]
                last_xy[first_scored] = start_xy[first_scored]
                last_yaw[first_scored] = start_yaw[first_scored]

            # -- termination checks (only once the settle window has passed) --
            below = height < G.FALL_HEIGHT
            low_height = torch.where(below & scored, low_height + 1, torch.zeros_like(low_height))
            fell = low_height >= G.FALL_HEIGHT_GRACE
            bad_ori = orientation_angle_from_projected_gravity(grav_b) > G.BAD_ORIENTATION_LIMIT
            base_hit = self.refs.base_contact_force() > G.BASE_CONTACT_FORCE

            # precedence: base contact, then attitude, then sustained low height
            for label, cond in (
                ("base_contact", base_hit),
                ("bad_orientation", bad_ori),
                ("fall_height", fell),
            ):
                newly = scored & cond & alive
                if newly.any():
                    idx = torch.nonzero(newly, as_tuple=False).flatten().tolist()
                    for i in idx:
                        reason[i] = label
                        survival_steps[i] = step - int(settle[i].item())
                    alive[newly] = False

            # -- metrics ------------------------------------------------------
            scored = alive & (step >= settle)
            if scored.any():
                accum.record(
                    mask=scored,
                    lin_vel_b=lin_vel_b,
                    ang_vel_b=ang_vel_b,
                    vel_cmd=vel_cmd,
                    height=height,
                    projected_gravity=grav_b,
                    contact=foot_contact,
                    des_contact=des_contact,
                    torque_norm=torque_norm,
                    foot_pos_b=self.refs.foot_pos_b(),
                    p_ref_b=getattr(self.u, "beta_p_ref_B", None),
                    hip_base_b=getattr(self.u, "beta_hip_base_B", None),
                )
                last_xy[scored] = robot.data.root_pos_w[scored][:, :2]
                last_yaw[scored] = _yaw_from_quat_wxyz(robot.data.root_quat_w)[scored]

            # -- video --------------------------------------------------------
            if film_slot >= 0 and step % self.args.video_every == 0:
                self._film(batch, film_slot, batch_idx, n_batches, step, settle, alive, reason)

            # -- episode end ---------------------------------------------------
            done_now = alive & (step >= settle + G.EPISODE_STEPS - 1)
            if done_now.any():
                for i in torch.nonzero(done_now, as_tuple=False).flatten().tolist():
                    reason[i] = "timeout"
                    survival_steps[i] = G.EPISODE_STEPS
                alive[done_now] = False

            if not bool(alive[:n_active].any()):
                break

        if film_slot >= 0 and self.video is not None:
            self.video.end_segment(
                episode_id=self._episode_ids[film_slot],
                episode=batch[film_slot],
                batch=batch_idx,
                env_slot=film_slot,
            )

        drift = last_xy - start_xy
        yaw_drift = _wrap_angle(last_yaw - start_yaw)
        return accum.get(), reason, survival_steps, drift, yaw_drift

    def _film(self, batch, slot, batch_idx, n_batches, step, settle, alive, reason):
        ep = batch[slot]
        s = int(settle[slot].item())
        t = (step - s) * G.POLICY_DT
        phase = "settle" if step < s else f"t={t:5.2f}s"
        status = "alive" if bool(alive[slot]) else reason[slot]
        header = (
            f"batch {batch_idx + 1}/{n_batches}  |  seed {ep.seed}  |  "
            f"gait {ep.gait_name.upper()}  |  {phase}  |  {status}"
        )
        lines = [
            f"cmd  vx={ep.vx:+.2f} m/s   vy={ep.vy:+.2f} m/s   wz={ep.wz:+.2f} rad/s",
            f"{self.policy_ref.experiment} / {self.policy_ref.run} / "
            f"{self.policy_ref.checkpoint.name}   [isaac]",
        ]
        try:
            frame = self.u.render()
        except Exception as exc:  # pragma: no cover
            print(f"[sweep] render failed, disabling video: {exc}")
            self.video.close()
            self.video = None
            return
        self.video.add_frame(frame, header=header, lines=lines)

    # -- driver ------------------------------------------------------------

    def run(self) -> None:
        args = self.args
        episodes = G.build_episodes(seeds=args.seeds, base_seed=args.base_seed, gaits=args.gaits)

        # Episodes are grouped by seed so a seed is always evaluated as a whole,
        # and shuffled inside each seed so a command cell lands on a different
        # environment slot - and therefore a differently randomised robot - per
        # seed. The shuffle happens before the resume slice so that resuming
        # continues the same ordering instead of re-running arbitrary cells.
        by_seed: dict[int, list[G.Episode]] = {}
        for ep in episodes:
            by_seed.setdefault(ep.seed, []).append(ep)
        ordered: list[G.Episode] = []
        for seed, seed_eps in by_seed.items():
            seed_eps = list(seed_eps)
            random.Random(seed).shuffle(seed_eps)
            ordered.extend(seed_eps)

        self._write_config(ordered)

        fh, writer, existing = self._open_csv()
        if existing:
            print(f"[sweep] resuming: {existing} episodes already in {self.csv_path}")
            ordered = ordered[existing:]

        remaining: dict[int, list[G.Episode]] = {}
        for ep in ordered:
            remaining.setdefault(ep.seed, []).append(ep)

        n_batches = sum((len(v) + self.n - 1) // self.n for v in remaining.values())
        print(f"\n{'=' * 72}")
        print("  ISAAC LAB ROBUSTNESS SWEEP")
        print(f"  policy    : {self.policy_ref.tag}")
        print(f"  episodes  : {len(ordered)} in {n_batches} batches of {self.n} parallel envs")
        print(f"  output    : {self.out_dir}")
        if self.video:
            print(f"  video     : {self.video.path} (1 episode per "
                  f"{args.video_stride} batch(es), {args.video_fps} fps)")
        print(f"{'=' * 72}\n")
        print(G.grid_summary(seeds=args.seeds, gaits=args.gaits))
        print()

        pbar = tqdm(total=n_batches, unit="batch") if tqdm else None
        episode_id = existing
        n_done = n_survived = 0
        batch_idx = 0

        for seed, seed_eps in remaining.items():
            # Seeding is what makes the repetitions independent: it drives the
            # reset randomisation of every episode evaluated under this seed.
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            for _, batch in _chunks(seed_eps, self.n):
                self._episode_ids = [episode_id + i for i in range(len(batch))]
                t0 = time.time()
                try:
                    metrics, reason, survival_steps, drift, yaw_drift = self._run_batch(
                        batch, batch_idx, n_batches
                    )
                    wall = time.time() - t0
                    for i, ep in enumerate(batch):
                        survived = reason[i] == "timeout"
                        n_survived += int(survived)
                        n_done += 1
                        writer.writerow(make_row(
                            episode_id=episode_id,
                            episode=ep,
                            provenance=self.provenance,
                            metrics=metrics,
                            index=i,
                            survived=survived,
                            survival_steps=int(survival_steps[i].item()),
                            termination_reason=reason[i],
                            drift_xy=(float(drift[i, 0]), float(drift[i, 1])),
                            yaw_drift=float(yaw_drift[i]),
                            wall_time_s=wall / max(len(batch), 1),
                        ))
                        episode_id += 1
                except Exception as exc:
                    wall = time.time() - t0
                    print(f"\n[ERROR] batch {batch_idx} failed: {type(exc).__name__}: {exc}")
                    traceback.print_exc()
                    for ep in batch:
                        writer.writerow(make_error_row(
                            episode_id=episode_id,
                            episode=ep,
                            provenance=self.provenance,
                            message=f"{type(exc).__name__}: {exc}",
                            wall_time_s=wall / max(len(batch), 1),
                        ))
                        episode_id += 1
                        n_done += 1
                fh.flush()

                batch_idx += 1
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"seed={seed} survival={100 * n_survived / max(n_done, 1):.1f}%"
                    )

        if pbar:
            pbar.close()
        fh.close()
        if self.video:
            self.video.close()

        print(f"\n{'=' * 72}")
        print("  SWEEP COMPLETE")
        print(f"  episodes : {n_done}")
        print(f"  survival : {100 * n_survived / max(n_done, 1):.1f}%")
        print(f"  episodes csv : {self.csv_path}")
        if self.video:
            print(f"  video        : {self.video.path}  "
                  f"({self.video.duration_s / 60:.1f} min, {self.video.n_segments} segments)")
            print(f"  manifest     : {self.video.manifest_path}")
        print(f"{'=' * 72}\n")
        # Isaac Sim tears the process down with os._exit, which skips the
        # flush of Python's buffered stdout - force it here.
        sys.stdout.flush()
