"""MuJoCo (sim-to-sim) backend of the §4.1 robustness sweep.

The policy is loaded from the TorchScript export of the run
(``<run>/exported/policy.pt``), which already contains the empirical observation
normaliser, so nothing from Isaac Lab is needed here.

What is reproduced from the training environment
------------------------------------------------
* joint order ``FR, FL, RR, RL`` x ``hip, thigh, calf`` and the default joint pose
  of ``UNITREE_GO2_CFG``
* the joint-position action: ``q_des = q_default + 0.25 * action``
* the Go2 HV actuator: PD (kp 25, kd 0.5) recomputed every physics step, clipped
  by the torque-speed curve (Y1 20.2 / Y2 23.4 N m, knee 13.5 rad/s, no-load 30 rad/s)
* the 53-dimensional policy observation, including the Raibert planner outputs
  (see :mod:`robustness_sweep.raibert_np`)
* the termination criteria, settle windows and metrics of the Isaac backend

Known sim-to-sim differences (they are the point of running both simulators, but
they are listed here so they are not mistaken for harness bugs): the MJCF carries
its own joint armature and damping, contact solver parameters differ, and the
start-up mass/friction randomisation of Isaac Lab has no equivalent here unless
``--domain_rand`` is given.
"""

from __future__ import annotations

import csv
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from robustness_sweep import grid as G
from robustness_sweep.metrics import (
    CSV_FIELDS,
    make_error_row,
    make_row,
    roll_pitch_from_projected_gravity,
)
from robustness_sweep.raibert_np import (
    RaibertPlanner,
    gait_conditioned_velocity,
    quat_wxyz_to_rotmat,
)
from robustness_sweep.video import SweepVideoWriter

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

# ---------------------------------------------------------------------------
# Robot constants, mirrored from UNITREE_GO2_CFG / ActionsCfg
# ---------------------------------------------------------------------------

JOINT_ORDER = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

# Order of the 12 policy *actions*. ``JointPositionActionCfg`` leaves
# ``preserve_order`` at its default of False, so Isaac Lab resolves the joint
# names into the articulation's own ordering rather than the order they are
# listed in. The Go2 USD groups its joints by type, which means action index 0
# drives FL_hip and not FR_hip. Verified against the running environment
# (``action_manager._terms["JointPositionAction"]._joint_names``); the policy was
# trained with this mapping, so the sim-to-sim rollout has to reproduce it.
ACTION_JOINT_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]

# ACTION_TO_JOINT[i] is the action index that drives JOINT_ORDER[i].
ACTION_TO_JOINT = np.array([ACTION_JOINT_ORDER.index(n) for n in JOINT_ORDER])

DEFAULT_JOINT_POS = np.array(
    [
        -0.1, 0.8, -1.5,   # FR: hip, thigh, calf
        +0.1, 0.8, -1.5,   # FL
        -0.1, 1.0, -1.5,   # RR
        +0.1, 1.0, -1.5,   # RL
    ],
    dtype=np.float64,
)

ACTION_SCALE = 0.25
KP = 25.0
KD = 0.5
SIM_DT = 0.002
DECIMATION = 5
INIT_HEIGHT = 0.4

# Go2 HV torque-speed curve (UnitreeActuatorCfg_Go2HV)
TORQUE_Y1 = 20.2   # torque and speed in the same direction
TORQUE_Y2 = 23.4   # torque and speed in opposite directions
VEL_X1 = 13.5      # knee point of the T-N curve
VEL_X2 = 30.0      # no-load speed

CONTACT_FORCE_THRESHOLD = 1.0  # N, matches ContactSensorCfg's force threshold


def clip_effort(effort: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
    """Torque-speed clipping of :class:`UnitreeActuator._clip_effort`."""
    same_direction = (joint_vel * effort) > 0
    max_effort = np.where(same_direction, TORQUE_Y1, TORQUE_Y2)
    k = -max_effort / (VEL_X2 - VEL_X1)
    tapered = np.clip(k * (np.abs(joint_vel) - VEL_X1) + max_effort, 0.0, None)
    max_effort = np.where(np.abs(joint_vel) < VEL_X1, max_effort, tapered)
    return np.clip(effort, -max_effort, max_effort)


# ---------------------------------------------------------------------------
# Simulator wrapper
# ---------------------------------------------------------------------------


class Go2MujocoSim:
    """Minimal Go2 harness: PD control, contacts, and the policy observation."""

    def __init__(
        self,
        mjcf_path: str,
        joint_friction: float = 0.01,
        joint_damping: float = 0.0,
        joint_armature: float = 0.0,
        ground_friction: float | None = 1.0,
        flatten_scene: bool = True,
        action_joint_order: list[str] | None = None,
        sim_dt: float = SIM_DT,
        decimation: int = DECIMATION,
    ):
        import mujoco  # noqa: F401 - imported here so the module stays importable

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.model.opt.timestep = sim_dt
        self.decimation = decimation
        self.data = mujoco.MjData(self.model)

        mj = mujoco.mjtObj
        # -- joints and actuators ------------------------------------------
        self.joint_ids, self.qpos_adr, self.dof_adr, self.act_ids = [], [], [], []
        for name in JOINT_ORDER:
            jid = mujoco.mj_name2id(self.model, mj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"joint '{name}' not found in {mjcf_path}")
            self.joint_ids.append(jid)
            self.qpos_adr.append(int(self.model.jnt_qposadr[jid]))
            self.dof_adr.append(int(self.model.jnt_dofadr[jid]))
            aid = int(np.flatnonzero(self.model.actuator_trnid[:, 0] == jid)[0]) \
                if np.any(self.model.actuator_trnid[:, 0] == jid) else -1
            if aid < 0:
                raise RuntimeError(f"no actuator drives joint '{name}' in {mjcf_path}")
            self.act_ids.append(aid)
        self.qpos_adr = np.asarray(self.qpos_adr)
        self.dof_adr = np.asarray(self.dof_adr)
        self.act_ids = np.asarray(self.act_ids)

        # -- free joint ------------------------------------------------------
        free = [j for j in range(self.model.njnt) if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise RuntimeError(f"expected exactly one free joint, found {len(free)}")
        self.free_qpos = int(self.model.jnt_qposadr[free[0]])
        self.free_dof = int(self.model.jnt_dofadr[free[0]])
        self.base_body = int(self.model.jnt_bodyid[free[0]])

        # -- feet and base geoms ---------------------------------------------
        self.foot_geoms = [self._find_foot_geom(leg) for leg in G.FOOT_ORDER]
        self.base_geoms = [
            g for g in range(self.model.ngeom)
            if int(self.model.geom_bodyid[g]) == self.base_body and int(self.model.geom_contype[g]) != 0
        ]
        if not self.base_geoms:
            raise RuntimeError("the base body has no collision geom; base_contact cannot be detected")

        # -- parity with the Isaac articulation --------------------------------
        # ``DelayedPDActuator`` is an explicit actuator: Isaac Lab leaves the
        # simulated joint damping at zero and runs the PD in software, and the
        # Go2 HV configuration sets no armature. Menagerie-style MJCFs ship their
        # own damping / armature / frictionloss, which would otherwise stack on
        # top of the PD and make the two simulators incomparable.
        self.model.dof_frictionloss[self.dof_adr] = joint_friction
        self.model.dof_damping[self.dof_adr] = joint_damping
        self.model.dof_armature[self.dof_adr] = joint_armature

        # -- flatten the scene -------------------------------------------------
        # §4.1 is swept on flat ground; scene files often add an obstacle course
        # around the robot. Their contacts are disabled and they are made
        # invisible instead of being edited out of the XML.
        self.disabled_geoms: list[int] = []
        if flatten_scene:
            for g in range(self.model.ngeom):
                if int(self.model.geom_bodyid[g]) != 0:
                    continue
                if int(self.model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
                    continue
                self.model.geom_contype[g] = 0
                self.model.geom_conaffinity[g] = 0
                self.model.geom_rgba[g, 3] = 0.0
                self.disabled_geoms.append(g)

        if ground_friction is not None:
            floor = mujoco.mj_name2id(self.model, mj.mjOBJ_GEOM, "floor")
            if floor < 0:
                # fall back to the first plane geom
                planes = np.flatnonzero(self.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
                floor = int(planes[0]) if planes.size else -1
            if floor >= 0:
                self.model.geom_friction[floor, 0] = ground_friction

        order = list(action_joint_order or ACTION_JOINT_ORDER)
        if sorted(order) != sorted(JOINT_ORDER):
            raise RuntimeError("action_joint_order must be a permutation of the 12 leg joints")
        self.action_joint_order = order
        self.action_to_joint = np.array([order.index(n) for n in JOINT_ORDER])

        self._force6 = np.zeros(6, dtype=np.float64)

    # -- name resolution ----------------------------------------------------

    def _find_foot_geom(self, leg: str) -> int:
        mujoco = self.mujoco
        for candidate in (leg, f"{leg}_foot", f"{leg}_foot_geom"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, candidate)
            if gid >= 0:
                return int(gid)
        # last resort: the lowest collision geom on the calf body
        calf = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf")
        if calf >= 0:
            cands = [
                g for g in range(self.model.ngeom)
                if int(self.model.geom_bodyid[g]) == calf and int(self.model.geom_contype[g]) != 0
            ]
            if cands:
                return int(min(cands, key=lambda g: self.model.geom_pos[g][2]))
        raise RuntimeError(f"cannot locate the foot geom of leg '{leg}'")

    # -- state accessors ----------------------------------------------------

    @property
    def base_pos(self) -> np.ndarray:
        return self.data.qpos[self.free_qpos : self.free_qpos + 3].copy()

    @property
    def base_quat(self) -> np.ndarray:
        return self.data.qpos[self.free_qpos + 3 : self.free_qpos + 7].copy()

    @property
    def base_lin_vel_w(self) -> np.ndarray:
        return self.data.qvel[self.free_dof : self.free_dof + 3].copy()

    @property
    def base_ang_vel_b(self) -> np.ndarray:
        # MuJoCo stores a free joint's angular velocity in the body frame
        return self.data.qvel[self.free_dof + 3 : self.free_dof + 6].copy()

    @property
    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    @property
    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self.dof_adr].copy()

    def projected_gravity(self) -> np.ndarray:
        R = quat_wxyz_to_rotmat(self.base_quat)
        return R.T @ np.array([0.0, 0.0, -1.0])

    def contact_forces(self) -> tuple[np.ndarray, float]:
        """Per-foot contact-force magnitude (order FOOT_ORDER) and the base force."""
        foot = np.zeros(4, dtype=np.float64)
        base = 0.0
        foot_lookup = {g: i for i, g in enumerate(self.foot_geoms)}
        base_set = set(self.base_geoms)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            self.mujoco.mj_contactForce(self.model, self.data, i, self._force6)
            mag = float(np.linalg.norm(self._force6[:3]))
            for g in (int(con.geom1), int(con.geom2)):
                if g in foot_lookup:
                    foot[foot_lookup[g]] += mag
                elif g in base_set:
                    base += mag
        return foot, base

    # -- control ------------------------------------------------------------

    def reset(self, rng: np.random.Generator, deterministic: bool = False) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        qpos = self.data.qpos
        qpos[self.free_qpos + 0] = rng.uniform(-0.5, 0.5)
        qpos[self.free_qpos + 1] = rng.uniform(-0.5, 0.5)
        qpos[self.free_qpos + 2] = INIT_HEIGHT
        qpos[self.free_qpos + 3 : self.free_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        qpos[self.qpos_adr] = DEFAULT_JOINT_POS
        self.data.qvel[:] = 0.0
        if not deterministic:
            self.data.qvel[self.dof_adr] = rng.uniform(-1.0, 1.0, size=12)
        self.mujoco.mj_forward(self.model, self.data)

    def apply_action(self, action: np.ndarray) -> np.ndarray:
        """One control step: PD at every physics substep, returns the last torques."""
        action = np.asarray(action, dtype=np.float64)[self.action_to_joint]
        q_des = DEFAULT_JOINT_POS + ACTION_SCALE * action
        tau = np.zeros(12)
        for _ in range(self.decimation):
            q = self.data.qpos[self.qpos_adr]
            qd = self.data.qvel[self.dof_adr]
            tau = clip_effort(KP * (q_des - q) - KD * qd, qd)
            self.data.ctrl[self.act_ids] = tau
            self.mujoco.mj_step(self.model, self.data)
        return tau

    def randomise(self, rng: np.random.Generator) -> None:
        """Coarse analogue of Isaac Lab's start-up randomisation."""
        self.model.body_mass[self.base_body] = self._nominal_base_mass + rng.uniform(-1.0, 3.0)
        friction = rng.uniform(0.4, 1.0)
        for g in self.foot_geoms:
            self.model.geom_friction[g, 0] = friction

    def snapshot_nominal(self) -> None:
        self._nominal_base_mass = float(self.model.body_mass[self.base_body])


# ---------------------------------------------------------------------------
# Observation assembly (mirrors mdp.observations.robot_state_s)
# ---------------------------------------------------------------------------


def build_observation(
    sim: Go2MujocoSim,
    planner_out: dict,
    vel_cmd: np.ndarray,
    foot_contact: np.ndarray,
) -> np.ndarray:
    p_rel = planner_out["p_ref_rel_w"]
    obs = np.concatenate([
        sim.projected_gravity(),        # 3
        sim.joint_pos,                  # 12
        sim.base_ang_vel_b,             # 3
        sim.joint_vel,                  # 12
        vel_cmd,                        # 3
        foot_contact,                   # 4
        planner_out["c_ref"],           # 4  desFeetContact
        p_rel[:, 2],                    # 4  refFootZ
        p_rel[:, 0],                    # 4  refFootX
        p_rel[:, 1],                    # 4  refFootY
    ])
    return np.clip(obs, -100.0, 100.0)


OBS_DIM = 53


# ---------------------------------------------------------------------------
# Per-episode accumulator (numpy, same readout keys as metrics.BatchAccumulator)
# ---------------------------------------------------------------------------


class EpisodeAccumulator:
    def __init__(self):
        self.count = 0.0
        self.sum_vel = np.zeros(3)
        self.sum_err = np.zeros(3)
        self.sum_h = 0.0
        self.sum_h2 = 0.0
        self.sum_rp = np.zeros(2)
        self.sum_abs_rp = np.zeros(2)
        self.sum_contact = np.zeros(4)
        self.sum_match = np.zeros(4)
        self.sum_match_all = 0.0
        self.sum_torque = 0.0

    def record(self, *, lin_vel_b, ang_vel_b, vel_cmd, height, projected_gravity,
               contact, des_contact, torque_norm):
        achieved = np.array([lin_vel_b[0], lin_vel_b[1], ang_vel_b[2]])
        self.count += 1.0
        self.sum_vel += achieved
        self.sum_err += np.abs(achieved - np.asarray(vel_cmd))
        self.sum_h += height
        self.sum_h2 += height * height
        roll, pitch = roll_pitch_from_projected_gravity(torch.as_tensor(projected_gravity))
        rp = np.array([float(roll), float(pitch)])
        self.sum_rp += rp
        self.sum_abs_rp += np.abs(rp)
        c = (np.asarray(contact) > 0.5).astype(np.float64)
        d = (np.asarray(des_contact) > 0.5).astype(np.float64)
        match = (c == d).astype(np.float64)
        self.sum_contact += c
        self.sum_match += match
        self.sum_match_all += float(match.sum() == 4)
        self.sum_torque += float(torque_norm)

    def get(self) -> dict:
        n = self.count
        nan = float("nan")
        if n <= 0:
            return {
                "count": np.array([0.0]),
                "vel": np.full((1, 3), nan), "err": np.full((1, 3), nan),
                "mean_height": np.array([nan]), "std_height": np.array([nan]),
                "rp": np.full((1, 2), nan), "abs_rp": np.full((1, 2), nan),
                "contact": np.full((1, 4), nan), "match": np.full((1, 4), nan),
                "match_all": np.array([nan]), "torque": np.array([nan]),
            }
        mean_h = self.sum_h / n
        var_h = max(self.sum_h2 / n - mean_h * mean_h, 0.0)
        return {
            "count": np.array([n]),
            "vel": (self.sum_vel / n)[None, :],
            "err": (self.sum_err / n)[None, :],
            "mean_height": np.array([mean_h]),
            "std_height": np.array([math.sqrt(var_h)]),
            "rp": (self.sum_rp / n)[None, :],
            "abs_rp": (self.sum_abs_rp / n)[None, :],
            "contact": (self.sum_contact / n)[None, :],
            "match": (self.sum_match / n)[None, :],
            "match_all": np.array([self.sum_match_all / n]),
            "torque": np.array([self.sum_torque / n]),
        }


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


class MujocoRobustnessSweep:
    def __init__(self, args, policy_ref, policy, sim: Go2MujocoSim):
        self.args = args
        self.policy_ref = policy_ref
        self.policy = policy
        self.sim = sim
        self.provenance = {
            "simulator": "mujoco",
            "experiment": policy_ref.experiment,
            "run": policy_ref.run,
            "checkpoint": policy_ref.checkpoint.name,
        }
        self.out_dir = Path(args.out_dir) if args.out_dir else policy_ref.output_dir("mujoco")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "episodes.csv"

        self.video: SweepVideoWriter | None = None
        self.renderer = None
        self.camera = None
        self._filmed: dict[int, int] = {}
        if args.video:
            self.video = SweepVideoWriter(
                self.out_dir / "video" / "sweep.mp4",
                fps=args.video_fps,
                manifest_path=self.out_dir / "video" / "manifest.csv",
            )
            self._setup_renderer()

    def _setup_renderer(self) -> None:
        mujoco = self.sim.mujoco
        w, h = self.args.video_resolution
        # The offscreen framebuffer defaults to 640x480 in most MJCFs; it has to
        # be at least as large as the frame we want before the renderer is built.
        vis = self.sim.model.vis.global_
        vis.offwidth = max(int(vis.offwidth), int(w))
        vis.offheight = max(int(vis.offheight), int(h))
        self.renderer = mujoco.Renderer(self.sim.model, height=h, width=w)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = self.args.video_distance
        self.camera.azimuth = self.args.video_azimuth
        self.camera.elevation = self.args.video_elevation

    # -- one episode -------------------------------------------------------

    def run_episode(self, ep: G.Episode, episode_id: int, film: bool) -> dict:
        """Roll out one command cell.

        The loop is written so that the Raibert schedule, the measured contact
        state and the base state that enter the metrics are all sampled at the
        *same* instant, which is what the Isaac backend records after its
        ``env.step``. Iteration ``k`` observes the state at ``t = k * dt``, scores
        the control step that just ended (``k - 1``) and then applies the action
        for the next one.
        """
        sim = self.sim
        rng = np.random.default_rng((ep.seed + 1) * 1_000_003 + episode_id)

        if self.args.domain_rand:
            sim.randomise(rng)
        sim.reset(rng, deterministic=self.args.deterministic_reset)

        planner = RaibertPlanner(ep.gait_id)
        vel_cmd_raw = np.array([ep.vx, ep.vy, ep.wz], dtype=np.float64)
        vel_cmd = gait_conditioned_velocity(ep.gait_id, vel_cmd_raw)

        settle = ep.settle_steps
        last_scored = settle + G.EPISODE_STEPS - 1
        accum = EpisodeAccumulator()

        start_xy = sim.base_pos[:2].copy()
        start_yaw = _yaw_from_quat(sim.base_quat)
        last_xy, last_yaw = start_xy.copy(), start_yaw

        reason = "running"
        survival_steps = 0
        low_height = 0
        prev_tau = np.zeros(12)

        if film:
            self.video.begin_segment()

        for step in range(last_scored + 2):
            # -- state at t = step * dt ------------------------------------
            planner_out = planner.step(step * G.POLICY_DT, sim.base_quat,
                                       sim.base_lin_vel_w, vel_cmd)
            foot_force, base_force = sim.contact_forces()
            measured_contact = (foot_force > CONTACT_FORCE_THRESHOLD).astype(np.float64)
            grav_b = sim.projected_gravity()
            height = float(sim.base_pos[2])
            R = quat_wxyz_to_rotmat(sim.base_quat)
            lin_vel_b = R.T @ sim.base_lin_vel_w
            ang_vel_b = sim.base_ang_vel_b

            # -- score the control step that just ended --------------------
            prev = step - 1
            if prev == settle:
                # latch the drift origin at the first scored step, so the drift
                # covers the scored window and not the settle phase
                start_xy = sim.base_pos[:2].copy()
                start_yaw = _yaw_from_quat(sim.base_quat)
                last_xy, last_yaw = start_xy.copy(), start_yaw
            if prev >= settle:
                low_height = low_height + 1 if height < G.FALL_HEIGHT else 0
                tilt = math.acos(max(min(-grav_b[2], 1.0), -1.0))
                if base_force > G.BASE_CONTACT_FORCE:
                    reason = "base_contact"
                elif abs(tilt) > G.BAD_ORIENTATION_LIMIT:
                    reason = "bad_orientation"
                elif low_height >= G.FALL_HEIGHT_GRACE:
                    reason = "fall_height"

                if reason == "running":
                    accum.record(
                        lin_vel_b=lin_vel_b,
                        ang_vel_b=ang_vel_b,
                        vel_cmd=vel_cmd_raw,
                        height=height,
                        projected_gravity=grav_b,
                        contact=measured_contact,
                        des_contact=planner_out["c_ref"],
                        torque_norm=float(np.linalg.norm(prev_tau)),
                    )
                    last_xy = sim.base_pos[:2].copy()
                    last_yaw = _yaw_from_quat(sim.base_quat)
                else:
                    survival_steps = prev - settle

            if film and step % self.args.video_every == 0:
                self._film(ep, episode_id, step, settle, reason)

            if reason != "running":
                break
            if prev >= last_scored:
                reason = "timeout"
                survival_steps = G.EPISODE_STEPS
                break

            # -- act -------------------------------------------------------
            obs = build_observation(sim, planner_out, vel_cmd, measured_contact)
            with torch.inference_mode():
                action = self.policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            prev_tau = sim.apply_action(action.squeeze(0).cpu().numpy())

        if film:
            self.video.end_segment(episode_id=episode_id, episode=ep, batch=episode_id, env_slot=0)

        metrics = accum.get()
        drift = (float(last_xy[0] - start_xy[0]), float(last_xy[1] - start_xy[1]))
        yaw_drift = float(_wrap(last_yaw - start_yaw))
        return {
            "metrics": metrics,
            "reason": reason,
            "survival_steps": survival_steps,
            "drift": drift,
            "yaw_drift": yaw_drift,
        }

    def _film(self, ep, episode_id, step, settle, reason) -> None:
        sim = self.sim
        self.camera.lookat[:] = sim.base_pos
        self.renderer.update_scene(sim.data, camera=self.camera)
        frame = self.renderer.render()
        t = (step - settle) * G.POLICY_DT
        phase = "settle" if step < settle else f"t={t:5.2f}s"
        header = (
            f"episode {episode_id}  |  seed {ep.seed}  |  "
            f"gait {ep.gait_name.upper()}  |  {phase}  |  "
            f"{'alive' if reason == 'running' else reason}"
        )
        lines = [
            f"cmd  vx={ep.vx:+.2f} m/s   vy={ep.vy:+.2f} m/s   wz={ep.wz:+.2f} rad/s",
            f"{self.policy_ref.experiment} / {self.policy_ref.run} / "
            f"{self.policy_ref.checkpoint.name}   [mujoco]",
        ]
        self.video.add_frame(frame, header=header, lines=lines)

    # -- driver ------------------------------------------------------------

    def _should_film(self, ep: G.Episode) -> bool:
        if self.video is None or ep.seed != self.args.base_seed:
            return False
        if self._filmed.get(ep.gait_id, 0) >= self.args.video_per_gait:
            return False
        self._filmed[ep.gait_id] = self._filmed.get(ep.gait_id, 0) + 1
        return True

    def _write_config(self, episodes) -> None:
        cfg = {
            "simulator": "mujoco",
            "mjcf": str(self.args.mjcf),
            "policy": self.policy_ref.as_dict(),
            "episodes": len(episodes),
            "nominal_full_grid_episodes": G.nominal_episode_count(self.args.seeds),
            "seeds": self.args.seeds,
            "base_seed": self.args.base_seed,
            "episode_steps": G.EPISODE_STEPS,
            "policy_dt": G.POLICY_DT,
            "sim_dt": self.sim.model.opt.timestep,
            "decimation": self.sim.decimation,
            "obs_dim": OBS_DIM,
            "grid": {"vx": G.VX_CMDS, "vy": G.VY_CMDS, "wz": G.WZ_CMDS},
            "fall_height_m": G.FALL_HEIGHT,
            "bad_orientation_limit_rad": G.BAD_ORIENTATION_LIMIT,
            "base_contact_force_n": G.BASE_CONTACT_FORCE,
            "domain_randomisation": bool(self.args.domain_rand),
            "deterministic_reset": bool(self.args.deterministic_reset),
            "video": bool(self.args.video),
        }
        (self.out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    def run(self) -> None:
        args = self.args
        episodes = G.build_episodes(seeds=args.seeds, base_seed=args.base_seed, gaits=args.gaits)
        self._write_config(episodes)

        existing = 0
        if self.csv_path.exists() and args.resume:
            with open(self.csv_path) as f:
                existing = max(sum(1 for _ in f) - 1, 0)
            fh = open(self.csv_path, "a", newline="")
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            print(f"[sweep] resuming: {existing} episodes already in {self.csv_path}")
            episodes = episodes[existing:]
        else:
            fh = open(self.csv_path, "w", newline="")
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()

        print(f"\n{'=' * 72}")
        print("  MUJOCO ROBUSTNESS SWEEP")
        print(f"  policy    : {self.policy_ref.tag}")
        print(f"  model     : {args.mjcf}")
        print(f"  episodes  : {len(episodes)}")
        print(f"  output    : {self.out_dir}")
        if self.video:
            print(f"  video     : {self.video.path} "
                  f"({args.video_per_gait} episodes per gait, {args.video_fps} fps)")
        print(f"{'=' * 72}\n")
        print(G.grid_summary(seeds=args.seeds, gaits=args.gaits))
        print()

        pbar = tqdm(total=len(episodes), unit="ep") if tqdm else None
        episode_id = existing
        n_done = n_survived = 0

        for ep in episodes:
            t0 = time.time()
            try:
                out = self.run_episode(ep, episode_id, self._should_film(ep))
                survived = out["reason"] == "timeout"
                n_survived += int(survived)
                writer.writerow(make_row(
                    episode_id=episode_id,
                    episode=ep,
                    provenance=self.provenance,
                    metrics=out["metrics"],
                    index=0,
                    survived=survived,
                    survival_steps=out["survival_steps"],
                    termination_reason=out["reason"],
                    drift_xy=out["drift"],
                    yaw_drift=out["yaw_drift"],
                    wall_time_s=time.time() - t0,
                ))
            except Exception as exc:
                print(f"\n[ERROR] episode {episode_id} ({ep.label}) failed: "
                      f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
                writer.writerow(make_error_row(
                    episode_id=episode_id,
                    episode=ep,
                    provenance=self.provenance,
                    message=f"{type(exc).__name__}: {exc}",
                    wall_time_s=time.time() - t0,
                ))
            episode_id += 1
            n_done += 1
            if n_done % 25 == 0:
                fh.flush()
            if pbar:
                pbar.update(1)
                pbar.set_postfix_str(
                    f"seed={ep.seed} survival={100 * n_survived / max(n_done, 1):.1f}%"
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
        print(f"{'=' * 72}\n")


def _yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
