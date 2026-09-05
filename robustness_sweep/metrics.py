"""Per-episode metric accumulation and the CSV schema of the robustness sweep.

Everything is accumulated with running sums over a batch of ``n`` episodes that
are simulated in parallel, so memory does not grow with the episode length. All
per-step quantities are masked twice:

* ``settle``  - the first ``settle_steps`` control steps of an episode are excluded
* ``alive``   - once an episode has terminated its values stop contributing

Both backends (Isaac Lab and MuJoCo) feed the same accumulator with torch
tensors, which keeps the two CSVs byte-for-byte comparable.
"""

from __future__ import annotations

import math

import torch

from robustness_sweep.grid import FOOT_ORDER, POLICY_DT

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

CSV_FIELDS: list[str] = [
    # provenance
    "episode_id", "simulator", "experiment", "run", "checkpoint", "seed",
    # command cell
    "gait_id", "gait_name", "vx_cmd", "vy_cmd", "wz_cmd", "zero_command",
    # survival
    "survived", "survival_steps", "survival_time_s", "termination_reason",
    # velocity tracking: achieved values and mean absolute errors
    "mean_vx_actual", "mean_vy_actual", "mean_wz_actual",
    "mean_vx_error", "mean_vy_error", "mean_wz_error",
    # base height
    "mean_height", "std_height",
    # attitude, derived from projected gravity
    "mean_roll", "mean_pitch", "mean_abs_roll", "mean_abs_pitch",
    # Raibert spatial tracking, sampled at touchdown
    "n_touchdowns", "place_err_x_signed", "place_err_x", "place_err_y",
    "place_err_xy", "ref_dx_at_touchdown",
    # per-foot contact fraction
    *[f"contact_frac_{f}" for f in FOOT_ORDER], "mean_contact_frac",
    # gait contact accuracy against the Raibert schedule
    *[f"contact_acc_{f}" for f in FOOT_ORDER],
    "gait_contact_accuracy", "gait_contact_accuracy_all_feet",
    # stability of zero-command episodes
    "xy_drift_m", "drift_x_m", "drift_y_m", "yaw_drift_rad",
    # extras
    "mean_torque_norm", "n_metric_steps", "settle_steps", "wall_time_s",
]

_NAN = float("nan")


# ---------------------------------------------------------------------------
# Attitude from projected gravity
# ---------------------------------------------------------------------------


def roll_pitch_from_projected_gravity(g_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll and pitch (rad) from the gravity vector expressed in the base frame.

    ``g_b`` is the unit gravity direction in body coordinates, i.e. ``(0, 0, -1)``
    for a level base (Isaac Lab's ``root_physx_view`` convention, also used by
    the policy observation). Returns ``(roll, pitch)``.
    """
    gx, gy, gz = g_b[..., 0], g_b[..., 1], g_b[..., 2]
    roll = torch.atan2(-gy, -gz)
    pitch = torch.atan2(gx, torch.sqrt(gy * gy + gz * gz))
    return roll, pitch


def orientation_angle_from_projected_gravity(g_b: torch.Tensor) -> torch.Tensor:
    """Angle (rad) between the base -z axis and gravity; matches ``mdp.bad_orientation``."""
    return torch.acos(torch.clamp(-g_b[..., 2], -1.0, 1.0)).abs()


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------


class BatchAccumulator:
    """Running per-episode statistics for ``n`` episodes simulated in parallel."""

    def __init__(self, n: int, device: torch.device | str):
        self.n = n
        self.device = device
        self.reset()

    def _z(self, *shape) -> torch.Tensor:
        return torch.zeros(self.n, *shape, device=self.device, dtype=torch.float64)

    def reset(self) -> None:
        self.count = self._z()               # scored control steps per episode
        self.sum_vel = self._z(3)            # vx, vy, wz achieved
        self.sum_err = self._z(3)            # |achieved - commanded|
        self.sum_h = self._z()
        self.sum_h2 = self._z()
        self.sum_rp = self._z(2)             # roll, pitch (signed)
        self.sum_abs_rp = self._z(2)
        self.sum_contact = self._z(4)        # per-foot contact indicator
        self.sum_match = self._z(4)          # per-foot agreement with the schedule
        self.sum_match_all = self._z()       # all four feet agree
        self.sum_torque = self._z()
        # -- Raibert foot-placement tracking, sampled at touchdown ------------
        # The gait accuracy above only checks contact TIMING. These check the
        # planner's spatial output: where the foot actually lands relative to
        # the reference the policy was given.
        self.count_td = self._z()            # touchdown events (summed over feet)
        self.sum_td_ex = self._z()           # signed  (foot_x - ref_x)
        self.sum_td_ax = self._z()           # |foot_x - ref_x|
        self.sum_td_ay = self._z()           # |foot_y - ref_y|
        self.sum_td_axy = self._z()          # ||(foot - ref)_xy||
        self.sum_td_refdx = self._z()        # signed (ref_x - hip_x): the stride the
                                             # planner asked for, AFTER the x_lim clamp
        self.prev_contact = None

    def record(
        self,
        *,
        mask: torch.Tensor,          # (n,) bool - step is scored for this episode
        lin_vel_b: torch.Tensor,     # (n,3) base linear velocity in body frame
        ang_vel_b: torch.Tensor,     # (n,3) base angular velocity in body frame
        vel_cmd: torch.Tensor,       # (n,3) commanded (vx, vy, wz)
        height: torch.Tensor,        # (n,)  base height above ground
        projected_gravity: torch.Tensor,  # (n,3) gravity direction in body frame
        contact: torch.Tensor,       # (n,4) measured foot contact (0/1), order FOOT_ORDER
        des_contact: torch.Tensor,   # (n,4) Raibert-scheduled contact (0/1)
        torque_norm: torch.Tensor,   # (n,)  ||applied joint torque||
        foot_pos_b: torch.Tensor | None = None,   # (n,4,3) measured foot pos, body frame
        p_ref_b: torch.Tensor | None = None,      # (n,4,3) Raibert reference, body frame
        hip_base_b: torch.Tensor | None = None,   # (n,4,3) static hip positions, body frame
    ) -> None:
        m = mask.to(dtype=torch.float64)
        m4 = m.unsqueeze(-1)

        achieved = torch.stack([lin_vel_b[:, 0], lin_vel_b[:, 1], ang_vel_b[:, 2]], dim=-1)
        achieved = achieved.to(torch.float64)
        cmd = vel_cmd.to(torch.float64)

        self.count += m
        self.sum_vel += achieved * m4
        self.sum_err += (achieved - cmd).abs() * m4

        h = height.to(torch.float64)
        self.sum_h += h * m
        self.sum_h2 += h * h * m

        roll, pitch = roll_pitch_from_projected_gravity(projected_gravity.to(torch.float64))
        rp = torch.stack([roll, pitch], dim=-1)
        self.sum_rp += rp * m4
        self.sum_abs_rp += rp.abs() * m4

        c = (contact > 0.5).to(torch.float64)
        d = (des_contact > 0.5).to(torch.float64)
        match = (c == d).to(torch.float64)
        self.sum_contact += c * m4
        self.sum_match += match * m4
        self.sum_match_all += (match.sum(dim=-1) == 4).to(torch.float64) * m
        self.sum_torque += torque_norm.to(torch.float64) * m

        # -- foot placement at touchdown --------------------------------------
        if foot_pos_b is not None and p_ref_b is not None:
            cb = contact > 0.5
            if self.prev_contact is None:
                self.prev_contact = torch.zeros_like(cb)
            touchdown = cb & (~self.prev_contact)          # swing -> stance edge
            self.prev_contact = cb.clone()

            w = touchdown.to(torch.float64) * m4           # (n,4), scored steps only
            d = (foot_pos_b - p_ref_b).to(torch.float64)
            ex, ey = d[..., 0], d[..., 1]
            self.count_td += w.sum(dim=-1)
            self.sum_td_ex += (ex * w).sum(dim=-1)
            self.sum_td_ax += (ex.abs() * w).sum(dim=-1)
            self.sum_td_ay += (ey.abs() * w).sum(dim=-1)
            self.sum_td_axy += (torch.sqrt(ex * ex + ey * ey) * w).sum(dim=-1)
            if hip_base_b is not None:
                refdx = (p_ref_b[..., 0] - hip_base_b[..., 0]).to(torch.float64)
                self.sum_td_refdx += (refdx * w).sum(dim=-1)

    # -- readout ------------------------------------------------------------

    def _mean(self, s: torch.Tensor) -> torch.Tensor:
        cnt = self.count.clone()
        if s.dim() == 2:
            cnt = cnt.unsqueeze(-1)
        return torch.where(cnt > 0, s / cnt.clamp(min=1.0), torch.full_like(s, _NAN))

    def get(self) -> dict[str, torch.Tensor]:
        mean_h = self._mean(self.sum_h)
        var_h = self._mean(self.sum_h2) - mean_h * mean_h
        std_h = torch.sqrt(var_h.clamp(min=0.0))
        return {
            "count": self.count,
            "vel": self._mean(self.sum_vel),
            "err": self._mean(self.sum_err),
            "mean_height": mean_h,
            "std_height": std_h,
            "rp": self._mean(self.sum_rp),
            "abs_rp": self._mean(self.sum_abs_rp),
            "contact": self._mean(self.sum_contact),
            "match": self._mean(self.sum_match),
            "match_all": self._mean(self.sum_match_all),
            "torque": self._mean(self.sum_torque),
            "n_touchdowns": self.count_td,
            "td_ex": self._mean_td(self.sum_td_ex),
            "td_ax": self._mean_td(self.sum_td_ax),
            "td_ay": self._mean_td(self.sum_td_ay),
            "td_axy": self._mean_td(self.sum_td_axy),
            "td_refdx": self._mean_td(self.sum_td_refdx),
        }

    def _mean_td(self, s: torch.Tensor) -> torch.Tensor:
        """Mean over touchdown events rather than over control steps."""
        return torch.where(self.count_td > 0, s / self.count_td.clamp(min=1.0),
                           torch.full_like(s, _NAN))


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def _f(x, nd: int = 4):
    v = float(x)
    if math.isnan(v) or math.isinf(v):
        return ""
    return round(v, nd)


def make_row(
    *,
    episode_id: int,
    episode,                      # robustness_sweep.grid.Episode
    provenance: dict,
    metrics: dict[str, torch.Tensor],
    index: int,
    survived: bool,
    survival_steps: int,
    termination_reason: str,
    drift_xy: tuple[float, float],
    yaw_drift: float,
    wall_time_s: float,
) -> dict:
    """Assemble one CSV row for episode ``index`` of a parallel batch."""
    vel = metrics["vel"][index]
    err = metrics["err"][index]
    rp = metrics["rp"][index]
    arp = metrics["abs_rp"][index]
    contact = metrics["contact"][index]
    match = metrics["match"][index]

    row = {
        "episode_id": episode_id,
        "simulator": provenance.get("simulator", ""),
        "experiment": provenance.get("experiment", ""),
        "run": provenance.get("run", ""),
        "checkpoint": provenance.get("checkpoint", ""),
        "seed": episode.seed,
        "gait_id": episode.gait_id,
        "gait_name": episode.gait_name,
        "vx_cmd": round(episode.vx, 3),
        "vy_cmd": round(episode.vy, 3),
        "wz_cmd": round(episode.wz, 3),
        "zero_command": int(episode.zero_command),
        "survived": int(survived),
        "survival_steps": int(survival_steps),
        "survival_time_s": round(int(survival_steps) * POLICY_DT, 3),
        "termination_reason": termination_reason,
        "mean_vx_actual": _f(vel[0]),
        "mean_vy_actual": _f(vel[1]),
        "mean_wz_actual": _f(vel[2]),
        "mean_vx_error": _f(err[0]),
        "mean_vy_error": _f(err[1]),
        "mean_wz_error": _f(err[2]),
        "mean_height": _f(metrics["mean_height"][index]),
        "std_height": _f(metrics["std_height"][index]),
        "mean_roll": _f(rp[0]),
        "mean_pitch": _f(rp[1]),
        "mean_abs_roll": _f(arp[0]),
        "mean_abs_pitch": _f(arp[1]),
        "mean_contact_frac": _f(contact.mean()),
        "gait_contact_accuracy": _f(match.mean()),
        "gait_contact_accuracy_all_feet": _f(metrics["match_all"][index]),
        "xy_drift_m": _f(math.hypot(*drift_xy)),
        "drift_x_m": _f(drift_xy[0]),
        "drift_y_m": _f(drift_xy[1]),
        "yaw_drift_rad": _f(yaw_drift),
        "mean_torque_norm": _f(metrics["torque"][index]),
        "n_touchdowns": int(metrics["n_touchdowns"][index].item()),
        "place_err_x_signed": _f(metrics["td_ex"][index]),
        "place_err_x": _f(metrics["td_ax"][index]),
        "place_err_y": _f(metrics["td_ay"][index]),
        "place_err_xy": _f(metrics["td_axy"][index]),
        "ref_dx_at_touchdown": _f(metrics["td_refdx"][index]),
        "n_metric_steps": int(metrics["count"][index].item()),
        "settle_steps": episode.settle_steps,
        "wall_time_s": round(wall_time_s, 3),
    }
    for j, foot in enumerate(FOOT_ORDER):
        row[f"contact_frac_{foot}"] = _f(contact[j])
        row[f"contact_acc_{foot}"] = _f(match[j])
    return row


def make_error_row(
    *,
    episode_id: int,
    episode,
    provenance: dict,
    message: str,
    wall_time_s: float,
) -> dict:
    """Placeholder row so a crashed batch still leaves a trace in the CSV."""
    row = {k: "" for k in CSV_FIELDS}
    row.update(
        episode_id=episode_id,
        simulator=provenance.get("simulator", ""),
        experiment=provenance.get("experiment", ""),
        run=provenance.get("run", ""),
        checkpoint=provenance.get("checkpoint", ""),
        seed=episode.seed,
        gait_id=episode.gait_id,
        gait_name=episode.gait_name,
        vx_cmd=round(episode.vx, 3),
        vy_cmd=round(episode.vy, 3),
        wz_cmd=round(episode.wz, 3),
        zero_command=int(episode.zero_command),
        survived=0,
        survival_steps=-1,
        survival_time_s=-1,
        termination_reason=f"error: {message}"[:120],
        settle_steps=episode.settle_steps,
        wall_time_s=round(wall_time_s, 3),
    )
    return row
