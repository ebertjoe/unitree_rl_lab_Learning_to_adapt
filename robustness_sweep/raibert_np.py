"""NumPy port of the Raibert gait planner used by the policy observation.

Line-for-line equivalent of ``beta_l_raibert`` in
``source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/observations.py``,
for a single robot. The MuJoCo backend needs it because it has to reproduce the
policy's observation - and because the scheduled contact state it returns is the
reference for the gait contact accuracy of §4.1.

The Isaac backend does *not* use this module: there the planner state is read
straight out of the environment, which rules out any drift between the two.
"""

from __future__ import annotations

import numpy as np

from robustness_sweep.grid import GAIT_CONFIGS, STAND_GAIT_ID

# Static hip reference positions in the base frame, order FR, FL, RR, RL.
# Mirrors ``get_go2_hip_positions_B``.
HIP_POS_B = np.array(
    [
        [0.183, -0.122, 0.0],
        [0.183, 0.122, 0.0],
        [-0.183, -0.122, 0.0],
        [-0.183, 0.122, 0.0],
    ],
    dtype=np.float64,
)

BLEND_ALPHA = 0.1
STEP_HEIGHT = 0.10
BOUND_FRONT_SCALE = 1.4
BOUND_GAIT_ID = 0


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R_WB (body -> world) from a (w, x, y, z) quaternion."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class RaibertPlanner:
    """Single-robot Raibert planner with the environment's blending and latching."""

    def __init__(self, gait_id: int):
        self.reset(gait_id)

    # -- gait parameters ---------------------------------------------------

    @staticmethod
    def _params(gait_id: int) -> dict:
        cfg = GAIT_CONFIGS[str(int(gait_id))]
        return {
            "period": float(cfg["period"]),
            "threshold": float(cfg["threshold"]),
            "offset": np.asarray(cfg["offset"], dtype=np.float64),
            "k": float(cfg["k"]),
            "z_nom": float(cfg["z_nom"]),
            "x_lim": float(cfg["x_lim"]),
            "y_lim": float(cfg["y_lim"]),
        }

    def reset(self, gait_id: int) -> None:
        """Restart at phase 0 with the target gait's parameters already blended in."""
        self.gait_id = int(gait_id)
        self.p = self._params(self.gait_id)
        self.period_blended = self.p["period"]
        self.znom_blended = self.p["z_nom"]
        self.p_ref_B = HIP_POS_B.copy()
        self.p_ref_B[:, 2] = self.znom_blended
        self.prev_c = np.ones(4, dtype=np.float64)

    # -- one control step --------------------------------------------------

    def step(
        self,
        t_exec: float,
        quat_wxyz: np.ndarray,
        lin_vel_w: np.ndarray,
        v_cmd: np.ndarray,
    ) -> dict:
        """Advance the planner by one control step.

        Args:
            t_exec: elapsed episode time in seconds (``episode_length_buf * step_dt``).
            quat_wxyz: base orientation as (w, x, y, z).
            lin_vel_w: base linear velocity in world coordinates.
            v_cmd: gait-conditioned command ``(vx, vy, wz)`` in the base frame.

        Returns:
            ``c_ref`` (4,) scheduled contact, ``p_ref_B`` (4,3) reference foot
            positions in the base frame and ``p_ref_rel_w`` (4,3) the same vector
            rotated into world axes - the quantity that enters the observation.
        """
        p = self.p

        # 4) smooth transition of period / nominal height
        self.period_blended = BLEND_ALPHA * p["period"] + (1.0 - BLEND_ALPHA) * self.period_blended
        self.znom_blended = BLEND_ALPHA * p["z_nom"] + (1.0 - BLEND_ALPHA) * self.znom_blended
        period = self.period_blended
        z_nominal = self.znom_blended
        threshold = p["threshold"]

        # 8) phase (the gait is fixed for the whole episode, so no compensation)
        global_phase = (t_exec % period) / period
        leg_phase = (global_phase + p["offset"]) % 1.0
        c_ref = (leg_phase < threshold).astype(np.float64)

        # 9) body-frame velocity
        R_WB = quat_wxyz_to_rotmat(quat_wxyz)
        v_B = R_WB.T @ np.asarray(lin_vel_w, dtype=np.float64)

        # 11) Raibert foot placement
        Tst = threshold * period
        dx = 0.5 * Tst * v_B[0] + p["k"] * (v_B[0] - float(v_cmd[0]))
        dy = 0.5 * Tst * v_B[1] + p["k"] * (v_B[1] - float(v_cmd[1]))

        per_leg_scale = np.ones(4, dtype=np.float64)
        if self.gait_id == BOUND_GAIT_ID:
            per_leg_scale[0:2] = BOUND_FRONT_SCALE  # FR, FL

        new_p = HIP_POS_B.copy()
        new_p[:, 0] += dx * per_leg_scale
        new_p[:, 1] += dy
        new_p[:, 2] = z_nominal

        # 12) clamp around the static hip position
        new_p[:, 0] = np.clip(
            new_p[:, 0], HIP_POS_B[:, 0] - p["x_lim"], HIP_POS_B[:, 0] + p["x_lim"]
        )
        new_p[:, 1] = np.clip(
            new_p[:, 1], HIP_POS_B[:, 1] - p["y_lim"], HIP_POS_B[:, 1] + p["y_lim"]
        )

        # 14) the reference is latched at lift-off (stance -> swing)
        liftoff = (self.prev_c > 0.5) & (c_ref < 0.5)
        self.p_ref_B[liftoff] = new_p[liftoff]
        self.prev_c = c_ref.copy()

        p_ref_B = self.p_ref_B.copy()

        # 16) swing height profile
        swing = (c_ref < 0.5).astype(np.float64)
        denom = max(1.0 - threshold, 1e-6)
        x = np.clip((leg_phase - threshold) / denom, 0.0, 1.0)
        z_swing = 0.5 * STEP_HEIGHT * (1.0 - np.cos(2.0 * np.pi * x)) * swing
        p_ref_B[:, 2] = z_nominal + z_swing

        # 17) into world axes (still relative to the base origin)
        p_ref_rel_w = (R_WB @ p_ref_B.T).T

        return {"c_ref": c_ref, "p_ref_B": p_ref_B, "p_ref_rel_w": p_ref_rel_w}


def gait_conditioned_velocity(gait_id: int, v_cmd) -> np.ndarray:
    """``stand`` is always evaluated at zero velocity, as in the policy observation."""
    v = np.asarray(v_cmd, dtype=np.float64).copy()
    if int(gait_id) == STAND_GAIT_ID:
        v[:] = 0.0
    return v
