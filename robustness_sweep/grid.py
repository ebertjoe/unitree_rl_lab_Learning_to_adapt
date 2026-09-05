"""Command grid, gait table and protocol constants of the §4.1 robustness sweep.

Table 4.1
---------
    vx_cmd  [m/s]    0.0, 0.3, 0.6, 0.9, 1.2
    vy_cmd  [m/s]   -0.4, -0.2, 0.0, 0.2, 0.4
    wz_cmd  [rad/s] -0.5, -0.25, 0.0, 0.25, 0.5
    gaits            bound, trot, hop, amble, pronk, limp, stand, run
    seeds            5

``stand`` is a special case: it is only evaluated at zero command, so the number
of episodes per simulator is

    (7 locomotion gaits * 5 * 5 * 5 + 1 stand) * 5 seeds = 876 * 5 = 4380

The nominal budget quoted in the thesis (5000 episodes = 8 * 125 * 5) counts the
full grid for ``stand`` as well; the harness reports both numbers so the deviation
is explicit rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

# ---------------------------------------------------------------------------
# Gaits
# ---------------------------------------------------------------------------

GAIT_NAMES: dict[int, str] = {
    0: "bound",
    1: "trot",
    2: "hop",
    3: "amble",
    4: "pronk",
    5: "limp",
    6: "stand",
    7: "run",
}
GAIT_IDS: list[int] = sorted(GAIT_NAMES)
NAME_TO_GAIT_ID: dict[str, int] = {v: k for k, v in GAIT_NAMES.items()}
STAND_GAIT_ID = 6

# Mirror of ``GAIT_CONFIGS`` in
# source/unitree_rl_lab/.../locomotion/robots/go2/velocity_env_cfg.py.
# The Isaac backend asserts that this table still matches the environment; the
# MuJoCo backend needs it because it re-implements the Raibert planner itself.
GAIT_CONFIGS: dict[str, dict] = {
    "0": {"name": "bound", "period": 0.4,  "threshold": 0.4,   "offset": [0.5, 0.5, 0.0, 0.0],
          "k": 0.03, "z_nom": -0.32, "x_lim": 0.12, "y_lim": 0.10},
    "1": {"name": "trot",  "period": 0.4,  "threshold": 0.5,   "offset": [0.0, 0.5, 0.5, 0.0],
          "k": 0.03, "z_nom": -0.32, "x_lim": 0.10, "y_lim": 0.10},
    "2": {"name": "hop",   "period": 0.3,  "threshold": 0.5,   "offset": [0.0, 0.0, 0.0, 0.0],
          "k": 0.03, "z_nom": -0.30, "x_lim": 0.15, "y_lim": 0.10},
    "3": {"name": "amble", "period": 0.5,  "threshold": 0.625, "offset": [0.0, 0.5, 0.25, 0.75],
          "k": 0.02, "z_nom": -0.32, "x_lim": 0.14, "y_lim": 0.12},
    "4": {"name": "pronk", "period": 0.5,  "threshold": 0.5,   "offset": [0.0, 0.0, 0.0, 0.0],
          "k": 0.01, "z_nom": -0.32, "x_lim": 0.08, "y_lim": 0.10},
    "5": {"name": "limp",  "period": 0.4,  "threshold": 0.5,   "offset": [0.5, 0.5, 0.5, 0.0],
          "k": 0.03, "z_nom": -0.32, "x_lim": 0.12, "y_lim": 0.10},
    "6": {"name": "stand", "period": 1.0,  "threshold": 1.0,   "offset": [0.0, 0.0, 0.0, 0.0],
          "k": 0.01, "z_nom": -0.32, "x_lim": 0.10, "y_lim": 0.10},
    "7": {"name": "run",   "period": 0.3,  "threshold": 0.4,   "offset": [0.0, 0.5, 0.5, 0.0],
          "k": 0.03, "z_nom": -0.32, "x_lim": 0.12, "y_lim": 0.10},
}

# ---------------------------------------------------------------------------
# Command grid (Table 4.1)
# ---------------------------------------------------------------------------

VX_CMDS: list[float] = [0.0, 0.3, 0.6, 0.9, 1.2]
VY_CMDS: list[float] = [-0.4, -0.2, 0.0, 0.2, 0.4]
WZ_CMDS: list[float] = [-0.5, -0.25, 0.0, 0.25, 0.5]
N_SEEDS = 5

# ---------------------------------------------------------------------------
# Protocol constants (§4.1)
# ---------------------------------------------------------------------------

POLICY_DT = 0.01          # decimation 5 * sim.dt 0.002
EPISODE_STEPS = 1000      # 10 s of scored simulation per episode

# Steps at the start of every episode that are excluded from the metrics and
# during which a termination is not declared. Hop and pronk need a longer
# window because their first flight phase starts from a standing pose.
DEFAULT_SETTLE_STEPS = 100
GAIT_SETTLE_STEPS: dict[int, int] = {2: 150, 4: 200}
MAX_SETTLE_STEPS = max([DEFAULT_SETTLE_STEPS, *GAIT_SETTLE_STEPS.values()])

# Termination criteria. ``base_contact`` and ``bad_orientation`` mirror the
# training environment's TerminationsCfg, ``fall_height`` is the additional
# physical-fall criterion shared with the MuJoCo protocol.
FALL_HEIGHT = 0.15               # m, base height
FALL_HEIGHT_GRACE = 10           # consecutive control steps below FALL_HEIGHT
BAD_ORIENTATION_LIMIT = 0.8      # rad, angle between base -z and gravity
BASE_CONTACT_FORCE = 1.0         # N, illegal-contact threshold on the base link

FOOT_ORDER: list[str] = ["FR", "FL", "RR", "RL"]

TERMINATION_REASONS = ("timeout", "fall_height", "base_contact", "bad_orientation", "error")


# ---------------------------------------------------------------------------
# Episode specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Episode:
    """One (gait, command, seed) cell of the sweep."""

    gait_id: int
    vx: float
    vy: float
    wz: float
    seed: int

    @property
    def gait_name(self) -> str:
        return GAIT_NAMES[self.gait_id]

    @property
    def is_stand(self) -> bool:
        return self.gait_id == STAND_GAIT_ID

    @property
    def zero_command(self) -> bool:
        """True when the commanded twist is zero (stand, or the (0,0,0) cell)."""
        return self.is_stand or (self.vx == 0.0 and self.vy == 0.0 and self.wz == 0.0)

    @property
    def settle_steps(self) -> int:
        return GAIT_SETTLE_STEPS.get(self.gait_id, DEFAULT_SETTLE_STEPS)

    @property
    def label(self) -> str:
        return (
            f"{self.gait_name:<5s} "
            f"vx={self.vx:+.2f} vy={self.vy:+.2f} wz={self.wz:+.2f} seed={self.seed}"
        )


def command_grid() -> list[tuple[float, float, float]]:
    """The full 5x5x5 velocity grid of Table 4.1, in a deterministic order."""
    return [tuple(c) for c in product(VX_CMDS, VY_CMDS, WZ_CMDS)]


def build_commands(gait_id: int) -> list[tuple[float, float, float]]:
    """Commands evaluated for one gait. ``stand`` only gets the zero command."""
    if gait_id == STAND_GAIT_ID:
        return [(0.0, 0.0, 0.0)]
    return command_grid()


def build_episodes(
    seeds: int = N_SEEDS,
    base_seed: int = 0,
    gaits: list[int] | None = None,
) -> list[Episode]:
    """Full episode list of the sweep, ordered seed-major then gait-major.

    Seed-major ordering means a partially completed sweep still contains whole
    seeds, which keeps the per-seed statistics usable after an interruption.
    """
    gaits = GAIT_IDS if gaits is None else gaits
    episodes: list[Episode] = []
    for s in range(seeds):
        for gait_id in gaits:
            for vx, vy, wz in build_commands(gait_id):
                episodes.append(Episode(gait_id, vx, vy, wz, base_seed + s))
    return episodes


def nominal_episode_count(seeds: int = N_SEEDS) -> int:
    """Episode budget if ``stand`` were swept over the full grid (5000 for 5 seeds)."""
    return len(GAIT_IDS) * len(command_grid()) * seeds


def grid_summary(seeds: int = N_SEEDS, gaits: list[int] | None = None) -> str:
    eps = build_episodes(seeds=seeds, gaits=gaits)
    combos = len(eps) // max(seeds, 1)
    return (
        f"gaits            : {', '.join(GAIT_NAMES[g] for g in (gaits or GAIT_IDS))}\n"
        f"vx [m/s]         : {VX_CMDS}\n"
        f"vy [m/s]         : {VY_CMDS}\n"
        f"wz [rad/s]       : {WZ_CMDS}\n"
        f"seeds            : {seeds}\n"
        f"command combos   : {combos} (stand only at zero command)\n"
        f"episodes         : {len(eps)}  (nominal full-grid budget "
        f"{nominal_episode_count(seeds)})\n"
        f"episode length   : {EPISODE_STEPS * POLICY_DT:.0f} s scored "
        f"(+ {DEFAULT_SETTLE_STEPS * POLICY_DT:.1f}-"
        f"{MAX_SETTLE_STEPS * POLICY_DT:.1f} s settle)"
    )
