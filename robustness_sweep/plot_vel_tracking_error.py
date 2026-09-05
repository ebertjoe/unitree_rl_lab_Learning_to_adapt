"""Velocity tracking error across five policies (custom comparison figure).

    2026-08-16_16-51-37   69-dim
    2026-08-16_22-05-59   68-dim
    2026-08-17_09-10-28   56-dim
    2026-09-02_14-22-00   53-dim with lin_vel in Gait Planner
    2026-08-17_20-51-38   53-dim without lin_vel in Gait Planner

Reuses the loading/reduction helpers from plot_degradation.py (seed-first
reduction, isaac_it14800 sweep = common training iteration model_14800.pt).

Usage
-----
    python -m robustness_sweep.plot_vel_tracking_error
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robustness_sweep import plot_degradation as pd_

RUNS: list[tuple[str, str, str]] = [
    ("2026-08-16_16-51-37", "69-dim", "69-dim"),
    ("2026-08-16_22-05-59", "68-dim", "68-dim"),
    ("2026-08-17_09-10-28", "56-dim", "56-dim"),
    ("2026-09-02_14-22-00", "53-dim₁", "53-dim, lin_vel in Gait Planner"),
    ("2026-08-17_20-51-38", "53-dim₂", "53-dim, no lin_vel in Gait Planner"),
]

METRICS = [
    ("mean_vx_error", r"$|e_{v_x}|$  [m/s]"),
    ("mean_vy_error", r"$|e_{v_y}|$  [m/s]"),
    ("mean_wz_error", r"$|e_{\omega_z}|$  [rad/s]"),
]

COLORS = [pd_.C_BLUE, pd_.C_ORANGE, pd_.C_AQUA, pd_.C_YELLOW, "#8c5ee8"]


def main() -> None:
    pd_.SWEEP_DIR = "isaac_it14800"
    pd_._mpl_style()

    data = {run: pd_.load_run(run) for run, _, _ in RUNS}

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.3))
    x = np.arange(len(RUNS))

    for ax, (key, label) in zip(axes, METRICS):
        mus, sds = [], []
        for run, _, _ in RUNS:
            mu, sd, _ = pd_.seed_stats(data[run], key)
            mus.append(mu)
            sds.append(sd if np.isfinite(sd) else 0.0)
        mus, sds = np.array(mus), np.array(sds)

        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.bar(x, mus, 0.62, color=COLORS, zorder=3)
        ax.errorbar(x, mus, yerr=sds, lw=0, ecolor=pd_.INK3,
                    elinewidth=1.1, capsize=3, capthick=1.1, zorder=4)
        ax.set_title(label, color=pd_.INK, pad=7)
        ax.set_xticks(x)
        ax.set_xticklabels([short for _r, short, _l in RUNS], fontsize=8.5)
        ax.set_xlim(-0.65, len(RUNS) - 0.35)
        ax.set_ylim(0, (mus + sds).max() * 1.25)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(handles, [full for _r, _s, full in RUNS], loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, 0.0), fontsize=7.6, handlelength=1.1,
               columnspacing=1.3, frameon=False)

    pd_.header(
        fig,
        "Velocity tracking error across the observation ladder and Raibert planner variants",
        "Isaac Lab sweep at a common training iteration (model_14800.pt), 4380 episodes per policy\n"
        "(8 gaits x 5x5x5 command grid x 5 seeds). Bars = mean of the five seed means, whisker = std over seeds.\n"
        "stand excluded from the averages.",
        top=0.76,
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.32, left=0.06, right=0.99)

    out = Path(__file__).resolve().parent / "figures"
    out.mkdir(exist_ok=True)
    pd_._save(fig, out / "vel_tracking_error_comparison", ["png", "pdf"])
    print(f"wrote {out / 'vel_tracking_error_comparison.png'}")


if __name__ == "__main__":
    main()
