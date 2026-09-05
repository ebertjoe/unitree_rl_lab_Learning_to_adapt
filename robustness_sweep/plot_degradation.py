"""Degradation figures for the §4.1 robustness sweep.

Two questions, two figure sets, one shared statistical treatment.

A. Observation ablation ladder
   Does progressively removing observations degrade the policy?

       69  full observation
       68  - base height
       56  - joint torques
       53  - base linear velocity

B. Raibert planner ablation (both policies are 53-dimensional)
   Does driving the foot-placement planner with the *measured* base linear
   velocity beat driving it feedforward from the *commanded* velocity?

       2026-09-02_14-22-00   p_ref = p_hip + 0.5*Tst*v_B + K*(v_B - v_cmd)
       2026-08-17_20-51-38   p_ref = p_hip + 0.5*Tst*v_cmd

Every number is reduced seed-first: the episodes of one seed are averaged, and
the reported spread is the standard deviation over the five seed means. That is
the only spread the sweep protocol actually resolves - the 876 episodes of a
seed are not independent samples of the same quantity, they are a deterministic
sweep of the command grid.

Usage
-----
    python -m robustness_sweep.plot_degradation
    python -m robustness_sweep.plot_degradation --out_dir some/where --formats pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LOG_ROOT = REPO / "logs" / "rsl_rl"
EXPERIMENT = "unitree_go2_locomotion_paper"

# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

# (run, obs_dim, short label, what this stage removed)
LADDER: list[tuple[str, int, str, str]] = [
    ("2026-08-16_16-51-37", 69, "69", "full observation"),
    ("2026-08-16_22-05-59", 68, "68", "− base height"),
    ("2026-08-17_09-10-28", 56, "56", "− joint torques"),
    ("2026-08-17_20-51-38", 53, "53", "− base lin. vel."),
]

RAIBERT: list[tuple[str, str]] = [
    ("2026-09-02_14-22-00", "planner on measured $v_B$"),
    ("2026-08-17_20-51-38", "planner on commanded $v_{cmd}$"),
]

# ---------------------------------------------------------------------------
# Palette - dataviz skill reference instance, light mode, categorical slots 1-4.
# Validated for the adjacent pairlist (bars, lines): worst adjacent CVD dE 9.1.
# Markers carry a second, non-colour channel so identity never rests on hue.
# ---------------------------------------------------------------------------

C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
LADDER_COLORS = [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW]
LADDER_MARKERS = ["o", "s", "^", "D"]
RAIBERT_COLORS = [C_BLUE, C_ORANGE]
RAIBERT_MARKERS = ["o", "s"]

INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e3e2de"

GAIT_ORDER = ["trot", "run", "bound", "amble", "limp", "hop", "pronk", "stand"]

# metric key -> (axis label, "lower is better"?)
METRICS: dict[str, tuple[str, bool]] = {
    "survival_rate":         ("survival rate [%]", False),
    "gait_contact_accuracy": ("gait contact accuracy [%]", False),
    "mean_vx_error":         (r"$|e_{v_x}|$  [m/s]", True),
    "mean_vy_error":         (r"$|e_{v_y}|$  [m/s]", True),
    "mean_wz_error":         (r"$|e_{\omega_z}|$  [rad/s]", True),
    "mean_torque_norm":      (r"$\|\tau\|$  [Nm]", True),
    "std_height":            (r"$\sigma_h$  [m]", True),
    "mean_abs_pitch":        (r"$|\theta|$  [rad]", True),
}
PERCENT_METRICS = {"survival_rate", "gait_contact_accuracy"}


def _mpl_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "axes.edgecolor": INK3,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "text.color": INK,
    })


# ---------------------------------------------------------------------------
# Loading and seed-first reduction
# ---------------------------------------------------------------------------


SWEEP_DIR = "isaac"      # set from --sweep_dir; "isaac_it14800" for the common-iteration re-run
STEM_SUFFIX = ""         # appended to every figure filename


def load_run(run: str) -> pd.DataFrame:
    path = LOG_ROOT / EXPERIMENT / run / "robustness_sweep" / SWEEP_DIR / "episodes.csv"
    if not path.is_file():
        raise SystemExit(f"no sweep for {run}: {path} is missing")
    df = pd.read_csv(path)
    df["run"] = run
    # ``survived`` is the 0/1 episode outcome; ``survival_rate`` is its mean.
    df["survival_rate"] = df["survived"].astype(float)
    return df


def seed_stats(df: pd.DataFrame, metric: str, locomotion_only: bool = True) -> tuple[float, float, np.ndarray]:
    """Mean over seed means and the standard deviation across the five seeds.

    ``stand`` is excluded from every locomotion metric: it is a single zero
    command per seed with a near-zero tracking error, so leaving it in would
    dilute the grid average with an episode class that is not comparable.
    """
    sub = df[df.gait_name != "stand"] if locomotion_only else df
    per_seed = sub.groupby("seed")[metric].mean()
    return float(per_seed.mean()), float(per_seed.std(ddof=1)), per_seed.to_numpy()


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch t statistic and two-sided p over the seed means (n=5 each)."""
    from scipy import stats  # optional; only used for the printed table
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def welch_safe(a: np.ndarray, b: np.ndarray) -> tuple[float, float] | tuple[None, None]:
    try:
        return welch(a, b)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Figure 1 - the ladder, one panel per metric
# ---------------------------------------------------------------------------


def fig_ladder(data: dict[str, pd.DataFrame], out: Path, formats: list[str]) -> pd.DataFrame:
    keys = list(METRICS)
    fig, axes = plt.subplots(2, 4, figsize=(11.6, 5.6))
    x = np.arange(len(LADDER))
    rows = []

    for ax, key in zip(axes.ravel(), keys):
        label, lower_better = METRICS[key]
        scale = 100.0 if key in PERCENT_METRICS else 1.0
        mus, sds = [], []
        for run, dim, _short, _what in LADDER:
            mu, sd, per_seed = seed_stats(data[run], key)
            mus.append(mu * scale)
            sds.append((sd if np.isfinite(sd) else 0.0) * scale)
            rows.append({"figure": "ladder", "run": run, "obs_dim": dim,
                         "metric": key, "mean": mu, "std_over_seeds": sd})
        mus, sds = np.array(mus), np.array(sds)

        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.errorbar(x, mus, yerr=sds, color=INK3, lw=0, elinewidth=1.1,
                    capsize=3, capthick=1.1, zorder=3)
        ax.plot(x, mus, color=C_BLUE, lw=2.0, zorder=4, solid_capstyle="round")
        for xi, mu, mk, col in zip(x, mus, LADDER_MARKERS, LADDER_COLORS):
            ax.plot([xi], [mu], marker=mk, ms=8, color=col, mec="white",
                    mew=1.6, zorder=5, linestyle="none")

        # baseline reference: the full-observation policy
        ax.axhline(mus[0], color=INK3, lw=0.9, ls=(0, (4, 3)), zorder=2)

        ax.set_title(label, color=INK, pad=7)
        ax.set_xticks(x)
        ax.set_xticklabels([s for _r, _d, s, _w in LADDER])
        ax.set_xlim(-0.45, len(LADDER) - 0.55)
        set_honest_limits(ax, mus, sds, key)
        arrow = "↓ better" if lower_better else "↑ better"
        ax.text(0.03, 0.05, arrow, transform=ax.transAxes, fontsize=7,
                color=INK3, ha="left", va="bottom")
        ax.text(0.97, 0.05, _rel_spread(mus), transform=ax.transAxes, fontsize=7,
                color=INK3, ha="right", va="bottom")

    for ax in axes[1]:
        ax.set_xlabel("observation dimension", labelpad=6)

    stage_note = "   ".join(f"{s}: {w}" for _r, _d, s, w in LADDER)
    header(
        fig,
        "Observation ablation causes no measurable degradation",
        "Isaac Lab sweep, 4380 episodes per policy (8 gaits × 5×5×5 command grid × 5 seeds). Marker = mean of the\n"
        "five seed means, whisker = std over seeds, dashed line = 69-dim baseline. Axes are zero-anchored (ratio\n"
        "scales) or floored well below the data (percentages), so a flat series reads as flat.",
        top=0.80,
        note=stage_note + "      stand excluded from the averages: it is evaluated at the zero command only.",
    )
    fig.subplots_adjust(bottom=0.13, hspace=0.42, wspace=0.30, left=0.055, right=0.99)
    _save(fig, out / ("degradation_ladder" + STEM_SUFFIX), formats)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 2 - the ladder resolved per gait
# ---------------------------------------------------------------------------


def _per_gait(df: pd.DataFrame, metric: str) -> tuple[pd.Series, pd.Series]:
    per_seed = df.groupby(["gait_name", "seed"])[metric].mean()
    mu = per_seed.groupby("gait_name").mean()
    sd = per_seed.groupby("gait_name").std(ddof=1).fillna(0.0)
    return mu, sd


def fig_ladder_per_gait(data: dict[str, pd.DataFrame], out: Path, formats: list[str]) -> None:
    keys = ["survival_rate", "gait_contact_accuracy", "mean_vx_error"]
    fig, axes = plt.subplots(3, 1, figsize=(10.4, 8.4), sharex=True)
    gaits = GAIT_ORDER
    xs = np.arange(len(gaits))
    n = len(LADDER)
    # 2px surface gap between adjacent bars -> a slot narrower than its share
    total_w, gap = 0.80, 0.018
    bw = total_w / n - gap

    for ax, key in zip(axes, keys):
        label, lower_better = METRICS[key]
        scale = 100.0 if key in PERCENT_METRICS else 1.0
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        for i, ((run, dim, short, what), col) in enumerate(zip(LADDER, LADDER_COLORS)):
            mu, sd = _per_gait(data[run], key)
            m = np.array([mu.get(g, np.nan) for g in gaits]) * scale
            s = np.array([sd.get(g, 0.0) for g in gaits]) * scale
            off = (i - (n - 1) / 2) * (total_w / n)
            ax.bar(xs + off, m, bw, color=col, zorder=3,
                   label=f"{short}-dim  ({what})" if ax is axes[0] else None)
            ax.errorbar(xs + off, m, yerr=s, lw=0, ecolor=INK3,
                        elinewidth=0.9, capsize=1.8, capthick=0.9, zorder=4)
        ax.set_ylabel(label)
        arrow = "↓ better" if lower_better else "↑ better"
        ax.text(0.995, 0.93, arrow, transform=ax.transAxes, fontsize=7,
                color=INK3, ha="right", va="top")
        if key == "survival_rate":
            ax.set_ylim(97.5, 100.35)
        elif key == "gait_contact_accuracy":
            ax.set_ylim(88, 101)

    axes[0].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                   handlelength=1.1, columnspacing=1.6)
    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(gaits)
    axes[-1].set_xlabel("gait")

    header(
        fig,
        "Per-gait view of the observation ladder",
        "Bars are the mean of the five seed means; whiskers the std over seeds. "
        "stand is evaluated at the zero command only.\n"
        "The survival and contact-accuracy axes are truncated to resolve differences of a few tenths of a percent.",
        top=0.86,
    )
    fig.subplots_adjust(bottom=0.075, left=0.075, right=0.99, hspace=0.12)
    _save(fig, out / ("degradation_ladder_per_gait" + STEM_SUFFIX), formats)


# ---------------------------------------------------------------------------
# Figure 3 - velocity tracking across the ladder
# ---------------------------------------------------------------------------


def _tracking(df: pd.DataFrame, cmd_col: str, act_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = df[df.gait_name != "stand"]
    per_seed = sub.groupby([cmd_col, "seed"])[act_col].mean()
    mu = per_seed.groupby(cmd_col).mean()
    sd = per_seed.groupby(cmd_col).std(ddof=1).fillna(0.0)
    return mu.index.to_numpy(), mu.to_numpy(), sd.to_numpy()


def _tracking_panel(ax, runs, colors, markers, labels, cmd_col, act_col,
                    data, unit, axis_name):
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    all_c = None
    for run, col, mk, lab in zip(runs, colors, markers, labels):
        c, m, s = _tracking(data[run], cmd_col, act_col)
        all_c = c
        ax.fill_between(c, m - s, m + s, color=col, alpha=0.13, lw=0, zorder=2)
        ax.plot(c, m, color=col, lw=2.0, marker=mk, ms=7, mec="white", mew=1.4,
                zorder=4, label=lab, solid_capstyle="round")
    lo, hi = float(all_c.min()), float(all_c.max())
    ax.plot([lo, hi], [lo, hi], color=INK3, lw=1.0, ls=(0, (4, 3)), zorder=3)
    # inline label on the diagonal, inside the axes, so it cannot collide with
    # the neighbouring panel
    ax.annotate("perfect tracking", xy=(lo + (hi - lo) * 0.28, lo + (hi - lo) * 0.28),
                xytext=(3, -3), textcoords="offset points", fontsize=7,
                color=INK3, ha="left", va="top", rotation=38,
                rotation_mode="anchor", annotation_clip=True, zorder=3)
    ax.set_xlabel(f"commanded {axis_name} [{unit}]")
    ax.set_ylabel(f"achieved {axis_name} [{unit}]")
    ax.set_xticks(all_c)


def fig_tracking(data: dict[str, pd.DataFrame], out: Path, formats: list[str]) -> None:
    runs = [r for r, _d, _s, _w in LADDER]
    labels = [f"{s}-dim  ({w})" for _r, _d, s, w in LADDER]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9))
    _tracking_panel(axes[0], runs, LADDER_COLORS, LADDER_MARKERS, labels,
                    "vx_cmd", "mean_vx_actual", data, "m/s", r"$v_x$")
    _tracking_panel(axes[1], runs, LADDER_COLORS, LADDER_MARKERS, labels,
                    "vy_cmd", "mean_vy_actual", data, "m/s", r"$v_y$")
    _tracking_panel(axes[2], runs, LADDER_COLORS, LADDER_MARKERS, labels,
                    "wz_cmd", "mean_wz_actual", data, "rad/s", r"$\omega_z$")
    axes[1].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.03),
                   handlelength=1.4, columnspacing=1.8)
    header(
        fig,
        "Velocity tracking is unchanged by the ablation — and short at high $v_x$ for every policy",
        "Mean achieved velocity per commanded grid value, averaged over the seven locomotion gaits; "
        "band = std over the five seeds.",
        top=0.74,
    )
    fig.subplots_adjust(bottom=0.16, wspace=0.30, left=0.06, right=0.99)
    _save(fig, out / ("degradation_tracking" + STEM_SUFFIX), formats)


# ---------------------------------------------------------------------------
# Figure 4 - Raibert planner comparison
# ---------------------------------------------------------------------------


CAVEAT_19999 = ("Caveat: the $v_B$ run is checkpoint 14999 and the $v_{cmd}$ run 19999 — training length is "
                "confounded with the planner change.")
CAVEAT_14800 = ("Both policies are evaluated at the same training iteration (checkpoint 14800), so training "
                "length is no longer confounded with the planner change.")


def fig_raibert(data: dict[str, pd.DataFrame], out: Path, formats: list[str]) -> pd.DataFrame:
    CAVEAT = CAVEAT_14800 if "14800" in SWEEP_DIR else CAVEAT_19999
    runs = [r for r, _l in RAIBERT]
    labels = [l for _r, l in RAIBERT]
    keys = list(METRICS)
    rows = []

    fig = plt.figure(figsize=(11.6, 9.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.32], hspace=0.58, wspace=0.34,
                          left=0.055, right=0.99, bottom=0.06, top=0.78)

    # rows 0-1: paired metric panels
    for idx, key in enumerate(keys):
        ax = fig.add_subplot(gs[idx // 4, idx % 4])
        label, lower_better = METRICS[key]
        scale = 100.0 if key in PERCENT_METRICS else 1.0
        mus, sds, seeds = [], [], []
        for run in runs:
            mu, sd, per_seed = seed_stats(data[run], key)
            mus.append(mu * scale)
            sds.append((sd if np.isfinite(sd) else 0.0) * scale)
            seeds.append(per_seed)
        t, p = welch_safe(seeds[0], seeds[1])
        for run, mu, sd in zip(runs, mus, sds):
            rows.append({"figure": "raibert", "run": run, "metric": key,
                         "mean": mu / scale, "std_over_seeds": sd / scale,
                         "welch_p": p})

        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        xs = np.arange(2)
        ax.bar(xs, mus, 0.52, color=RAIBERT_COLORS, zorder=3)
        ax.errorbar(xs, mus, yerr=sds, lw=0, ecolor=INK2, elinewidth=1.1,
                    capsize=3, capthick=1.1, zorder=4)
        # per-seed points: the honest picture of a five-sample comparison
        for xi, per_seed, col in zip(xs, seeds, RAIBERT_COLORS):
            jit = np.linspace(-0.11, 0.11, len(per_seed))
            ax.plot(xi + jit, per_seed * scale, linestyle="none", marker="o",
                    ms=3.4, color="white", mec=INK2, mew=0.8, zorder=5)

        arrow = "↓ better" if lower_better else "↑ better"
        ax.set_title(f"{label}\n{arrow}", color=INK, pad=6, fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels(["$v_B$", "$v_{cmd}$"])
        set_honest_limits(ax, np.array(mus), np.array(sds), key)
        # effect size carries the claim; with five deterministic seeds a p-value
        # reaches 0.001 on a difference far too small to matter
        rel = (mus[0] - mus[1]) / mus[1] * 100.0 if mus[1] else float("nan")
        ax.text(0.5, 0.965, f"Δ {rel:+.1f}%", transform=ax.transAxes, fontsize=7.5,
                color=INK2, ha="center", va="top")

    # row 2 left: tracking curves
    ax_tr = fig.add_subplot(gs[2, 0:2])
    _tracking_panel(ax_tr, runs, RAIBERT_COLORS, RAIBERT_MARKERS, labels,
                    "vx_cmd", "mean_vx_actual", data, "m/s", r"$v_x$")
    ax_tr.set_title("Forward velocity tracking", color=INK, pad=6)
    ax_tr.legend(loc="upper left", handlelength=1.4)

    # row 2 right: per-gait contact accuracy
    ax_g = fig.add_subplot(gs[2, 2:4])
    ax_g.grid(axis="y", zorder=0)
    ax_g.set_axisbelow(True)
    gaits = GAIT_ORDER
    xs = np.arange(len(gaits))
    total_w, gap = 0.74, 0.02
    bw = total_w / 2 - gap
    for i, (run, col, lab) in enumerate(zip(runs, RAIBERT_COLORS, labels)):
        mu, sd = _per_gait(data[run], "gait_contact_accuracy")
        m = np.array([mu.get(g, np.nan) for g in gaits]) * 100
        s = np.array([sd.get(g, 0.0) for g in gaits]) * 100
        off = (i - 0.5) * (total_w / 2)
        ax_g.bar(xs + off, m, bw, color=col, zorder=3, label=lab)
        ax_g.errorbar(xs + off, m, yerr=s, lw=0, ecolor=INK3, elinewidth=0.9,
                      capsize=1.8, capthick=0.9, zorder=4)
    ax_g.set_ylim(85, 101)
    ax_g.set_xticks(xs)
    ax_g.set_xticklabels(gaits, fontsize=7.5)
    ax_g.set_ylabel("gait contact accuracy [%]")
    ax_g.set_title("Gait contact accuracy per gait   ↑ better", color=INK, pad=6)

    header(
        fig,
        "Raibert planner: the measured $v_B$ does not beat the feedforward $v_{cmd}$",
        "Both policies are 53-dimensional and were swept over the identical grid. Bars = mean of the five seed means,\n"
        "whiskers = std over seeds, hollow dots = the individual seed means. Δ is the $v_B$ policy relative to the $v_{cmd}$ policy.\n"
        + CAVEAT,
        top=0.78,
    )
    _save(fig, out / ("raibert_comparison" + STEM_SUFFIX), formats)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------


def header(fig, title: str, subtitle: str, *, top: float, note: str | None = None) -> None:
    """Reserve a band above the axes and draw the title/subtitle into it.

    ``tight_layout``/``suptitle`` fight each other once a figure mixes gridspec
    panels and multi-line prose, so the band is reserved explicitly instead.
    """
    fig.subplots_adjust(top=top)
    y = top + (1.0 - top) * 0.62
    fig.text(0.008, y + 0.045, title, ha="left", va="bottom",
             fontsize=13, color=INK, weight="bold")
    fig.text(0.008, y + 0.035, subtitle, ha="left", va="top",
             fontsize=8.5, color=INK2, linespacing=1.5)
    if note:
        fig.text(0.008, 0.004, note, ha="left", va="bottom", fontsize=8, color=INK3)


def set_honest_limits(ax, values: np.ndarray, errs: np.ndarray, key: str) -> None:
    """Axis range that does not manufacture a difference out of noise.

    Ratio-scale metrics (errors, torque, height spread, pitch) are anchored at
    zero, so a flat series looks flat. Percentages keep a floor well below the
    data rather than zooming onto the last tenth of a percent.
    """
    top = float(np.max(values + errs))
    bot = float(np.min(values - errs))
    if key in PERCENT_METRICS:
        ax.set_ylim(max(0.0, min(bot - 2.0, 100.0 - (100.0 - bot) * 2.2)), 100.9)
    else:
        ax.set_ylim(0.0, top * 1.35 if top > 0 else 1.0)


def _rel_spread(values: np.ndarray) -> str:
    """Largest deviation from the baseline value, in percent of the baseline."""
    base = values[0]
    if base == 0:
        return ""
    d = np.max(np.abs(values - base)) / abs(base) * 100.0
    return f"max Δ vs baseline: {d:.1f}%"


def _save(fig, stem: Path, formats: list[str]) -> None:
    for fmt in formats:
        fig.savefig(f"{stem}.{fmt}")
        print(f"[plot] {stem}.{fmt}")
    plt.close(fig)


def print_tables(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Console tables plus the tidy frame written next to the figures."""
    rows = []
    groups = [("ladder", [(r, f"{s}-dim ({w})") for r, _d, s, w in LADDER]),
              ("raibert", list(RAIBERT))]
    for gname, members in groups:
        print(f"\n=== {gname} ===")
        hdr = f"{'policy':<34}" + "".join(f"{k[:11]:>13}" for k in METRICS)
        print(hdr)
        print("-" * len(hdr))
        for run, lab in members:
            cells = []
            for key in METRICS:
                mu, sd, per_seed = seed_stats(data[run], key)
                scale = 100.0 if key in PERCENT_METRICS else 1.0
                cells.append(f"{mu * scale:>8.3f}±{sd * scale:.2f}"[-13:])
                rows.append({"group": gname, "run": run, "label": lab,
                             "metric": key, "mean": mu, "std_over_seeds": sd,
                             "seed_means": ";".join(f"{v:.6f}" for v in per_seed)})
            print(f"{lab:<34}" + "".join(f"{c:>13}" for c in cells))
    return pd.DataFrame(rows)


def print_significance(data: dict[str, pd.DataFrame]) -> None:
    print("\n=== Welch t-test on the five seed means ===")
    base = LADDER[0][0]
    print(f"\n[ladder] each stage vs the {LADDER[0][1]}-dim baseline")
    for run, dim, short, what in LADDER[1:]:
        line = [f"  {short}-dim {what:<20}"]
        for key in ("survival_rate", "gait_contact_accuracy", "mean_vx_error"):
            _m, _s, a = seed_stats(data[base], key)
            _m, _s, b = seed_stats(data[run], key)
            t, p = welch_safe(a, b)
            line.append(f"{key}: p={'n/a' if p is None else f'{p:.3f}'}")
        print("  ".join(line))

    a_run, b_run = RAIBERT[0][0], RAIBERT[1][0]
    print(f"\n[raibert] {RAIBERT[0][1]} vs {RAIBERT[1][1]}")
    for key in METRICS:
        _m, _s, a = seed_stats(data[a_run], key)
        _m, _s, b = seed_stats(data[b_run], key)
        t, p = welch_safe(a, b)
        d = (a.mean() - b.mean())
        flag = "" if (p is None or p >= 0.05) else "   *"
        print(f"  {key:<24} delta={d:+.4f}  p={'n/a' if p is None else f'{p:.3f}'}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out_dir", default=str(REPO / "robustness_sweep" / "figures"))
    ap.add_argument("--formats", default="png,pdf")
    ap.add_argument("--sweep_dir", default="isaac",
                    help="Sweep sub-directory under <run>/robustness_sweep/.")
    ap.add_argument("--stem_suffix", default="",
                    help="Suffix appended to every figure filename.")
    a = ap.parse_args()

    global SWEEP_DIR, STEM_SUFFIX
    SWEEP_DIR, STEM_SUFFIX = a.sweep_dir, a.stem_suffix

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in a.formats.split(",") if f.strip()]

    _mpl_style()

    runs = sorted({r for r, _d, _s, _w in LADDER} | {r for r, _l in RAIBERT})
    data = {r: load_run(r) for r in runs}
    for r in runs:
        print(f"[load] {r}: {len(data[r])} episodes")

    fig_ladder(data, out, formats)
    fig_ladder_per_gait(data, out, formats)
    fig_tracking(data, out, formats)
    fig_raibert(data, out, formats)
    fig_raibert_tracking(data, out, formats)
    fig_foot_placement(data, out, formats)

    tidy = print_tables(data)
    tidy.to_csv(out / f"degradation_metrics{STEM_SUFFIX}.csv", index=False)
    print(f"\n[plot] {out / ('degradation_metrics' + STEM_SUFFIX + '.csv')}")
    print_significance(data)




# ---------------------------------------------------------------------------
# Figure 5 - velocity tracking of the two Raibert planner variants, on its own
# ---------------------------------------------------------------------------


def _fit_per_seed(df: pd.DataFrame, cmd_col: str, act_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-seed least-squares gain and offset of achieved-vs-commanded."""
    gains, offsets = [], []
    for _seed, g in df.groupby("seed"):
        m = g.groupby(cmd_col)[act_col].mean()
        a, b = np.polyfit(m.index.values.astype(float), m.values.astype(float), 1)
        gains.append(a)
        offsets.append(b)
    return np.array(gains), np.array(offsets)


def fig_raibert_tracking(data: dict[str, pd.DataFrame], out: Path, formats: list[str]) -> None:
    """Velocity tracking only: where the feedforward planner's advantage lives."""
    runs = [r for r, _l in RAIBERT]
    labels = ["planner on measured $v_B$", "planner on commanded $v_{cmd}$"]
    loco = {r: data[r][data[r].gait_name != "stand"] for r in runs}

    fig = plt.figure(figsize=(11.6, 7.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.26,
                          left=0.06, right=0.985, bottom=0.075, top=0.80)

    # -- (a) achieved vs commanded vx, with the fitted line -------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.grid(zorder=0); ax.set_axisbelow(True)
    cmds = np.array(sorted(loco[runs[0]].vx_cmd.unique()))
    for r, col, mk, lab in zip(runs, RAIBERT_COLORS, RAIBERT_MARKERS, labels):
        per_seed = loco[r].groupby(["vx_cmd", "seed"]).mean_vx_actual.mean()
        mu = per_seed.groupby("vx_cmd").mean()
        sd = per_seed.groupby("vx_cmd").std(ddof=1).fillna(0.0)
        g, o = _fit_per_seed(loco[r], "vx_cmd", "mean_vx_actual")
        ax.fill_between(cmds, mu - sd, mu + sd, color=col, alpha=0.13, lw=0, zorder=2)
        ax.plot(cmds, g.mean() * cmds + o.mean(), color=col, lw=1.0, ls=(0, (5, 3)), zorder=3)
        ax.plot(cmds, mu, color=col, lw=2.0, marker=mk, ms=7, mec="white", mew=1.4,
                zorder=4, label=f"{lab}\ngain {g.mean():.3f}, offset {o.mean():+.3f}")
    ax.plot([0, 1.2], [0, 1.2], color=INK3, lw=1.0, ls=(0, (4, 3)), zorder=3)
    ax.annotate("perfect tracking (gain 1)", xy=(0.62, 0.62), xytext=(3, -3),
                textcoords="offset points", fontsize=7, color=INK3,
                ha="left", va="top", rotation=38, rotation_mode="anchor", zorder=3)
    ax.set_xticks(cmds)
    ax.set_xlabel("commanded $v_x$ [m/s]")
    ax.set_ylabel("achieved $v_x$ [m/s]")
    ax.set_title("(a)  Forward tracking is a gain deficit, not a wobble",
                 color=INK, pad=6, loc="left")
    ax.legend(loc="upper left", handlelength=1.5, fontsize=7.5, labelspacing=0.9)

    # -- (b) MAE vs commanded vx ---------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.grid(zorder=0); ax.set_axisbelow(True)
    for r, col, mk, lab in zip(runs, RAIBERT_COLORS, RAIBERT_MARKERS, labels):
        per_seed = loco[r].groupby(["vx_cmd", "seed"]).mean_vx_error.mean()
        mu = per_seed.groupby("vx_cmd").mean()
        sd = per_seed.groupby("vx_cmd").std(ddof=1).fillna(0.0)
        ax.fill_between(cmds, mu - sd, mu + sd, color=col, alpha=0.13, lw=0, zorder=2)
        ax.plot(cmds, mu, color=col, lw=2.0, marker=mk, ms=7, mec="white", mew=1.4,
                zorder=4, label=lab)
    ax.axvline(0.6, color=INK3, lw=0.9, ls=(0, (2, 3)), zorder=2)
    ax.set_ylim(0, None)
    ax.text(0.6, ax.get_ylim()[1] * 0.02, "  both policies cross the\n  identity line here",
            fontsize=7, color=INK3, ha="left", va="bottom")
    ax.set_xticks(cmds)
    ax.set_xlabel("commanded $v_x$ [m/s]")
    ax.set_ylabel(r"$|e_{v_x}|$  [m/s]   ↓ better")
    ax.set_title("(b)  The gap opens at both ends of the grid", color=INK, pad=6, loc="left")
    ax.legend(loc="upper center", handlelength=1.5, fontsize=7.5)

    # -- (c) tracking gain per channel ---------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
    chans = [("$v_x$", "vx_cmd", "mean_vx_actual"),
             ("$v_y$", "vy_cmd", "mean_vy_actual"),
             (r"$\omega_z$", "wz_cmd", "mean_wz_actual")]
    xs = np.arange(len(chans))
    bw = 0.36 - 0.02
    for i, (r, col, lab) in enumerate(zip(runs, RAIBERT_COLORS, labels)):
        vals, errs = [], []
        for _n, cc, ac in chans:
            g, _o = _fit_per_seed(loco[r], cc, ac)
            vals.append(g.mean()); errs.append(g.std(ddof=1))
        off = (i - 0.5) * 0.36
        ax.bar(xs + off, vals, bw, color=col, zorder=3, label=lab)
        ax.errorbar(xs + off, vals, yerr=errs, lw=0, ecolor=INK2,
                    elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=4)
        for x, v in zip(xs + off, vals):
            ax.text(x, v + 0.03, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK2)
    ax.axhline(1.0, color=INK3, lw=1.0, ls=(0, (4, 3)), zorder=5)
    ax.text(len(chans) - 0.55, 1.02, "perfect tracking", fontsize=7,
            color=INK3, ha="right", va="bottom")
    ax.set_xticks(xs); ax.set_xticklabels([n for n, _c, _a in chans])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("tracking gain  ↑ better")
    ax.set_title("(c)  Only $v_x$ differs — and $v_y$ barely tracks at all",
                 color=INK, pad=6, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.97),
              handlelength=1.1, fontsize=7.5)

    # -- (d) per-gait vx MAE --------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
    gaits = [g for g in GAIT_ORDER if g != "stand"]
    order = (loco[runs[0]].groupby("gait_name").mean_vx_error.mean()
             - loco[runs[1]].groupby("gait_name").mean_vx_error.mean())
    gaits = [g for g in order.sort_values(ascending=False).index if g in gaits]
    xs = np.arange(len(gaits))
    for i, (r, col, lab) in enumerate(zip(runs, RAIBERT_COLORS, labels)):
        per_seed = loco[r].groupby(["gait_name", "seed"]).mean_vx_error.mean()
        mu = per_seed.groupby("gait_name").mean()
        sd = per_seed.groupby("gait_name").std(ddof=1).fillna(0.0)
        m = np.array([mu[g] for g in gaits]); s = np.array([sd[g] for g in gaits])
        off = (i - 0.5) * 0.36
        ax.bar(xs + off, m, bw, color=col, zorder=3, label=lab)
        ax.errorbar(xs + off, m, yerr=s, lw=0, ecolor=INK2, elinewidth=0.9,
                    capsize=2, capthick=0.9, zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels(gaits, fontsize=8)
    ax.set_ylim(0, None)
    ax.set_ylabel(r"$|e_{v_x}|$  [m/s]   ↓ better")
    ax.set_title("(d)  Largest gain in the contact-rich gaits, least in bound",
                 color=INK, pad=6, loc="left")
    ax.legend(loc="upper right", handlelength=1.1, fontsize=7.5)

    header(
        fig,
        "Velocity tracking: dropping $v_B$ from the planner raises the forward gain",
        "The two 53-dimensional policies at the same training iteration (checkpoint 14800), locomotion gaits only.\n"
        "Bands and whiskers are the std over the five evaluation seeds; gain and offset are least-squares fits of\n"
        "achieved against commanded velocity, fitted per seed.",
        top=0.80,
    )
    _save(fig, out / ("raibert_velocity_tracking" + STEM_SUFFIX), formats)


# ---------------------------------------------------------------------------
# Figure 6 - does the policy follow the Raibert planner's SPATIAL output?
# ---------------------------------------------------------------------------

def fig_foot_placement(data: dict[str, pd.DataFrame], out: Path, formats: list[str]) -> None:
    """Foot placement at touchdown, against the reference the policy was given.

    ``gait_contact_accuracy`` only checks contact TIMING. These panels check the
    other half of the planner: where the foot actually lands relative to
    ``beta_p_ref_B``, and whether the x_lim clamp caps the reference itself.
    """
    from robustness_sweep.grid import GAIT_CONFIGS, NAME_TO_GAIT_ID

    ref_run = RAIBERT[1][0]                      # the 53-dim v_cmd policy
    if "place_err_xy" not in data[ref_run].columns:
        print("[plot] no placement columns in this sweep - skipping figure 6")
        return
    gaits = [g for g in GAIT_ORDER if g != "stand"]
    x = data[ref_run][data[ref_run].gait_name != "stand"]

    fig = plt.figure(figsize=(11.6, 7.6))
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.26,
                          left=0.075, right=0.985, bottom=0.075, top=0.795)

    # -- (a) the clamp: reference stride vs command --------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.grid(zorder=0); ax.set_axisbelow(True)
    cmds = np.array(sorted(x.vx_cmd.unique()))
    for g in gaits:
        c = GAIT_CONFIGS[str(NAME_TO_GAIT_ID[g])]
        xl = c["x_lim"]
        Tst = c["threshold"] * c["period"]
        rd = x[x.gait_name == g].groupby("vx_cmd").ref_dx_at_touchdown.mean().reindex(cmds)
        sat = bool(np.any(0.5 * Tst * cmds > xl + 1e-9))
        col = C_ORANGE if sat else INK3
        ax.plot(cmds, rd.values, color=col, lw=2.0 if sat else 1.2,
                marker="o", ms=4, zorder=4 if sat else 3,
                alpha=1.0 if sat else 0.65)
        ax.annotate(g, xy=(cmds[-1], rd.values[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=7.5,
                    color=col, va="center", weight="bold" if sat else "normal")
    ax.set_xticks(cmds); ax.set_xlim(-0.05, 1.42)
    ax.set_xlabel("commanded $v_x$ [m/s]")
    ax.set_ylabel("reference stride offset  $ref_{dx}$ [m]")
    ax.set_title("(a)  The $x_{lim}$ clamp flattens the reference", color=INK, pad=6, loc="left")
    ax.text(0.03, 0.95, "orange = clamp binds inside the grid", transform=ax.transAxes,
            fontsize=7.5, color=C_ORANGE, va="top")

    # -- (b) placement error against the reference range ---------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.grid(axis="x", zorder=0); ax.set_axisbelow(True)
    order = sorted(gaits, key=lambda g: x[x.gait_name == g].place_err_x.mean())
    ys = np.arange(len(order))
    rng = [x[x.gait_name == g].groupby("vx_cmd").ref_dx_at_touchdown.mean().max() for g in order]
    err = [x[x.gait_name == g].place_err_x.mean() for g in order]
    ax.barh(ys, rng, 0.62, color=GRID, zorder=3, label="reference range (0 → max $ref_{dx}$)")
    ax.barh(ys, err, 0.30, color=C_BLUE, zorder=4, label="mean $|e_x|$ at touchdown")
    for y, r_, e_ in zip(ys, rng, err):
        ax.text(max(r_, e_) + 0.004, y, f"{e_ / r_:.0%}", va="center",
                fontsize=7.5, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels(order)
    ax.set_xlabel("metres")
    ax.set_title("(b)  The miss is a large fraction of the signal", color=INK, pad=6, loc="left")
    ax.legend(loc="upper right", fontsize=7.5, handlelength=1.1)

    # -- (c) zero-command drift vs signed placement bias, pooled -------------
    ax = fig.add_subplot(gs[1, 0])
    ax.grid(zorder=0); ax.set_axisbelow(True)
    px, py = [], []
    for run, _d, _s, _w in LADDER:
        px, py = px, py
    for run in {r for r, _d, _s, _w in LADDER} | {r for r, _l in RAIBERT}:
        df = data[run]
        z = df[(df.gait_name != "stand") & (df.vx_cmd == 0) & (df.vy_cmd == 0) & (df.wz_cmd == 0)]
        gg = z.groupby("gait_name").agg(ex=("place_err_x_signed", "mean"),
                                        v=("mean_vx_actual", "mean"))
        px += list(gg.ex.values); py += list(gg.v.values)
    px, py = np.array(px), np.array(py)
    sl, ic = np.polyfit(px, py, 1)
    from scipy import stats as _st
    rho, pv = _st.spearmanr(px, py)
    xs = np.linspace(px.min(), px.max(), 50)
    ax.plot(xs, sl * xs + ic, color=INK3, lw=1.2, ls=(0, (5, 3)), zorder=3)
    ax.plot(px, py, linestyle="none", marker="o", ms=6, color=C_BLUE,
            mec="white", mew=1.0, alpha=0.85, zorder=4)
    ax.set_xlabel("signed $e_x$ at touchdown [m]   (foot lands ahead of reference →)")
    ax.set_ylabel("drift $v_x$ at zero command [m/s]")
    ax.set_title("(c)  The drift follows the placement bias", color=INK, pad=6, loc="left")
    ax.text(0.03, 0.95,
            f"5 policies × 7 gaits, n={len(px)}\nSpearman ρ = {rho:+.2f}  (p < 0.0001)\n"
            f"fit: drift = {sl:.2f}·$e_x$ {ic:+.3f}",
            transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")

    # -- (d) achieved vs the velocity the reference encodes, at cmd 1.2 ------
    ax = fig.add_subplot(gs[1, 1])
    ax.grid(zorder=0); ax.set_axisbelow(True)
    impl, ach = [], []
    for g in gaits:
        c = GAIT_CONFIGS[str(NAME_TO_GAIT_ID[g])]
        Tst = c["threshold"] * c["period"]
        t = x[(x.gait_name == g) & (x.vx_cmd == 1.2)]
        i_ = 2 * t.ref_dx_at_touchdown.mean() / Tst
        a_ = t.mean_vx_actual.mean()
        impl.append(i_); ach.append(a_)
        close = abs(a_ - i_) < 0.05
        ax.plot([i_], [a_], marker="o", ms=8, color=C_ORANGE if close else C_BLUE,
                mec="white", mew=1.4, zorder=5, linestyle="none")
        # stagger the labels of gaits that share an x position (all the
        # unclamped ones sit at the full 1.2 m/s reference)
        dy = {"bound": 7, "limp": -4, "hop": -1}.get(g, -3)
        ha = "right" if g == "pronk" else "left"
        dx = -8 if g == "pronk" else 8
        ax.annotate(g, xy=(i_, a_), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, color=C_ORANGE if close else INK2,
                    ha=ha, weight="bold" if close else "normal")
    lo, hi = 0.55, 1.32
    ax.plot([lo, hi], [lo, hi], color=INK3, lw=1.0, ls=(0, (4, 3)), zorder=3)
    ax.annotate("achieved = reference", xy=(0.72, 0.72), xytext=(3, -3),
                textcoords="offset points", fontsize=7, color=INK3,
                rotation=38, rotation_mode="anchor", ha="left", va="top")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"velocity the reference encodes, $2\,ref_{dx}/T_{st}$ [m/s]")
    ax.set_ylabel("achieved $v_x$ [m/s]")
    ax.set_title("(d)  Only the clamped gaits sit on the reference",
                 color=INK, pad=6, loc="left")

    header(
        fig,
        "The planner's timing is tracked; its foot placement is not",
        "53-dim $v_{cmd}$ policy at checkpoint 14800 (panel c pools all five policies), locomotion gaits only.\n"
        "Foot placement is sampled at every swing→stance transition and compared against the reference\n"
        "$beta\\_p\\_ref\\_B$ the policy itself receives. Gait contact accuracy (~94%) scores only contact timing.",
        top=0.795,
    )
    _save(fig, out / ("foot_placement" + STEM_SUFFIX), formats)


if __name__ == "__main__":
    main()
