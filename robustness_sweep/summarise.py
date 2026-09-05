"""Aggregation of a raw ``episodes.csv`` produced by either sweep backend.

Only the standard library is used so the summary can also be produced on a
machine that has neither Isaac Lab nor MuJoCo installed:

    python -m robustness_sweep.summarise path/to/episodes.csv
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from robustness_sweep.grid import FOOT_ORDER, GAIT_NAMES

# metrics averaged over the episodes of a group
MEAN_FIELDS = [
    "mean_vx_error", "mean_vy_error", "mean_wz_error",
    "mean_vx_actual", "mean_vy_actual", "mean_wz_actual",
    "mean_height", "std_height",
    "mean_abs_roll", "mean_abs_pitch",
    "mean_contact_frac", "gait_contact_accuracy", "gait_contact_accuracy_all_feet",
    *[f"contact_frac_{f}" for f in FOOT_ORDER],
    *[f"contact_acc_{f}" for f in FOOT_ORDER],
    "mean_torque_norm", "survival_time_s",
]


def _num(value):
    if value in ("", None):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _std(values):
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0 if vals else None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _fmt(v, nd=4):
    return "" if v is None else round(v, nd)


def read_episodes(csv_path: Path | str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _aggregate(rows: list[dict], key_fields: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k, "") for k in key_fields)].append(r)

    out = []
    for key, group in sorted(groups.items(), key=lambda kv: kv[0]):
        rec = dict(zip(key_fields, key))
        rec["n_episodes"] = len(group)
        survived = [int(r["survived"]) for r in group if r["survived"] not in ("", None)]
        rec["survival_rate"] = round(sum(survived) / len(survived), 4) if survived else ""

        reasons = defaultdict(int)
        for r in group:
            reasons[r["termination_reason"] or "unknown"] += 1
        rec["termination_reasons"] = ";".join(
            f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])
        )

        for field in MEAN_FIELDS:
            rec[field] = _fmt(_mean([_num(r.get(field)) for r in group]))
        # seed spread of the two headline numbers
        rec["survival_rate_std_over_seeds"] = _fmt(_seed_spread(group, "survived"))
        rec["gait_contact_accuracy_std_over_seeds"] = _fmt(
            _seed_spread(group, "gait_contact_accuracy")
        )

        zero = [r for r in group if r.get("zero_command") == "1"]
        rec["n_zero_command"] = len(zero)
        rec["mean_xy_drift_m_zero_cmd"] = _fmt(_mean([_num(r.get("xy_drift_m")) for r in zero]))
        rec["max_xy_drift_m_zero_cmd"] = _fmt(
            max((v for v in (_num(r.get("xy_drift_m")) for r in zero) if v is not None), default=None)
        )
        rec["mean_yaw_drift_rad_zero_cmd"] = _fmt(
            _mean([_num(r.get("yaw_drift_rad")) for r in zero])
        )
        out.append(rec)
    return out


def _seed_spread(group: list[dict], field: str):
    """Standard deviation across seeds of the per-seed mean of ``field``."""
    per_seed: dict[str, list[float]] = defaultdict(list)
    for r in group:
        v = _num(r.get(field))
        if v is not None:
            per_seed[r.get("seed", "")].append(v)
    means = [sum(v) / len(v) for v in per_seed.values() if v]
    return _std(means) if len(means) > 1 else 0.0 if means else None


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarise(csv_path: Path | str, out_dir: Path | str | None = None) -> dict:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir) if out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_episodes(csv_path)
    if not rows:
        print(f"[summarise] {csv_path} is empty")
        return {}

    by_gait = _aggregate(rows, ["simulator", "gait_name"])
    by_command = _aggregate(rows, ["simulator", "gait_name", "vx_cmd", "vy_cmd", "wz_cmd"])
    by_seed = _aggregate(rows, ["simulator", "seed"])

    _write(out_dir / "summary_by_gait.csv", by_gait)
    _write(out_dir / "summary_by_command.csv", by_command)
    _write(out_dir / "summary_by_seed.csv", by_seed)

    _print_table(by_gait, rows)
    return {"by_gait": by_gait, "by_command": by_command, "by_seed": by_seed}


def _print_table(by_gait: list[dict], rows: list[dict]) -> None:
    hdr = (f"{'gait':<8}{'n':>6}{'surv':>8}{'+-seed':>8}"
           f"{'e_vx':>8}{'e_vy':>8}{'e_wz':>8}"
           f"{'height':>8}{'sd_h':>7}{'|roll|':>8}{'|pitch|':>8}{'gaitAcc':>9}{'drift0':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    order = {name: i for i, name in enumerate(GAIT_NAMES.values())}
    for rec in sorted(by_gait, key=lambda r: order.get(r["gait_name"], 99)):
        def g(k, nd=3):
            v = rec.get(k, "")
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"
        print(
            f"{rec['gait_name']:<8}{rec['n_episodes']:>6}"
            f"{g('survival_rate'):>8}{g('survival_rate_std_over_seeds'):>8}"
            f"{g('mean_vx_error'):>8}{g('mean_vy_error'):>8}{g('mean_wz_error'):>8}"
            f"{g('mean_height'):>8}{g('std_height'):>7}"
            f"{g('mean_abs_roll'):>8}{g('mean_abs_pitch'):>8}"
            f"{g('gait_contact_accuracy'):>9}{g('mean_xy_drift_m_zero_cmd'):>8}"
        )
    n = len(rows)
    surv = sum(int(r["survived"]) for r in rows if r["survived"] not in ("", None))
    print("-" * len(hdr))
    print(f"{'ALL':<8}{n:>6}{surv / max(n, 1):>8.3f}")
    print(flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Summarise a robustness sweep episodes.csv")
    ap.add_argument("csv_path")
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    summarise(a.csv_path, a.out_dir)
