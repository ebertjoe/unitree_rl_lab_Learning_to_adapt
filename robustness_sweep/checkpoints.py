"""Resolution of policy runs and checkpoints under ``logs/rsl_rl/<experiment>/``."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EXPERIMENT = "unitree_go2_locomotion_paper"
DEFAULT_LOG_ROOT = Path("logs") / "rsl_rl"

_MODEL_RE = re.compile(r"^model_(\d+)\.pt$")


@dataclass(frozen=True)
class PolicyRef:
    """A resolved policy: its run directory, RSL-RL checkpoint and TorchScript export."""

    experiment: str
    run_dir: Path
    checkpoint: Path
    jit_policy: Path | None

    @property
    def run(self) -> str:
        return self.run_dir.name

    @property
    def tag(self) -> str:
        return f"{self.run}/{self.checkpoint.name}"

    def output_dir(self, simulator: str) -> Path:
        return self.run_dir / "robustness_sweep" / simulator

    def as_dict(self) -> dict:
        return {
            "experiment": self.experiment,
            "run": self.run,
            "run_dir": str(self.run_dir),
            "checkpoint": self.checkpoint.name,
            "checkpoint_path": str(self.checkpoint),
            "jit_policy": str(self.jit_policy) if self.jit_policy else "",
        }


def list_runs(log_root: Path = DEFAULT_LOG_ROOT, experiment: str = DEFAULT_EXPERIMENT) -> list[Path]:
    """Run directories of an experiment, oldest first (the names are timestamps)."""
    root = Path(log_root) / experiment
    if not root.is_dir():
        raise FileNotFoundError(f"experiment directory not found: {root}")
    runs = [p for p in root.iterdir() if p.is_dir() and any(p.glob("model_*.pt"))]
    return sorted(runs, key=lambda p: p.name)


def list_checkpoints(run_dir: Path) -> list[Path]:
    """``model_*.pt`` files of a run, sorted by training iteration."""
    ckpts = []
    for p in Path(run_dir).glob("model_*.pt"):
        m = _MODEL_RE.match(p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    return [p for _, p in sorted(ckpts)]


def resolve_policy(
    run: str = "latest",
    checkpoint: str = "latest",
    log_root: Path | str = DEFAULT_LOG_ROOT,
    experiment: str = DEFAULT_EXPERIMENT,
) -> PolicyRef:
    """Resolve ``--run``/``--checkpoint`` into a concrete :class:`PolicyRef`.

    ``run`` accepts ``latest``, a run directory name (``2026-08-29_10-25-57``) or
    an absolute/relative path. ``checkpoint`` accepts ``latest``, a file name
    (``model_9999.pt``), a bare iteration number, or a path to a ``.pt`` file.
    """
    log_root = Path(log_root)

    # -- run ---------------------------------------------------------------
    run_path = Path(run)
    if run not in ("latest", "last") and run_path.is_dir():
        run_dir = run_path
    elif run in ("latest", "last"):
        runs = list_runs(log_root, experiment)
        if not runs:
            raise FileNotFoundError(
                f"no run under {log_root / experiment} contains a model_*.pt checkpoint"
            )
        run_dir = runs[-1]
    else:
        run_dir = log_root / experiment / run
        if not run_dir.is_dir():
            available = ", ".join(p.name for p in list_runs(log_root, experiment)[-5:])
            raise FileNotFoundError(f"run '{run}' not found; latest runs: {available}")

    # -- checkpoint --------------------------------------------------------
    ckpt_path = Path(checkpoint)
    if checkpoint not in ("latest", "last") and ckpt_path.is_file():
        ckpt = ckpt_path
    else:
        ckpts = list_checkpoints(run_dir)
        if not ckpts:
            raise FileNotFoundError(f"no model_*.pt checkpoint in {run_dir}")
        if checkpoint in ("latest", "last"):
            ckpt = ckpts[-1]
        else:
            name = checkpoint if checkpoint.endswith(".pt") else f"model_{checkpoint}.pt"
            candidate = run_dir / name
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"checkpoint '{name}' not in {run_dir}; "
                    f"available: {', '.join(p.name for p in ckpts[-5:])}"
                )
            ckpt = candidate

    jit = run_dir / "exported" / "policy.pt"
    return PolicyRef(
        experiment=experiment,
        run_dir=run_dir,
        checkpoint=ckpt,
        jit_policy=jit if jit.is_file() else None,
    )
