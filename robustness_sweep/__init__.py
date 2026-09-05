"""Robustness sweep harness (thesis §4.1, command grid of Table 4.1).

The package holds everything that is *simulator independent*:

* :mod:`robustness_sweep.grid`        - the command grid, gait table and protocol constants
* :mod:`robustness_sweep.metrics`     - per-episode metric accumulation and the CSV schema
* :mod:`robustness_sweep.checkpoints` - policy/run resolution inside ``logs/rsl_rl/<experiment>/``
* :mod:`robustness_sweep.video`       - annotated mp4 writer + seekable video manifest
* :mod:`robustness_sweep.raibert_np`  - NumPy port of the Raibert gait planner (MuJoCo backend)
* :mod:`robustness_sweep.summarise`   - aggregation of the raw episode CSV

The two simulator backends are driven by the top-level entry points:

* ``run_robustness_sweep_isaac.py``   - Isaac Lab / Isaac Sim
* ``run_robustness_sweep_mujoco.py``  - MuJoCo (sim-to-sim)
"""

from robustness_sweep import grid, metrics  # noqa: F401

__all__ = ["grid", "metrics"]
