# Robustness sweep (thesis §4.1, Table 4.1)

Evaluates a Go2 locomotion policy from `logs/rsl_rl/unitree_go2_locomotion_paper/*`
over the full command grid, for every gait, in **both simulators**, and records one
row per episode plus an annotated video of the sweep.

## Grid

| dimension | values |
|---|---|
| `vx_cmd` [m/s] | 0.0, 0.3, 0.6, 0.9, 1.2 |
| `vy_cmd` [m/s] | −0.4, −0.2, 0.0, 0.2, 0.4 |
| `wz_cmd` [rad/s] | −0.5, −0.25, 0.0, 0.25, 0.5 |
| gaits | bound, trot, hop, amble, pronk, limp, stand, run |
| seeds | 5 |

`stand` is evaluated only at zero command, so the sweep is
`(7 × 125 + 1) × 5 = 4380` episodes per simulator. The nominal 5000 of Table 4.1
counts the full grid for `stand` as well; both numbers are printed and stored in
`config.json` so the difference is explicit.

Each episode is 10 s of scored simulation (1000 control steps at 100 Hz) preceded
by a settle window that is excluded from the metrics and during which no
termination is declared: 1.0 s by default, 1.5 s for `hop` and 2.0 s for `pronk`,
whose first flight phase starts from a standing pose.

## Running it

```bash
# Isaac Lab - latest checkpoint of the latest run
python run_robustness_sweep_isaac.py --headless

# ... a specific policy, with the sweep video
python run_robustness_sweep_isaac.py --headless --video \
    --run 2026-08-29_10-25-57 --checkpoint model_9999.pt

# MuJoCo (sim-to-sim); needs `pip install mujoco` and a Go2 MJCF scene
python run_robustness_sweep_mujoco.py \
    --mjcf ~/Github/unitree_mujoco/unitree_robots/go2/scene.xml --video

# shakedown run
python run_robustness_sweep_isaac.py --headless --seeds 1 --gaits trot,stand --num_envs 32
```

`--run` / `--checkpoint` accept `latest`, a run folder name, an iteration number,
a file name or a path. Both scripts also take `--gaits`, `--seeds`, `--out_dir`
and `--resume` (append to an existing `episodes.csv` and continue where it
stopped, in the same order).

The MuJoCo backend runs the **TorchScript export** `<run>/exported/policy.pt`,
which already contains the empirical observation normaliser. If a run has no
export yet, produce one with `scripts/rsl_rl/play.py`, or point `--policy_jit`
at one.

Cost, measured on this machine: Isaac ≈ 18 s per batch of 128 parallel episodes
(so ≈ 10 min for the full 4380 with `--num_envs 512`); MuJoCo ≈ 1.5 s per episode
single-threaded, ≈ 110 min for the full sweep.

## Output

Written to `<run>/robustness_sweep/<simulator>/` unless `--out_dir` is given.

| file | contents |
|---|---|
| `episodes.csv` | one row per episode, schema in `metrics.CSV_FIELDS` |
| `summary_by_gait.csv` | per gait, including the spread across seeds |
| `summary_by_command.csv` | per (gait, vx, vy, wz) cell, aggregated over seeds |
| `summary_by_seed.csv` | per seed |
| `config.json` | grid, thresholds and every protocol switch used |
| `video/sweep.mp4` | annotated video of the sweep (with `--video`) |
| `video/manifest.csv` | segment → episode, for seeking |

Per episode `episodes.csv` records survival and termination reason; the mean
tracking errors and the achieved values in `vx`, `vy` and `wz`; the mean and
standard deviation of the base height; mean roll and pitch derived from projected
gravity (signed and absolute); the per-foot contact fraction; the gait contact
accuracy; and the planar drift, which is the stability measure for the
zero-command episodes flagged by `zero_command`.

The **gait contact accuracy** is the fraction of control steps in which the
measured contact state matches the state scheduled by the Raibert planner. It is
reported per foot (`contact_acc_FR`, …), averaged over the four feet
(`gait_contact_accuracy`) and as the stricter fraction of steps where all four
feet agree at once (`gait_contact_accuracy_all_feet`). The Isaac backend reads
the schedule straight from `env.beta_contact_ref`, i.e. the exact signal the
policy is conditioned on; the MuJoCo backend re-derives it with `raibert_np.py`.

Drift is measured over the scored window, not from the reset pose, so the settle
phase does not contribute to it.

`python -m robustness_sweep.summarise <episodes.csv>` regenerates the summaries
on any machine - it uses only the standard library.

## The video

One continuous mp4 per sweep, with the batch, seed, gait, elapsed time, status
and commanded twist burned into every frame, plus `manifest.csv` mapping each
segment's timestamp back to its `episode_id`.

Filming every one of the 4380 episodes would produce twelve hours of video, so
the sweep films one episode per batch (Isaac) or `--video_per_gait` episodes per
gait on the first seed (MuJoCo). The Isaac subject is chosen to keep the gait
counts balanced and the viewport camera tracks it, with the rest of the batch
running in the background of the shot. `--video_stride`, `--video_every`,
`--video_per_gait` and `--num_envs` trade video length against coverage.

## Protocol details

Applied in both simulators:

* Both command terms are frozen for the whole episode. In Isaac they are
  rewritten into the command buffers after every `env.step`, because the command
  manager runs inside `step()` and would otherwise override the swept cell.
* Terminations: `base_contact` (>1 N on the base), `bad_orientation` (>0.8 rad
  from vertical) - both mirroring the training `TerminationsCfg` - and
  `fall_height` (base below 0.15 m for 10 consecutive steps). Anything that
  survives the full 10 s is recorded as `timeout`, which is what `survived` means.
* The reset yaw is fixed at 0. A random yaw combined with a body-frame velocity
  command makes part of the grid unreachable by construction.
* `push_robot` and observation noise are off by default (`--push_robot`,
  `--obs_noise` re-enable them); the reset joint-velocity randomisation is kept,
  since it is what makes the five seeds differ (`--deterministic_reset` removes it).

Isaac-specific:

* `base_contact` and `bad_orientation` are removed from the environment's
  termination manager and re-implemented in the harness. Isaac Lab resets an
  environment inside `step()` as soon as a term fires, which would corrupt that
  episode's metrics and make the per-gait settle window impossible to honour.
  With the terms removed the harness owns every episode boundary.
* The Raibert planner state (phase compensation, blended period and nominal
  height, latched reference feet) is primed at the start of every batch, so each
  episode starts at phase 0 with the parameters of the gait under test instead of
  inheriting the blend from the previous batch.
* Start-up mass and friction randomisation is kept on by default
  (`--no_domain_rand` turns it off). Each seed shuffles the assignment of command
  cells to environment slots, so a cell is evaluated on a different randomised
  robot in each of the five repetitions.
* `grid.GAIT_CONFIGS` mirrors the table in `velocity_env_cfg.py` for the MuJoCo
  backend's benefit; the Isaac backend asserts at start-up that the two still
  agree and refuses to run if they have drifted apart.

## Sim-to-sim parity

The MuJoCo backend reproduces the joint order, the default pose, the
`q_des = q_default + 0.25 · a` action, the Go2 HV actuator (PD 25/0.5 recomputed
every physics step, clipped by the 20.2 / 23.4 N·m torque-speed curve with its
13.5 rad/s knee), the 53-dimensional observation and the Raibert planner. Joint
damping and armature are forced to 0 and joint friction to 0.01, matching what
Isaac Lab actually configures - a Menagerie-style MJCF otherwise stacks its own
damping and armature on top of the software PD. Non-floor world geoms of the
scene file are disabled and hidden, because §4.1 is swept on flat ground and
these scenes usually ship an obstacle course (`--keep_scene_obstacles` keeps it).

**Action ordering.** `JointPositionActionCfg` in `velocity_env_cfg.py` lists the
joints as `FR, FL, RR, RL` but leaves `preserve_order` at its default of `False`,
so Isaac Lab resolves the names into the *articulation's* ordering, which the Go2
USD groups by joint type. Action 0 therefore drives `FL_hip`, not `FR_hip`:

```
FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf
```

This was verified against the running environment
(`action_manager._terms["JointPositionAction"]._joint_names` and a one-hot action
probe). The policy was trained with this mapping, so `mujoco_backend.py`
reproduces it via `ACTION_JOINT_ORDER`. Worth checking that anything else which
consumes the policy's actions - the deployment code in `deploy/robots/go2`, which
works in Unitree SDK order - uses the same mapping.

Remaining differences are the point of running both simulators rather than
harness bugs: total mass is 16.09 kg in the Isaac USD against 15.21 kg in the
MJCF, and the contact solvers differ.

## Validation performed

* Isaac observation layout confirmed against a recorded state trajectory: every
  one of the ten observation segments matches the corresponding state exactly.
* `raibert_np.py` reproduces `env.beta_contact_ref` and `env.beta_p_ref_rel_w`
  from the same state trajectory (exact once the environment's gait blending has
  converged).
* MuJoCo replays a recorded Isaac action sequence to within 0.007 m of base
  height with matching foot contact patterns.
* Termination path exercised by tightening the thresholds: `survival_steps`
  equals `n_metric_steps` in every row, and an episode that terminates during the
  settle window records no metrics at all.
* `--resume` reproduces the uninterrupted run's episode set and ordering exactly.
* Cross-simulator agreement on trot: gait contact accuracy 0.952 (Isaac) against
  0.948 (MuJoCo), base height 0.339 m against 0.340 m.

## Files

```
robustness_sweep/
  grid.py          command grid, gait table, protocol constants
  metrics.py       accumulator, attitude from projected gravity, CSV schema
  checkpoints.py   run/checkpoint resolution under logs/rsl_rl/<experiment>/
  video.py         annotated mp4 writer, manifest, gait-balanced subject picker
  raibert_np.py    NumPy port of the Raibert planner (MuJoCo backend)
  isaac_backend.py Isaac Lab rollout
  mujoco_backend.py MuJoCo rollout
  summarise.py     aggregation, also runnable standalone
run_robustness_sweep_isaac.py
run_robustness_sweep_mujoco.py
```
