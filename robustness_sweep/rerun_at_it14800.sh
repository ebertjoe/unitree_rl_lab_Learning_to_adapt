#!/usr/bin/env bash
# Re-run the five §4.1 sweeps at a common training iteration (model_14800.pt).
#
# For every run this restores the exact observations.py the run was TRAINED with,
# verifies it by git blob hash, exports that checkpoint to its own TorchScript,
# and sweeps into a new output directory. The existing `robustness_sweep/isaac/`
# results and `exported/policy.pt` of every run are left untouched.
#
# The blob hashes below are the training-time state, taken from each run's
# `git/*.diff` (the "index <base>..<result>" line) and independently confirmed
# against `observations_py_blob` in each run's sweep config.json.
#
#   run                    obs blob (training)                       commit
#   2026-08-16_16-51-37    eca8caeac53303c90ae99e6d80025ba61fa786cf  0840c87
#   2026-08-16_22-05-59    ae743a29c718cedf808a8fadd4edd1b2b18f14e9  def5f3d
#   2026-08-17_09-10-28    996f45d34ea55effa36f937f28efd1f994593138  674b7b1
#   2026-08-17_20-51-38    296f3f3a4098112fe9fa3c5e97e5bff886dc7211  958a644
#   2026-09-02_14-22-00    af8c23838ad5bd7b9e723b8ef16ade7b283e47c4  (no commit)
#
# The last one has no commit; its blob is loose in the object database and its
# base tree (967406a) differs from the others only in a comment and an unused
# UNITREE_ROS_DIR path, so restoring observations.py alone is sufficient and
# every run is evaluated under an identical environment.
#
# The 53-dim runs differ ONLY inside beta_l_raibert (measured v_B vs feedforward
# v_cmd) and have identical observation widths, so the sweep's own width check
# cannot distinguish them. The blob hash is what actually guarantees the right
# planner, which is why it is verified and the sweep aborted on any mismatch.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

OBS="source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/observations.py"
VCFG="source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py"
L="logs/rsl_rl/unitree_go2_locomotion_paper"
CKPT="model_14800.pt"
TAG="it14800"                       # names the TorchScript export dir (reused across re-runs)
OUT_TAG="${OUT_TAG:-$TAG}"          # names the sweep output dir; override to keep older results
NUM_ENVS="${NUM_ENVS:-876}"
LOGDIR="$REPO/robustness_sweep/rerun_logs_$OUT_TAG"
mkdir -p "$LOGDIR"

RUNS=(
  "2026-08-16_16-51-37:eca8caeac53303c90ae99e6d80025ba61fa786cf"
  "2026-08-16_22-05-59:ae743a29c718cedf808a8fadd4edd1b2b18f14e9"
  "2026-08-17_09-10-28:996f45d34ea55effa36f937f28efd1f994593138"
  "2026-08-17_20-51-38:296f3f3a4098112fe9fa3c5e97e5bff886dc7211"
  "2026-09-02_14-22-00:af8c23838ad5bd7b9e723b8ef16ade7b283e47c4"
)

# ---------------------------------------------------------------------------
# Save the working tree so the session's own edits survive, whatever happens.
# ---------------------------------------------------------------------------
BACKUP="$LOGDIR/_worktree_backup"
mkdir -p "$BACKUP"
cp "$OBS"  "$BACKUP/observations.py.orig"
cp "$VCFG" "$BACKUP/velocity_env_cfg.py.orig"
echo "[setup] backed up working-tree observations.py ($(git hash-object "$OBS"))"
echo "[setup] backed up working-tree velocity_env_cfg.py ($(git hash-object "$VCFG"))"

restore_worktree() {
  cp "$BACKUP/observations.py.orig"      "$OBS"
  cp "$BACKUP/velocity_env_cfg.py.orig"  "$VCFG"
  echo "[cleanup] working tree restored:"
  echo "          observations.py     $(git hash-object "$OBS")"
  echo "          velocity_env_cfg.py $(git hash-object "$VCFG")"
}
trap restore_worktree EXIT

# The uncommitted reward-weight edit (vcmd_tracking 15.0 -> 17.5) is in-progress
# work for a future run and was not present for any of these five. Rewards are
# not scored by the sweep, but the tree is put back to the committed state so
# "restored to the training state" is true of every file, not just the ones that
# happen to matter.
git checkout -- "$VCFG"
echo "[setup] velocity_env_cfg.py -> committed state ($(git hash-object "$VCFG"))"

echo
for entry in "${RUNS[@]}"; do
  run="${entry%%:*}"
  want="${entry##*:}"
  echo "=============================================================="
  echo "  $run   ->  $CKPT"
  echo "=============================================================="

  if [ ! -f "$L/$run/$CKPT" ]; then
    echo "[skip] $L/$run/$CKPT missing"; continue
  fi

  # ---- restore the training-time observation code ------------------------
  git cat-file blob "$want" > "$OBS"
  got="$(git hash-object "$OBS")"
  if [ "$got" != "$want" ]; then
    echo "[FATAL] observations.py hash mismatch: want $want got $got"; exit 1
  fi
  echo "[restore] observations.py = $got  (verified)"
  # Report which planner variant this actually is, as a human-readable check.
  if grep -q "0.5 \* Tst \* v_B" "$OBS"; then
    echo "[restore] Raibert planner: MEASURED v_B  (+ K feedback)"
  elif grep -q "0.5 \* Tst \* v_cmd" "$OBS"; then
    echo "[restore] Raibert planner: FEEDFORWARD v_cmd"
  else
    echo "[restore] Raibert planner: (pattern not recognised - inspect manually)"
  fi
  find source -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

  # ---- export this checkpoint to its own TorchScript ---------------------
  EXP_DIR="$L/$run/exported_$TAG"
  if [ -f "$EXP_DIR/policy.pt" ]; then
    echo "[export] reusing $EXP_DIR/policy.pt"
  else
    echo "[export] $CKPT -> $EXP_DIR"
    python robustness_sweep/export_checkpoint.py --headless \
      --run "$run" --checkpoint "$CKPT" --out_dir "$EXP_DIR" \
      > "$LOGDIR/${run}_export.log" 2>&1 || {
        echo "[FATAL] export failed; see $LOGDIR/${run}_export.log"
        tail -25 "$LOGDIR/${run}_export.log"; exit 1; }
    grep -E "^\[export\]" "$LOGDIR/${run}_export.log" | sed 's/^/          /'
  fi

  # ---- sweep -------------------------------------------------------------
  OUT="$L/$run/robustness_sweep/isaac_$OUT_TAG"
  if [ -f "$OUT/summary_by_gait.csv" ]; then
    echo "[sweep] already complete: $OUT"; echo; continue
  fi
  echo "[sweep] -> $OUT   (num_envs=$NUM_ENVS)"
  python run_robustness_sweep_isaac.py --headless \
    --run "$run" --checkpoint "$CKPT" \
    --policy_jit "$EXP_DIR/policy.pt" \
    --num_envs "$NUM_ENVS" --out_dir "$OUT" \
    > "$LOGDIR/${run}_sweep.log" 2>&1 || {
      echo "[FATAL] sweep failed; see $LOGDIR/${run}_sweep.log"
      tail -25 "$LOGDIR/${run}_sweep.log"; exit 1; }
  tail -14 "$LOGDIR/${run}_sweep.log" | sed 's/^/          /'
  echo
done

echo "=============================================================="
echo "all sweeps done -> $L/<run>/robustness_sweep/isaac_$OUT_TAG"
echo "=============================================================="
