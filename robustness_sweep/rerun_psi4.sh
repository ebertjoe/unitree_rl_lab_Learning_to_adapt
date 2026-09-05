#!/usr/bin/env bash
# Sweep 2026-06-12_12-51-26 -- the psi(err2*4.0) counterpart of 2026-09-02_14-22-00.
#
# Both are 53-dim with the MEASURED v_B Raibert planner, identical reward weights
# (-1.5 / 15.0 / -10.0 / -5.0, no extra terms), identical command ranges and gait
# table, seed 1, and both evaluated at model_14800.pt. The ONLY difference is
#     reward = psi(err2 * 4.0)   vs   reward = psi(err2)
# The observations.py blob 733a6f2 was reconstructed from the run's own git diff
# applied to its recorded base (eca8cae) and verified by hash before being stored.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
OBS="source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/observations.py"
VCFG="source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py"
L="logs/rsl_rl/unitree_go2_locomotion_paper"
RUN="2026-06-12_12-51-26"; WANT="733a6f282b6f8b19d2bbbdb85c5b94449cf05928"
CKPT="model_14800.pt"; OUT_TAG="it14800_placement"
LOGDIR="$REPO/robustness_sweep/rerun_logs_psi4"; mkdir -p "$LOGDIR"
BACKUP="$LOGDIR/_worktree_backup"; mkdir -p "$BACKUP"
cp "$OBS" "$BACKUP/observations.py.orig"; cp "$VCFG" "$BACKUP/velocity_env_cfg.py.orig"
restore(){ cp "$BACKUP/observations.py.orig" "$OBS"; cp "$BACKUP/velocity_env_cfg.py.orig" "$VCFG";
           echo "[cleanup] restored: $(git hash-object "$OBS") / $(git hash-object "$VCFG")"; }
trap restore EXIT
git checkout -- "$VCFG"
git cat-file blob "$WANT" > "$OBS"
got="$(git hash-object "$OBS")"
[ "$got" = "$WANT" ] || { echo "[FATAL] hash mismatch $got != $WANT"; exit 1; }
echo "[restore] observations.py = $got (verified)"
grep -q "0.5 \* Tst \* v_B" "$OBS" && echo "[restore] Raibert planner: MEASURED v_B" \
  || { echo "[FATAL] unexpected planner"; exit 1; }
find source -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
EXP="$L/$RUN/exported_it14800"
[ -f "$EXP/policy.pt" ] || python robustness_sweep/export_checkpoint.py --headless \
    --run "$RUN" --checkpoint "$CKPT" --out_dir "$EXP" > "$LOGDIR/export.log" 2>&1
grep -E "^\[export\]" "$LOGDIR/export.log" 2>/dev/null | sed 's/^/  /' || true
OUT="$L/$RUN/robustness_sweep/isaac_$OUT_TAG"
echo "[sweep] -> $OUT"
python run_robustness_sweep_isaac.py --headless --run "$RUN" --checkpoint "$CKPT" \
  --policy_jit "$EXP/policy.pt" --num_envs 876 --out_dir "$OUT" \
  > "$LOGDIR/sweep.log" 2>&1
tail -14 "$LOGDIR/sweep.log" | sed 's/^/  /'
echo "done -> $OUT"
