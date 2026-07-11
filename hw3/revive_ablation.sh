#!/usr/bin/env bash
# Re-run a subset of ablation configs (e.g. ones that died) at a fixed
# concurrency, streaming to logs/<config>.log exactly like run_ablation.sh so
# watch_progress.sh picks them up. Used when a few arms crash and the rest are
# still healthy — relaunch only the dead ones without disturbing the survivors.
#
# Usage:   MAX_PARALLEL=2 SEED=1 ./revive_ablation.sh <config> [<config> ...]
# Example: MAX_PARALLEL=2 ./revive_ablation.sh mspacman_abl_no_dueling mspacman
set -uo pipefail
cd "$(dirname "$0")"

SEED="${SEED:-1}"
PAR="${MAX_PARALLEL:-2}"
CONFIGS=("$@")
[ "${#CONFIGS[@]}" -eq 0 ] && { echo "usage: MAX_PARALLEL=N ./revive_ablation.sh <config>..."; exit 1; }

mkdir -p logs
# Clear all targets up front so queued configs read as "waiting" (not stale
# FROZEN) in the monitor until their turn to launch.
for cfg in "${CONFIGS[@]}"; do rm -f "logs/${cfg}_sd${SEED}.log" "logs/${cfg}_sd${SEED}.exit"; done

echo "reviving ${#CONFIGS[@]} configs, ${PAR} at a time, seed ${SEED}"
running=0
for cfg in "${CONFIGS[@]}"; do
  if [ "$running" -ge "$PAR" ]; then wait -n; running=$((running-1)); fi
  ( env CUDA_VISIBLE_DEVICES=0 uv run --no-sync src/scripts/run_dqn.py \
      -cfg "experiments/dqn/${cfg}.yaml" --seed "$SEED" --wandb_mode disabled \
      --num_final_videos 3 > "logs/${cfg}_sd${SEED}.log" 2>&1; echo $? > "logs/${cfg}_sd${SEED}.exit" ) &
  running=$((running+1))
  echo "launched ${cfg} (pid $!)"
done
wait
echo "=== revive complete ==="
