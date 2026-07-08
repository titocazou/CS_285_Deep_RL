#!/usr/bin/env bash
# Run the MsPacman Rainbow leave-one-out ablation in parallel, then zip results.
#
# Usage:   ./run_ablation.sh [SEED]           (SEED defaults to 1)
# Tuning:  MAX_PARALLEL=N ./run_ablation.sh   (force N concurrent runs)
#
# Run from inside hw3/ after `uv sync`. Each run needs ~14 GB RAM, so the number
# of concurrent runs is capped by both available RAM and (if present) GPU count.
# Runs are spread across GPUs round-robin via CUDA_VISIBLE_DEVICES.
set -uo pipefail
cd "$(dirname "$0")"

SEED="${1:-1}"
RAM_PER_RUN_GB=15   # ~14 GB frame buffer + headroom

CONFIGS=(
  mspacman_rainbow
  mspacman_abl_no_double
  mspacman_abl_no_dueling
  mspacman_abl_no_distributional
  mspacman_abl_no_noisy
  mspacman_abl_no_per
  mspacman
)

# --- how many GPUs? ---
if command -v nvidia-smi >/dev/null 2>&1; then
  NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
  NGPU=0
fi

# --- how much RAM can we spare? cap concurrency so we do not swap/OOM ---
if command -v free >/dev/null 2>&1; then
  AVAIL_GB=$(free -g | awk '/^Mem:/{print $7}')
else
  AVAIL_GB=16   # unknown (e.g. macOS): assume modest
fi
RAM_CAP=$(( AVAIL_GB / RAM_PER_RUN_GB ))
[ "$RAM_CAP" -lt 1 ] && RAM_CAP=1

# --- decide concurrency ---
if [ -n "${MAX_PARALLEL:-}" ]; then
  PAR="$MAX_PARALLEL"
elif [ "$NGPU" -ge 2 ]; then
  PAR="$NGPU"          # one run per GPU
elif [ "$NGPU" -eq 1 ]; then
  PAR=3                # a small DQN underuses one GPU; pack a few
else
  PAR=1                # CPU only: parallel would just thrash
fi
[ "$PAR" -gt "$RAM_CAP" ] && PAR="$RAM_CAP"          # never exceed the RAM cap
[ "$PAR" -gt "${#CONFIGS[@]}" ] && PAR="${#CONFIGS[@]}"

mkdir -p logs
echo "GPUs=$NGPU  avail_RAM=${AVAIL_GB}GB  ram_cap=$RAM_CAP  -> running $PAR in parallel"
echo "per-run output streams to logs/<config>.log"

running=0
i=0
for cfg in "${CONFIGS[@]}"; do
  # Throttle: block until fewer than PAR jobs are running. We track the count
  # ourselves and use `wait -n` (reap the next finished job) because `jobs`
  # inside $(...) runs in a subshell that never sees the parent's jobs, so the
  # old `jobs -rp | wc -l` check always read 0 and launched everything at once.
  if [ "$running" -ge "$PAR" ]; then
    wait -n
    running=$(( running - 1 ))
  fi

  if [ "$NGPU" -gt 0 ]; then
    gpu=$(( i % NGPU ))
    prefix="CUDA_VISIBLE_DEVICES=$gpu"
    gpu_flag=""
    echo "launch ${cfg} on GPU ${gpu}"
  else
    prefix=""
    gpu_flag="--no_gpu"
    echo "launch ${cfg} on CPU"
  fi

  rm -f "logs/${cfg}.exit"
  ( env ${prefix} uv run --no-sync src/scripts/run_dqn.py \
      -cfg "experiments/dqn/${cfg}.yaml" \
      --seed "${SEED}" \
      --wandb_mode disabled \
      --num_final_videos 3 \
      ${gpu_flag} \
      > "logs/${cfg}.log" 2>&1; echo $? > "logs/${cfg}.exit" ) &
  running=$(( running + 1 ))
  i=$(( i + 1 ))
done

# Drain the remaining jobs, then collect each run's recorded exit code.
wait
fail=0
for cfg in "${CONFIGS[@]}"; do
  ec=$(cat "logs/${cfg}.exit" 2>/dev/null || echo 1)
  if [ "$ec" != "0" ]; then
    echo "FAILED: ${cfg} (exit ${ec}, see logs/${cfg}.log)"
    fail=1
  fi
done

STAMP=$(date +%Y%m%d_%H%M%S)
ZIP="mspacman_ablation_seed${SEED}_${STAMP}.zip"
echo "=== zipping results into ${ZIP} ==="
# exp/ has one folder per run (log.csv, log.pkl, flags.json, agent.pt,
# videos/final_rollouts_*.mp4); logs/ has the console output of each run.
zip -r "${ZIP}" exp/ logs/
echo "=== DONE (failures=${fail}). Send back: hw3/${ZIP} ==="
