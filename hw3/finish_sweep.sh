#!/usr/bin/env bash
# Hands-off finisher for the MsPacman Rainbow ablation (7 configs x 3 seeds).
#
# Drains the ENTIRE remaining queue at a fixed, RAM-safe concurrency, then zips.
# Designed to run detached and finish with nobody watching:
#   - skips (config,seed) pairs already complete (any exp dir >= 990k steps)
#   - never relaunches a pair that already has a live worker (counts orphaned
#     workers from earlier launchers toward the budget, so it self-coordinates)
#   - global concurrency budget = PAR live run_dqn.py workers, GPU is 100% at 4
#     and RAM peaks ~12.5 GB/run, so 4 is the optimum on a 62 GB box
#   - retries a crashed pair up to MAX_RETRIES times, then gives up on it
#   - seed-major, config-order priority so seed 1 finishes first
#
# Usage:  PAR=4 nohup setsid ./finish_sweep.sh > logs/finish_sweep.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

PAR="${PAR:-4}"
SEEDS="${SEEDS:-1 2 3}"
MAX_RETRIES="${MAX_RETRIES:-3}"
POLL="${POLL:-45}"        # seconds between capacity checks
STAGGER="${STAGGER:-25}"  # seconds after a launch before considering the next

# Order = launch priority (seed-major). no_nd (no Noisy + no Distributional) is
# front-loaded: it's the only arm with zero completed runs, so it gets first
# claim on every free slot until all three seeds are underway.
CONFIGS=(
  mspacman_abl_no_nd
  mspacman_rainbow
  mspacman
  mspacman_abl_no_distributional
  mspacman_abl_no_double
  mspacman_abl_no_dueling
  mspacman_abl_no_noisy
  mspacman_abl_no_per
)

scen_of() {
  case "$1" in
    mspacman_rainbow)                echo rainbow ;;
    mspacman_abl_no_double)          echo no_double ;;
    mspacman_abl_no_dueling)         echo no_dueling ;;
    mspacman_abl_no_distributional)  echo no_distributional ;;
    mspacman_abl_no_noisy)           echo no_noisy ;;
    mspacman_abl_no_per)             echo no_per ;;
    mspacman_abl_no_nd)              echo no_nd ;;
    mspacman)                        echo dqn ;;
  esac
}

# complete = some exp dir for this (scenario,seed) reached >= 990k steps
is_done() {
  local scen sd d bs
  scen=$(scen_of "$1"); sd="$2"
  for d in exp/MsPacman_${scen}_sd${sd}_*/; do
    [ -f "${d}log.csv" ] || continue
    bs=$(tail -1 "${d}log.csv" | awk -F, '{printf "%d",$6+0}')
    [ "${bs:-0}" -ge 990000 ] && return 0
  done
  return 1
}

# is this exact (config,seed) currently a live worker? (matches any launcher)
is_running() {
  pgrep -af 'run_dqn.py' | grep -v grep \
    | grep -q -- "-cfg experiments/dqn/${1}.yaml --seed ${2} "
}

# total live workers across the whole box (the shared concurrency budget).
# Each worker is a process *pair* (uv-run wrapper + python child), so dedupe by
# the (config,seed) signature to count distinct runs, not raw processes.
live_count() {
  pgrep -af 'run_dqn.py' | grep -v grep \
    | grep -oE -- '-cfg experiments/dqn/[a-z_]+\.yaml --seed [0-9]+' \
    | sort -u | wc -l | tr -d ' '
}

launch() {
  local cfg="$1" sd="$2"
  rm -f "logs/${cfg}_sd${sd}.exit"
  ( env CUDA_VISIBLE_DEVICES=0 uv run --no-sync src/scripts/run_dqn.py \
      -cfg "experiments/dqn/${cfg}.yaml" --seed "$sd" \
      --wandb_mode disabled --num_final_videos 3 \
      > "logs/${cfg}_sd${sd}.log" 2>&1; echo $? > "logs/${cfg}_sd${sd}.exit" ) &
  echo "$(date '+%F %T') LAUNCH ${cfg} sd${sd} (pid $!)  [try ${3}]"
}

declare -A tries
mkdir -p logs
echo "$(date '+%F %T') finish_sweep start: PAR=${PAR} seeds='${SEEDS}' retries=${MAX_RETRIES}"

while :; do
  # anything not yet complete? (termination check)
  notdone=0
  for sd in $SEEDS; do for cfg in "${CONFIGS[@]}"; do
    is_done "$cfg" "$sd" || { notdone=1; break 2; }
  done; done
  [ "$notdone" -eq 0 ] && { echo "$(date '+%F %T') all pairs complete"; break; }

  # pick the highest-priority launchable pair (not done, not running, tries left)
  pick=''
  for sd in $SEEDS; do
    for cfg in "${CONFIGS[@]}"; do
      is_done "$cfg" "$sd" && continue
      is_running "$cfg" "$sd" && continue
      [ "${tries[${cfg}:${sd}]:-0}" -gt "$MAX_RETRIES" ] && continue
      pick="${cfg}:${sd}"; break 2
    done
  done

  live=$(live_count)
  if [ -n "$pick" ] && [ "$live" -lt "$PAR" ]; then
    cfg="${pick%:*}"; sd="${pick#*:}"
    tries[$pick]=$(( ${tries[$pick]:-0} + 1 ))
    launch "$cfg" "$sd" "${tries[$pick]}"
    sleep "$STAGGER"
  else
    # at capacity, or nothing launchable right now (all remaining are running
    # or exhausted their retries) -> wait for a slot / a crash to surface
    sleep "$POLL"
  fi
done

STAMP=$(date +%Y%m%d_%H%M%S)
ZIP="mspacman_ablation_ALL_${STAMP}.zip"
echo "$(date '+%F %T') zipping -> ${ZIP}"
zip -r "$ZIP" exp/ logs/ >/dev/null 2>&1
echo "$(date '+%F %T') === SWEEP COMPLETE -> hw3/${ZIP} ==="
# report any pair we gave up on
for sd in $SEEDS; do for cfg in "${CONFIGS[@]}"; do
  is_done "$cfg" "$sd" || echo "  INCOMPLETE: ${cfg} sd${sd} (tries=${tries[${cfg}:${sd}]:-0})"
done; done
