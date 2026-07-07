#!/usr/bin/env bash
# Run the MsPacman Rainbow leave-one-out ablation and zip the results.
# Usage:  ./run_ablation.sh [SEED]      (SEED defaults to 1)
# Run this from inside hw3/. GPU is used automatically if available.
set -euo pipefail
cd "$(dirname "$0")"

SEED="${1:-1}"

# Full Rainbow, then each config with one component removed.
CONFIGS=(
  mspacman_rainbow
  mspacman_abl_no_double
  mspacman_abl_no_dueling
  mspacman_abl_no_distributional
  mspacman_abl_no_noisy
  mspacman_abl_no_per
)

for cfg in "${CONFIGS[@]}"; do
  echo "==================================================================="
  echo "=== ${cfg}  (seed ${SEED})  $(date) ==="
  echo "==================================================================="
  uv run src/scripts/run_dqn.py \
    -cfg "experiments/dqn/${cfg}.yaml" \
    --seed "${SEED}" \
    --wandb_mode disabled
done

STAMP=$(date +%Y%m%d_%H%M%S)
ZIP="mspacman_ablation_seed${SEED}_${STAMP}.zip"
echo "=== zipping results into ${ZIP} ==="
# exp/ holds one folder per run: log.csv (metrics), log.pkl, flags.json, agent.pt.
# Videos (if any) are excluded to keep the archive small.
zip -r "${ZIP}" exp/ -x '*/videos/*'
echo "=== DONE. Send back: hw3/${ZIP} ==="
