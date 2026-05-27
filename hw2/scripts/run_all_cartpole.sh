#!/usr/bin/env bash
# Run all CartPole PG ablations sequentially from the hw2 directory.
set -euo pipefail

cd "$(dirname "$0")/.."

run() {
  echo ""
  echo "========== $* =========="
  uv run src/scripts/run.py "$@"
}

# Small batch (b=1000) -> exp/cartpole/
run --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
run --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg
run --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na
run --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na

# Large batch (b=4000) -> exp/cartpole_lb/
run --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb
run --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg
run --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na
run --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na

echo ""
echo "All runs finished."
