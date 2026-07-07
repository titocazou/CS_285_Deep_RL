# MsPacman Rainbow ablation: setup and run guide

This is a task for Claude Code running on a fresh Linux machine. The goal is to
run a leave-one-out ablation of the Rainbow DQN agent on MsPacman, then hand the
results back as a single zip file. Follow the steps in order. Do not skip the
sanity check.

## What you are running

Six training runs on `MsPacmanNoFrameskip-v4`, each for 1,000,000 steps:

| config | what it is |
| --- | --- |
| `mspacman_rainbow` | full Rainbow: double-Q + dueling + distributional (C51) + noisy + PER |
| `mspacman_abl_no_double` | Rainbow with double-Q removed |
| `mspacman_abl_no_dueling` | Rainbow with dueling removed |
| `mspacman_abl_no_distributional` | Rainbow with C51 removed (scalar Q) |
| `mspacman_abl_no_noisy` | Rainbow with noisy nets removed (epsilon-greedy instead) |
| `mspacman_abl_no_per` | Rainbow with prioritized replay removed (uniform replay) |

The config files already exist under `experiments/dqn/`. You do not need to
write any code.

## Requirements (check these first)

- **Linux** with `git`, `zip`, a C/C++ build toolchain, and `swig` (needed to
  build `gym[box2d]`).
- **RAM:** each run allocates the Atari frame buffer, about **14 GB**. Runs are
  sequential, so ~16 GB free RAM is enough. If this machine has less, stop and
  tell the user before starting (see Troubleshooting for how to shrink it).
- **GPU:** strongly recommended. On a single modern GPU each run takes roughly
  3-6 hours, so the full ablation is most of a day. On CPU it is impractically
  slow (days per run). The code uses the GPU automatically when one is present.

## Step 0 - system packages

On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y git swig zip build-essential
```

## Step 1 - install uv

`uv` is the Python toolchain used by this homework (same as hw1:
https://github.com/berkeleydeeprlcourse/homework_spring2026/tree/main/hw1).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# make uv available in the current shell (or open a new shell):
source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
uv --version
```

## Step 2 - clone the repo

```bash
git clone https://github.com/titocazou/CS_285_Deep_RL.git
cd CS_285_Deep_RL/hw3
```

The `main` branch already contains the Rainbow code and the ablation configs.
If the clone asks for credentials (private repo), ask the user for access.

## Step 3 - install dependencies

```bash
uv sync
```

This creates `.venv/` and installs everything from `pyproject.toml`, including
`torch`, `gym==0.25.2`, and the Atari packages. The `accept-rom-license` extra
installs the Atari ROMs automatically; you do not need a separate AutoROM step.

## Step 4 - sanity check (do not skip)

Confirm the environment and the Atari pipeline work with a tiny 2000-step run
before committing to the long jobs:

```bash
uv run python - <<'PY'
import gym
e = gym.make("MsPacmanNoFrameskip-v4")
print("env OK:", e.observation_space.shape, "actions:", e.action_space.n)
PY
```

You should see `actions: 9`. If this errors on the ROM or on `gym`, fix it
(Troubleshooting) before continuing.

Then a short training smoke test that writes a result folder without needing a
WandB login (edit is temporary, revert after):

```bash
uv run src/scripts/run_dqn.py -cfg experiments/dqn/cartpole_per.yaml \
  --wandb_mode disabled --eval_interval 100000 &
sleep 20 && kill %1 2>/dev/null || true
ls exp/   # you should see a CartPole_* folder with a log.csv
```

If a `log.csv` appears, the pipeline works.

## Step 5 - run the ablation

The runs are long, so start them inside `tmux` (or `nohup`) so they survive a
disconnect:

```bash
tmux new -s ablation
./run_ablation.sh 1        # the argument is the random seed (default 1)
```

`run_ablation.sh` runs all six configs one after another with
`--wandb_mode disabled` (no WandB account needed), then zips the results. To
detach from tmux without stopping the run: press `Ctrl-b` then `d`. Reattach
later with `tmux attach -t ablation`.

Progress: each run prints a tqdm step counter. Metrics stream to
`exp/<run_name>/log.csv` as it goes, so you can `tail -f` a run's `log.csv` to
watch `Eval_AverageReturn` climb.

## Step 6 - collect the results

When `run_ablation.sh` finishes it prints the zip name and creates it in `hw3/`:

```
hw3/mspacman_ablation_seed1_<timestamp>.zip
```

The zip contains one folder per run, each with:

- `log.csv` - the metrics over training (this is the main deliverable)
- `log.pkl`, `flags.json` - the same metrics plus the run config
- `agent.pt` - the final model weights

If `run_ablation.sh` was interrupted and you need to zip manually:

```bash
zip -r mspacman_ablation_manual.zip exp/ -x '*/videos/*'
```

## Step 7 - hand back

Tell the user the zip is ready and give its path. They will copy it back to
their own machine. Report, per run, the final `Eval_AverageReturn` from each
`log.csv` so the user gets a quick summary.

## Troubleshooting

- **WandB wants a login / API key.** Always pass `--wandb_mode disabled`
  (the script already does). The CSV/pkl/checkpoint outputs do not depend on
  WandB. `WANDB_MODE=disabled` alone is not enough here, use the flag.
- **Atari ROM error** (`ROM ... not found`). `uv sync` should install ROMs via
  the `accept-rom-license` extra. If not, run `uv run AutoROM --accept-license`.
- **`swig` / box2d build failure.** Install `swig` (Step 0), then `uv sync`
  again.
- **Out of memory.** Each run needs ~14 GB for the frame buffer. To shrink it,
  lower the replay capacity: in `src/scripts/run_dqn.py`, find where
  `PrioritizedMemoryEfficientReplayBuffer` and `MemoryEfficientReplayBuffer`
  are constructed (the image branch) and pass e.g. `capacity=250000` (about
  3.5 GB). This changes results slightly; note it in the report.
- **GPU not being used.** Check `uv run python -c "import torch; print(torch.cuda.is_available())"`.
  If `False`, the CUDA build of torch did not install; reinstall torch with the
  right CUDA wheel for this machine, then re-run.
- **Want a faster, rougher ablation.** Lower `total_steps` in each
  `experiments/dqn/mspacman_*.yaml` (keep it at least 100000; the built-in
  learning-rate and exploration schedules have a breakpoint at 20000 steps and
  assume a horizon well above that).
