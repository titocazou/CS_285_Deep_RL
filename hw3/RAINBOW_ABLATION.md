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
- **RAM:** each run allocates the Atari frame buffer, about **14 GB**. The runs
  are parallelized, so total RAM scales with how many run at once. The script
  auto-caps concurrency to the available RAM (~14 GB per run); with little RAM it
  falls back toward sequential. If this machine has under ~16 GB, stop and tell
  the user (see Troubleshooting for how to shrink the buffer).
- **GPU:** strongly recommended. Each run takes roughly 3-6 hours on a modern
  GPU. The script runs the six configs in parallel, one per GPU when several are
  present (round-robin via `CUDA_VISIBLE_DEVICES`), so with 6 GPUs the whole
  ablation finishes in about one run's time. On a single GPU it packs a few runs
  together (a batch-32 DQN underuses one GPU). On CPU it is impractically slow
  (days per run) and stays sequential.

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

`run_ablation.sh` launches the six configs in parallel (see the GPU/RAM note
above for how it picks the concurrency), each with `--wandb_mode disabled` (no
WandB account needed) and `--num_final_videos 3` (three rollout videos of the
trained agent, rendered once at the end of each run). It waits for all of them,
reports any failures, then zips the results. To force a specific number of
concurrent runs, set `MAX_PARALLEL`:

```bash
MAX_PARALLEL=2 ./run_ablation.sh 1
```

To detach from tmux without stopping the runs: press `Ctrl-b` then `d`.
Reattach later with `tmux attach -t ablation`.

Progress: each run streams its console output (the tqdm step counter) to
`logs/<config>.log`, and metrics stream to `exp/<run_name>/log.csv`. Watch one
with `tail -f logs/mspacman_rainbow.log` or `tail -f exp/<run_name>/log.csv`.

## Step 6 - collect the results

When `run_ablation.sh` finishes it prints the zip name and creates it in `hw3/`:

```
hw3/mspacman_ablation_seed1_<timestamp>.zip
```

The zip contains one folder per run, each with:

- `log.csv` - the metrics over training (this is the main deliverable)
- `log.pkl`, `flags.json` - the same metrics plus the run config
- `agent.pt` - the final model weights
- `videos/final_rollouts_step*.mp4` - three rollouts of the trained agent,
  rendered at the end of the run (mp4 writing works out of the box; the bundled
  `imageio-ffmpeg` is installed by `uv sync`, no system ffmpeg needed)

The zip also includes `logs/`, the per-run console output.

If `run_ablation.sh` was interrupted and you need to zip manually:

```bash
zip -r mspacman_ablation_manual.zip exp/ logs/
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
- **Out of memory.** Each run needs ~14 GB for the frame buffer. First lower the
  concurrency with `MAX_PARALLEL` (e.g. `MAX_PARALLEL=1`). If a single run still
  will not fit, shrink the replay capacity: in `src/scripts/run_dqn.py`, find
  where `PrioritizedMemoryEfficientReplayBuffer` and `MemoryEfficientReplayBuffer`
  are constructed (the image branch) and pass e.g. `capacity=250000` (about
  3.5 GB). This changes results slightly; note it in the report.
- **GPU not being used.** Check `uv run python -c "import torch; print(torch.cuda.is_available())"`.
  If `False`, the CUDA build of torch did not install; reinstall torch with the
  right CUDA wheel for this machine, then re-run.
- **Want a faster, rougher ablation.** Lower `total_steps` in each
  `experiments/dqn/mspacman_*.yaml` (keep it at least 100000; the built-in
  learning-rate and exploration schedules have a breakpoint at 20000 steps and
  assume a horizon well above that).

## Re-run after the reward-clipping fix (2026-07-08)

The first run of this ablation had a bug: `wrap_deepmind` never applied reward
clipping, but the distributional (C51) head keeps its atom support at
`v_min=-10, v_max=10`. Unclipped MsPacman returns are ~250, so every
distributional Bellman target got clamped to the top atom and the five
distributional configs saturated (logged `q_values` sat at ~8, against the ceiling
of 10). Only `no_distributional` (scalar Q) ran on the right scale. The fix wires
`ClipRewardEnv` into `wrap_deepmind` after `AtariPreprocessing`; it is already on
`main`. Re-run the whole ablation to get clean data.

Run these steps in order on the machine. Do not skip the sanity check.

### Step A - get the fix

```bash
cd CS_285_Deep_RL/hw3          # the existing checkout
git pull origin main
```

If `git pull` reports a conflict from a local edit, run `git stash` then pull
again. A completely fresh `git clone` of the repo also works and starts with an
empty `exp/` (if you clone fresh, skip Step E).

### Step B - confirm the fix is present

```bash
grep -n "ClipRewardEnv(env)" src/infrastructure/atari_wrappers.py
```

You should see one match inside `wrap_deepmind`, before the `FrameStack` line.
If there is no match, the pull did not land; stop and recheck the branch.

### Step C - dependencies

```bash
uv sync
```

Idempotent. If the environment is already set up from the first run this is a
no-op; run it anyway.

### Step D - sanity check (do not skip)

Confirm reward clipping is now active and that logged returns stay on the raw
game scale. This takes a few seconds and saves hours of wasted compute:

```bash
uv run python - <<'PY'
import sys; sys.path.insert(0, "src")
import gym
from infrastructure.atari_wrappers import wrap_deepmind
env = wrap_deepmind(gym.make("MsPacmanNoFrameskip-v4"))
env.reset()
seen, raw = set(), None
for _ in range(4000):
    _, r, done, info = env.step(env.action_space.sample())
    seen.add(float(r))
    if done:
        raw = info["episode"]["r"] if "episode" in info else None
        break
print("rewards agent sees:", sorted(seen))     # must be a subset of {-1.0, 0.0, 1.0}
print("logged raw episode return:", raw)         # must be a large raw score (hundreds)
PY
```

The rewards the agent sees must be within `{-1, 0, 1}`. The logged episode return
must still be a raw score in the hundreds (not clipped). If the agent sees
rewards like `10.0`, the fix is not active: go back to Step A.

### Step E - clear the old (clamped) runs

`run_ablation.sh` zips everything in `exp/` and `logs/`, so move the first run's
folders aside first, or the new zip will mix the clamped runs in:

```bash
mkdir -p ../old_clamped_runs && mv exp/* logs/* ../old_clamped_runs/ 2>/dev/null || true
```

### Step F - run all three seeds

Same as the original run, once per seed. Each call runs all six configs for one
seed; see the GPU/RAM notes above for how concurrency is chosen. Use `tmux` so
the jobs survive a disconnect:

```bash
tmux new -s ablation
./run_ablation.sh 1
./run_ablation.sh 2
./run_ablation.sh 3
```

The seed argument only names the zip; the runs accumulate in `exp/`. After seed 3
finishes, the file `mspacman_ablation_seed3_<timestamp>.zip` contains all 18
corrected runs (six configs x three seeds).

### Step G - hand back

Report, per run, the final `Eval_AverageReturn` from each `exp/<run>/log.csv`
(as in Step 7 above), and give the path of the seed-3 zip. As a quick gut-check
that the fix took: the `no_distributional` runs' logged `q_values` should have
dropped from ~250 (first run) to single digits, since every config now trains on
clipped rewards, while `Eval_AverageReturn` should still be on the same raw
~1000-2500 scale as before.
