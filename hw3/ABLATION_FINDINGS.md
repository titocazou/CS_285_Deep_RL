# MsPacman Rainbow Ablation — Findings

**Game:** `MsPacmanNoFrameskip-v4`  ·  **Budget:** 1,000,000 env steps/run  ·  **Seeds:** 1, 2, 3
**Sweep:** 8 configurations × 3 seeds = 24 runs, all completed (≥990k steps).
**Date compiled:** 2026-07-11

This is a **leave-one-out ablation** of Rainbow DQN plus two reference points
(full Rainbow, and plain DQN with every extension off). The goal is to measure
each component's marginal contribution on MsPacman.

---

## 1. Headline result

**Full Rainbow is *not* the best configuration here — plain DQN is.** Turning
off individual Rainbow components tends to *help*, not hurt. The single most
harmful component is **NoisyNet exploration**, and it interacts strongly with
the **distributional (C51)** head.

Score = **MEAN5**: mean of each run's last-5 evaluation returns (a
noise-reduced final-performance estimate, matching the live monitor), then
averaged across the 3 seeds. Higher is better.

| Configuration                 |  sd1 |  sd2 |  sd3 | **MEAN** | ±SD | best MaxReturn |
|-------------------------------|-----:|-----:|-----:|---------:|----:|---------------:|
| **Plain DQN (no extensions)** | 1741 | 1474 | 2242 | **1819** | 318 | 4710 |
| Rainbow − Noisy nets          | 1571 | 1589 | 1391 | **1517** |  89 | 4570 |
| Rainbow − PER                 | 1367 | 1528 | 1200 | **1365** | 134 | 4140 |
| Rainbow − Double Q            |  894 |  817 | 1586 | **1099** | 346 | 3910 |
| Rainbow − Distributional (C51)| 1289 |  688 |  997 |  **991** | 245 | 4520 |
| Rainbow (full)                |  531 | 1090 |  649 |  **757** | 241 | 4080 |
| Rainbow − Noisy & Distrib. (`no_nd`) | 792 | 972 | 501 | **755** | 194 | 3910 |
| Rainbow − Dueling             |  515 |  802 |  626 |  **648** | 118 | 2040 |

**Marginal effect of *removing* each component** (vs. full Rainbow = 757):

| Removed component | Δ vs Rainbow | Reading |
|-------------------|-------------:|---------|
| Noisy nets        | **+760**     | Noisy hurts a lot (in the presence of C51) |
| PER               | **+608**     | PER hurts |
| Double Q          | **+342**     | Double Q hurts |
| Distributional    | **+234**     | C51 mildly hurts |
| Dueling           | **−109**     | Dueling is the **only** component that helps |

---

## 2. The Noisy × Distributional interaction

Four of the runs form a clean 2×2 factorial over (NoisyNet, Distributional)
while holding Double Q, Dueling, and PER **on**:

|                     | **Distrib. ON** | **Distrib. OFF** |
|---------------------|:---------------:|:----------------:|
| **Noisy ON**        | 757 (Rainbow)   | 991 (`no_distributional`) |
| **Noisy OFF**       | 1517 (`no_noisy`) | 755 (`no_nd`)  |

The effect of NoisyNet **flips sign** depending on the value head:

- With **C51 on**, removing Noisy is a huge win: **757 → 1517**.
- With **C51 off**, removing Noisy *hurts*: **991 → 755**.

So NoisyNet and C51 are individually harmful *and* redundant/antagonistic
together. The best cell of this 2×2 is `no_noisy` (Noisy off, C51 on) = 1517 —
but even that is beaten by plain DQN (1819), which also drops Double/Dueling/PER.

Also notable: adding Double + Dueling + PER on top of the scalar
epsilon-greedy base takes plain DQN from **1819 → 755** (`no_nd`). In this
regime those three extensions collectively hurt, even though **Dueling helps**
inside the full-Rainbow context (removing it drops 757 → 648). Component
contributions are **not additive** — they depend on what else is enabled.

---

## 3. Exploration verification (`no_noisy` / `no_nd` correctness check)

The runs that disable NoisyNet must not lose exploration. **Verified they do
not.** In `atari_dqn_config` (`src/configs/dqn_config.py:195–206`), when
`use_noisy: false` the exploration schedule is *not* zeroed — it falls back to
epsilon-greedy:

```
ε: 1.0  →  1.0 (held to step 20k)  →  0.01 (linear to step 500k)  →  0.01 after
```

`ConstantSchedule(0.0)` (no epsilon exploration) is used **only** when
`use_noisy: true`, where weight noise supplies exploration instead.
`DQNAgent.get_action` applies ε correctly (ε=1.0 ⇒ always random, ε=0.01 ⇒
mostly greedy).

Confirmed empirically against the logged `epsilon` column (matches the schedule
to 4 decimals):

| run          | step   | logged ε | schedule predicts |
|--------------|-------:|---------:|------------------:|
| no_noisy sd3 | 126000 | 0.78138  | 0.7814 |
| no_nd sd1    |  59000 | 0.91956  | 0.9196 |
| no_nd sd2    |  56000 | 0.92575  | 0.9257 |
| no_nd sd3    |  54000 | 0.92988  | 0.9299 |

**No code change was required** — the epsilon-greedy fallback was already
correct.

---

## 4. Caveats — read the numbers cautiously

- **Only 3 seeds, high variance.** Per-arm SD reaches 346. Differences within
  ~1 SD are not reliable: Rainbow (757) vs `no_nd` (755) is a tie; `no_dueling`
  (648) is within noise of Rainbow.
- **Single game, single budget.** MsPacman at exactly 1M steps. Rankings can
  differ on other Atari games and at larger budgets (Rainbow's advantages are
  usually reported at 50–200M frames).
- **C51 support was recalibrated** earlier in this project: the distributional
  runs use `num_atoms=241`, `v_max=2600` (Δz≈10.8) after finding the original
  `v_max=6000`/51-atom support (Δz≈120) far exceeded the per-step reward scale.
  Distributional still underperforms even after that fix.
- **MEAN5 is a smoothed *final-window* metric**, not area-under-curve; it says
  nothing about sample efficiency earlier in training. See each run's
  `log.csv` for full learning curves.

---

## 5. Operational notes (how this sweep was run)

- **Orchestration:** `finish_sweep.sh` drains the 8×3 queue at a fixed
  concurrency (PAR=4 — the RAM ceiling on this 62 GB box, ~12.5 GB/run), is
  seed-major (seed 1 finishes first), skips completed pairs, and retries
  crashes. Launch priority = order of its `CONFIGS` array.
- **No checkpoint/resume.** `run_dqn.py` cannot resume; killing ("pausing") a
  run discards its progress and it restarts from step 0 when relaunched. Pausing
  is only cheap for barely-started runs.
- **Naming:** the "no Noisy + no Distributional" arm is `no_nd`
  (`experiments/dqn/mspacman_abl_no_nd.yaml`, `exp_name: no_nd`). The short
  label also keeps `watch_progress.sh`'s table aligned (fits the 17-wide
  SCENARIO column; the old `no_noisy_no_distributional` overflowed it).
- **Monitoring:** `./watch_progress.sh` for a live 24-row status board.

---

## 6. What's in this archive

```
ABLATION_FINDINGS.md            this document
experiments/dqn/*.yaml          all run configs (mspacman_*, plus cartpole/lunarlander)
exp/MsPacman_<scen>_sd<n>_*/    per-run outputs:
    log.csv                     eval + train metrics per step (incl. epsilon, per_beta, losses)
    flags.json                  resolved hyperparameters
    agent.pt                    final trained weights
    videos/final_rollouts_*.mp4 final greedy-policy rollouts
logs/<config>_sd<n>.log         raw console output per run
finish_sweep.sh / run_ablation.sh / revive_ablation.sh / watch_progress.sh
                                orchestration + monitoring scripts
```

**Reproduce a single arm:**
```bash
uv run --no-sync src/scripts/run_dqn.py \
  -cfg experiments/dqn/mspacman_abl_no_nd.yaml --seed 1 --wandb_mode disabled
```
