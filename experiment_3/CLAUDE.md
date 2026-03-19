# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## What This Is

`experiment_3/` is Experiment 3 of the FYP: "A Comparative Analysis of Deep Q-Network and Actor-Critic Reinforcement Learning Algorithms". It trains DQN and PPO on Atari environments (Pong, Breakout, Seaquest) with CnnPolicy on GPU for 20M steps, then computes rliable statistical metrics.

**Experiment 1** (`rl_eval_bench/`) and **Experiment 2** (`atari_bench/`) are frozen — do not modify them.

**Algorithm families (discrete only):**
- DQN (value-based, off-policy, buffer_size=400K)
- PPO (actor-critic, on-policy)

## Commands

```bash
# Setup
uv sync

# Full pipeline: train all algos × envs → evaluate → figures
bash run.sh

# Train a single run
uv run python train.py --algo dqn --env Pong --device cuda
uv run python train.py --algo ppo --env Seaquest --device cuda

# Evaluate
uv run python evaluate.py --algo dqn
uv run python evaluate.py --algo ppo --envs Pong
uv run python evaluate.py --pairwise-only

# Generate figures (15 PNGs → results/figures/ + paper_overleaf assets)
uv run python generate_figures.py

# Interactive notebook
uv run marimo edit notebook/report.py

# Tests
uv run pytest tests/
```

## Architecture

**Pipeline: train.py → evaluate.py → generate_figures.py**

1. `train.py` — Trains DQN or PPO with CnnPolicy on Atari via `make_atari_env()` + `VecFrameStack(4)`. GPU by default. Logs both wall-clock and CPU process time. Saves per-seed models to `results/<algo>/<slug>/seed_<N>/`.
2. `evaluate.py` — Loads models, runs fresh eval via SB3's `evaluate_policy` (reads true returns from Monitor wrapper, bypassing ClipRewardEnv/EpisodicLifeEnv). Random baselines use raw `gym.make()`. Computes rliable metrics.
3. `generate_figures.py` — 15 publication-quality matplotlib figures.
4. `notebook/report.py` — Interactive marimo notebook with altair charts.

**Key module — `env_config.py`:**
- `ENV_REGISTRY` has Pong, Breakout, and Seaquest with `EnvSpec` dataclasses
- `N_SEEDS = 10`, `MASTER_SEED = 20260301`
- 20M timesteps per env, eval_freq=100K (200 checkpoints)
- Random baselines built into `EnvSpec` (Pong: -20.5, Breakout: 1.5, Seaquest: 68.4)

**Results directory layout:**
```
results/
  <algo>/<slug>/config.json
  <algo>/<slug>/seed_<N>/
  metrics/<algo>/evaluation_results.json
  metrics/<algo>/score_matrix.npy
  metrics/<algo>/raw_score_matrix.npy
  metrics/<algo>/learning_curves/<slug>.npz
  metrics/<algo>/sample_efficiency/<slug>.npz
  metrics/pairwise_poi.json
  figures/*.png  (15 figures)
  notebook/report.py  (marimo notebook)
```

## Key Differences from Previous Experiments

| | Experiment 1 | Experiment 2 | Experiment 3 |
|---|---|---|---|
| Algos | 5 (A2C, DQN, PPO, QR-DQN, RPPO) | 2 (DQN, PPO) | 2 (DQN, PPO) |
| Envs | 3 classic-control | 2 Atari | 3 Atari |
| Policy | MlpPolicy | CnnPolicy | CnnPolicy |
| Seeds | 15 | 10 | 10 |
| Budget | 200K-500K steps | 5M steps | 20M steps |
| DQN buffer | default | 100K | 400K |
| Timing | wall-clock only | wall-clock only | wall-clock + CPU process time |
| Device | CPU (deterministic) | GPU | GPU |

## Figures (15)

`learning_curves_pong.png`, `learning_curves_breakout.png`, `learning_curves_seaquest.png`, `combined_learning_curves.png`, `score_distribution.png`, `per_seed_heatmap.png`, `per_seed_boxswarm.png`, `final_performance.png`, `performance_profile.png`, `optimality_gap.png`, `poi_heatmap.png`, `timing_analysis.png`, `sample_efficiency_pong.png`, `sample_efficiency_breakout.png`, `sample_efficiency_seaquest.png`

## Rules

- **No Co-Authored-By**: Do not include Claude attribution in commit messages.
- **Do NOT modify rl_eval_bench/ or atari_bench/**: Experiments 1 and 2 are frozen.
- **Keep CLAUDE.md current** when changing structure.
