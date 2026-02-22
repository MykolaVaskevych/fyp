# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## What This Is

`atari_bench/` is Experiment 2 of the FYP: "A Comparative Analysis of Deep Q-Network and Actor-Critic Reinforcement Learning Algorithms". It trains DQN and PPO on Atari environments (Pong, Breakout) with CnnPolicy on GPU, then computes rliable statistical metrics.

**Experiment 1** (`rl_eval_bench/`) is frozen — do not modify it.

**Algorithm families (discrete only):**
- DQN (value-based, off-policy)
- PPO (actor-critic, on-policy)

## Commands

```bash
# Setup
uv sync

# Full pipeline: train all algos × envs → evaluate → figures
bash run.sh

# Train a single run
uv run python train.py --algo dqn --env Pong --device cuda
uv run python train.py --algo ppo --env Breakout --device cuda

# Evaluate
uv run python evaluate.py --algo dqn
uv run python evaluate.py --algo ppo --envs Pong
uv run python evaluate.py --pairwise-only

# Generate figures
uv run python generate_figures.py

# Tests
uv run pytest tests/
```

## Architecture

**Pipeline: train.py → evaluate.py → generate_figures.py**

1. `train.py` — Trains DQN or PPO with CnnPolicy on Atari via `make_atari_env()` + `VecFrameStack(4)`. GPU by default. Saves per-seed models to `results/<algo>/<slug>/seed_<N>/`.
2. `evaluate.py` — Loads models, runs fresh eval with Atari wrappers, computes rliable metrics. Same output format as Experiment 1.
3. `generate_figures.py` — Publication-quality matplotlib plots.

**Key module — `env_config.py`:**
- `ENV_REGISTRY` has Pong and Breakout with `EnvSpec` dataclasses
- `N_SEEDS = 10`, `MASTER_SEED = 20260215`
- Random baselines built into `EnvSpec` (Pong: -20.5, Breakout: 1.5)

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
  figures/*.png
```

## Key Differences from Experiment 1

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Algos | 5 (A2C, DQN, PPO, QR-DQN, RPPO) | 2 (DQN, PPO) |
| Envs | 3 classic-control | 2 Atari |
| Policy | MlpPolicy | CnnPolicy |
| Seeds | 15 | 10 |
| Budget | 200K-500K steps | 5M steps |
| Device | CPU (deterministic) | GPU (statistical validity) |

## Rules

- **No Co-Authored-By**: Do not include Claude attribution in commit messages.
- **Do NOT modify rl_eval_bench/**: Experiment 1 is frozen.
- **Keep CLAUDE.md current** when changing structure.
