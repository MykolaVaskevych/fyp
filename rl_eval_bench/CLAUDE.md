# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

`rl_eval_bench` is the RL evaluation benchmark within a larger FYP (Final Year Project): "A Comparative Analysis of Deep Q-Network and Actor-Critic Reinforcement Learning Algorithms". It trains agents on discrete classic control environments with deterministic seeding, then computes rliable statistical metrics (IQM, bootstrap CIs, performance profiles, optimality gap) following Agarwal et al. (2021).

**Algorithm families (all discrete):**
- DQN family: `dqn`, `qrdqn` (QR-DQN from sb3-contrib)
- Actor-Critic family: `a2c`, `ppo`, `rppo` (RecurrentPPO from sb3-contrib, uses MlpLstmPolicy)

This is a subdirectory of the parent FYP repo (`fyp/`), not a standalone git repo. It has its own `pyproject.toml` and `uv.lock`.

## Commands

```bash
# Setup
uv sync

# Full pipeline: train all compatible envs → evaluate → open report
bash run.sh           # defaults to a2c
bash run.sh ppo       # train PPO on all compatible envs

# Train a single environment
uv run python train.py --algo a2c --env CartPole-v1
uv run python train.py --algo ppo --env LunarLander-v3
uv run python train.py --algo dqn --env Acrobot-v1
uv run python train.py --algo qrdqn --env CartPole-v1
uv run python train.py --algo rppo --env CartPole-v1

# Evaluate (computes rliable metrics for all trained envs)
uv run python evaluate.py --algo a2c
uv run python evaluate.py --algo ppo --envs CartPole-v1   # single env
uv run python evaluate.py --algo a2c --episodes 100        # more eval episodes
uv run python evaluate.py --pairwise-only                  # recompute P(X>Y) only

# Verify deterministic seeding works
uv run python train.py --algo a2c --env CartPole-v1 --verify

# View the marimo report notebook
uv run marimo edit notebook/report.py

# Tests
uv run pytest tests/
uv run pytest tests/test_pipeline.py::TestAlgoEnvCompat

# Wipe all results (models, metrics, figures)
bash clear.sh
```

## Architecture

**Pipeline: train.py → evaluate.py → notebook/report.py**

1. `train.py` — Trains an algorithm (A2C, DQN, PPO, QR-DQN, RecurrentPPO) across N seeds with deterministic seeding. Uses `ALGO_MAP` dict for algorithm class lookup and `ALGO_POLICY` for policy selection (MlpLstmPolicy for rppo, MlpPolicy for others). Saves per-seed models and evaluation logs to `results/<algo>/<env-slug>/seed_<N>/`. Validates algo-env compatibility before training.
2. `evaluate.py` — Loads trained models, runs fresh evaluation episodes, computes per-environment and cross-environment rliable metrics. Algorithm class lookup via `ALGO_CLASSES` dict. Handles LSTM states in `fresh_evaluate()` for RecurrentPPO. Saves metrics to `results/metrics/<algo>/`. Skips cross-env metrics when < 2 environments. Also computes pairwise P(X>Y) across all evaluated algos via `compute_pairwise_poi()`. Use `--pairwise-only` to recompute just the pairwise metrics.
3. `notebook/report.py` — Marimo reactive notebook (20 cells) with dual-view layout: static matplotlib (publication, LaTeX, 600 DPI) on the left, interactive altair on the right. Uses `ALGO_COLORS` and `ALGO_LABELS` dicts. Cells include: learning curves (per-env + combined), score distributions, per-seed heatmap, per-seed box/swarm with anomaly detection, sample efficiency (per-env + combined), cross-env bar chart, performance profile, optimality gap, probability of improvement heatmap, wall-clock timing analysis, summary rankings, and metric tables.

**Key module — `env_config.py`** (single source of truth):
- `ENV_REGISTRY` dict maps env IDs (e.g. `"CartPole-v1"`) to `EnvSpec` dataclasses containing max_return, timesteps, eval_freq, action_space_type, etc.
- `ENV_ORDER` controls evaluation ordering (3 discrete environments: CartPole-v1, LunarLander-v3, Acrobot-v1).
- `ALGO_ENV_COMPAT` maps algo names to supported action space types (all 5 algos support discrete only).
- `check_algo_env_compat()` validates algo-env pairs, `get_compatible_envs()` lists compatible envs.
- `MASTER_SEED` + `generate_seeds()` derives uncorrelated seeds via NumPy `SeedSequence`.
- `normalize_score()` implements min-max normalization (random→0, max→1).

**Adding a new algorithm:**
1. Add the SB3 class to `ALGO_MAP` in `train.py`
2. Add the SB3 class to `ALGO_CLASSES` in `evaluate.py`
3. Add action space compatibility to `ALGO_ENV_COMPAT` in `env_config.py`
4. Add color and label entries to `ALGO_COLORS` and `ALGO_LABELS` in `notebook/report.py`
5. Results go under `results/<algo>/`, metrics under `results/metrics/<algo>/`

**Adding a new environment:**
1. Add an `EnvSpec` entry to `ENV_REGISTRY` in `env_config.py` with `action_space_type`
2. Add the env ID to `ENV_ORDER`
3. Ensure the gymnasium extra is installed (e.g. `gymnasium[box2d]`)

**Results directory layout:**
```
results/
  <algo>/<env-slug>/config.json       # training config + timing
  <algo>/<env-slug>/seed_<N>/         # model checkpoints + eval logs
  metrics/<algo>/evaluation_results.json  # all computed metrics
  metrics/<algo>/score_matrix.npy         # (n_seeds, n_envs) normalized scores
  metrics/<algo>/raw_score_matrix.npy    # (n_seeds, n_envs) unnormalized scores
  metrics/<algo>/learning_curves/<slug>.npz
  metrics/<algo>/sample_efficiency/<slug>.npz
  metrics/pairwise_poi.json              # P(X>Y) for all algorithm pairs
  figures/*.png                       # generated plots
```

## Key Constraints

- **Python 3.10 only** (`requires-python = "==3.10.*"`)
- All 5 algorithms (a2c, dqn, ppo, qrdqn, rppo) work with **discrete** action spaces only
- **RecurrentPPO** (rppo) uses `MlpLstmPolicy`; all others use `MlpPolicy`
- Training determinism is CPU-only; CPU vs GPU results differ (PyTorch limitation)
- `tests/conftest.py` adds project root to `sys.path` so `import env_config` works from tests
- Marimo notebooks are `.py` files, not `.ipynb` — edit with `uv run marimo edit`
- The report notebook uses LaTeX rendering (`text.usetex: True`) — requires a TeX installation
- The report notebook uses altair for interactive charts alongside matplotlib static charts

## Rules

- **No Co-Authored-By**: Do not include Claude attribution in commit messages.
- **Keep CLAUDE.md current** when changing structure or adding new commands.
