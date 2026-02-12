# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Final Year Project (FYP) comparing Deep Q-Network (DQN) and Actor-Critic (A2C, PPO) reinforcement learning algorithms. The research evaluates these algorithms on Atari and classic control environments, measuring convergence speed, sample efficiency, and training stability.

## Commands

```bash
# Install dependencies
uv sync

# Run the DQN notebook (marimo reactive notebook)
uv run marimo edit DQN_ATARY/notebook/CS4287-Assignment-2-Deep-Reinforcment-Learning.py

# Lint with ruff
uv run ruff check .
uv run ruff format .
```

## Repository Structure

- **DQN_ATARY/** - Git submodule containing DQN implementation for Atari Breakout
  - `notebook/` - Marimo notebooks for DQN experiments
  - `option_pain/` - Training scripts and metrics (`train.py`, `metrics.py`, `plots.py`)
- **paper_overleaf/** - Git submodule containing LaTeX thesis source files
- **drafts/** - Experimental notebooks and draft notes

## Key Technologies

- **uv** - Python package manager (used instead of pip/poetry)
- **marimo** - Reactive Python notebooks (`.py` files, not `.ipynb`)
- **stable-baselines3** - RL algorithm implementations
- **gymnasium** - RL environments (Atari, classic control)
- **ruff** - Linting and formatting

## Submodules

Both `DQN_ATARY/` and `paper_overleaf/` are git submodules with their own repositories. After cloning, initialize them with:
```bash
git submodule update --init --recursive
```

## Dependency Management

This project uses **path dependencies** to integrate submodule code while keeping submodules standalone-capable.

### Architecture
- **Root project**: References submodules as editable path dependencies
- **Submodules**: Have their own `pyproject.toml` and `uv.lock` for standalone use
- Dependencies flow transitively: root's `uv sync` installs all submodule deps

### For root project development
```bash
uv sync                    # Install everything (submodules + their deps)
uv run python -c "import option_pain"  # Imports from DQN_ATARY work
```

### For standalone submodule work
```bash
cd DQN_ATARY
uv sync                    # Uses DQN_ATARY's own pyproject.toml
```

### Adding new submodules (A2C, PPO, etc.)
1. Add submodule: `git submodule add <url> A2C`
2. Create `A2C/pyproject.toml` with its own dependencies
3. Add to root's pyproject.toml:
```toml
[project]
dependencies = [
    "dqn-atary",
    "a2c",  # Add new package
]

[tool.uv.sources]
dqn-atary = { path = "./DQN_ATARY", editable = true }
a2c = { path = "./A2C", editable = true }  # Add path
```
4. Run `uv lock && uv sync`

## Rules

- **Keep CLAUDE.md current**: When adding new commands, changing project structure, or introducing new patterns, update this file accordingly.
- **No Co-Authored-By**: Do not include `Co-Authored-By: Claude` or any Claude attribution in commit messages.
