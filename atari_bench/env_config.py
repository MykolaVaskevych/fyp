"""Environment registry — single source of truth for Atari experiment (Experiment 2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import SeedSequence


@dataclass(frozen=True)
class EnvSpec:
    env_id: str  # ALE env name, e.g. "PongNoFrameskip-v4"
    slug: str  # filesystem-safe key, e.g. "pong"
    max_return: float  # normalization ceiling for rliable
    random_baseline: float  # expected return of uniform-random agent
    total_timesteps: int  # training budget
    eval_freq: int  # per-env steps between evaluations
    n_envs: int  # parallel Atari envs for training
    n_eval_episodes: int  # episodes per eval checkpoint
    action_space_type: str  # always "discrete" for Atari


ENV_REGISTRY: dict[str, EnvSpec] = {
    "Pong": EnvSpec(
        env_id="PongNoFrameskip-v4",
        slug="pong",
        max_return=21.0,
        random_baseline=-20.5,
        total_timesteps=5_000_000,
        eval_freq=25_000,
        n_envs=8,
        n_eval_episodes=10,
        action_space_type="discrete",
    ),
    "Breakout": EnvSpec(
        env_id="BreakoutNoFrameskip-v4",
        slug="breakout",
        max_return=400.0,
        random_baseline=1.5,
        total_timesteps=5_000_000,
        eval_freq=25_000,
        n_envs=8,
        n_eval_episodes=10,
        action_space_type="discrete",
    ),
}

ENV_ORDER: list[str] = ["Pong", "Breakout"]

# Algo -> supported action space types
ALGO_ENV_COMPAT: dict[str, list[str]] = {
    "dqn": ["discrete"],
    "ppo": ["discrete"],
}

N_SEEDS: int = 10
MASTER_SEED: int = 20260215  # different from Experiment 1's 20260212


def check_algo_env_compat(algo: str, env_name: str) -> None:
    """Raise ValueError if algo is incompatible with env's action space."""
    if algo not in ALGO_ENV_COMPAT:
        raise ValueError(f"Unknown algorithm: {algo!r}")
    spec = ENV_REGISTRY[env_name]
    supported = ALGO_ENV_COMPAT[algo]
    if spec.action_space_type not in supported:
        raise ValueError(
            f"{algo} does not support {spec.action_space_type} action spaces "
            f"(env={env_name}). Supported: {supported}"
        )


def get_compatible_envs(algo: str) -> list[str]:
    """Return list of env names compatible with the given algorithm."""
    if algo not in ALGO_ENV_COMPAT:
        raise ValueError(f"Unknown algorithm: {algo!r}")
    supported = ALGO_ENV_COMPAT[algo]
    return [name for name, spec in ENV_REGISTRY.items() if spec.action_space_type in supported]


def normalize_score(
    score: np.ndarray | float, random_baseline: float, max_return: float
) -> np.ndarray | float:
    """Normalize score: random -> 0, max -> 1 (Agarwal et al., 2021)."""
    denom = max_return - random_baseline
    if denom == 0:
        return np.zeros_like(score) if isinstance(score, np.ndarray) else 0.0
    return (score - random_baseline) / denom


def generate_seeds(n: int = N_SEEDS) -> list[int]:
    """Derive n uncorrelated seeds from MASTER_SEED via NumPy SeedSequence."""
    ss = SeedSequence(MASTER_SEED)
    children = ss.spawn(n)
    return [int(child.generate_state(1)[0]) for child in children]
