"""Environment registry — single source of truth for per-env constants."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import SeedSequence


@dataclass(frozen=True)
class EnvSpec:
    env_id: str  # gymnasium env name, e.g. "CartPole-v1"
    slug: str  # filesystem-safe key, e.g. "cartpole-v1"
    max_return: float  # normalization constant for rliable
    total_timesteps: int  # default training budget
    eval_freq: int  # per-env steps between evaluations
    n_envs: int  # DummyVecEnv parallelism
    n_eval_episodes: int  # episodes per eval checkpoint
    action_space_type: str  # "discrete" or "continuous"


ENV_REGISTRY: dict[str, EnvSpec] = {
    "CartPole-v1": EnvSpec(
        env_id="CartPole-v1",
        slug="cartpole-v1",
        max_return=500.0,
        total_timesteps=200_000,
        eval_freq=1250,
        n_envs=4,
        n_eval_episodes=10,
        action_space_type="discrete",
    ),
    "LunarLander-v3": EnvSpec(
        env_id="LunarLander-v3",
        slug="lunarlander-v3",
        max_return=300.0,
        total_timesteps=500_000,
        eval_freq=2500,
        n_envs=4,
        n_eval_episodes=10,
        action_space_type="discrete",
    ),
    "Acrobot-v1": EnvSpec(
        env_id="Acrobot-v1",
        slug="acrobot-v1",
        max_return=-100.0,
        total_timesteps=200_000,
        eval_freq=1250,
        n_envs=4,
        n_eval_episodes=10,
        action_space_type="discrete",
    ),
}

ENV_ORDER: list[str] = [
    "CartPole-v1",
    "LunarLander-v3",
    "Acrobot-v1",
]

# Algo → supported action space types
ALGO_ENV_COMPAT: dict[str, list[str]] = {
    "a2c": ["discrete"],
    "dqn": ["discrete"],
    "ppo": ["discrete"],
    "qrdqn": ["discrete"],
    "rppo": ["discrete"],
}


def check_algo_env_compat(algo: str, env_id: str) -> None:
    """Raise ValueError if algo is incompatible with env's action space."""
    if algo not in ALGO_ENV_COMPAT:
        raise ValueError(f"Unknown algorithm: {algo!r}")
    spec = ENV_REGISTRY[env_id]
    supported = ALGO_ENV_COMPAT[algo]
    if spec.action_space_type not in supported:
        raise ValueError(
            f"{algo} does not support {spec.action_space_type} action spaces "
            f"(env={env_id}). Supported: {supported}"
        )


def get_compatible_envs(algo: str) -> list[str]:
    """Return list of env IDs compatible with the given algorithm."""
    if algo not in ALGO_ENV_COMPAT:
        raise ValueError(f"Unknown algorithm: {algo!r}")
    supported = ALGO_ENV_COMPAT[algo]
    return [eid for eid, spec in ENV_REGISTRY.items() if spec.action_space_type in supported]
N_SEEDS: int = 15
MASTER_SEED: int = 20260212


def normalize_score(
    score: np.ndarray | float, random_baseline: float, max_return: float
) -> np.ndarray | float:
    """Normalize score: random → 0, max → 1 (Agarwal et al., 2021)."""
    denom = max_return - random_baseline
    if denom == 0:
        return np.zeros_like(score) if isinstance(score, np.ndarray) else 0.0
    return (score - random_baseline) / denom


def generate_seeds(n: int = N_SEEDS) -> list[int]:
    """Derive n uncorrelated seeds from MASTER_SEED via NumPy SeedSequence.

    Avoids linear-dependent seeds (0, 1, 2, ...) which can produce correlated
    PRNG streams.  See Matsumoto et al., "Common defects in initialization of
    pseudorandom number generators".
    """
    ss = SeedSequence(MASTER_SEED)
    children = ss.spawn(n)
    return [int(child.generate_state(1)[0]) for child in children]
