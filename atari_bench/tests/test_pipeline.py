"""Smoke tests for the Atari evaluation pipeline (Experiment 2)."""

from __future__ import annotations

import numpy as np
import pytest

from env_config import (
    ALGO_ENV_COMPAT,
    ENV_ORDER,
    ENV_REGISTRY,
    EnvSpec,
    MASTER_SEED,
    N_SEEDS,
    check_algo_env_compat,
    generate_seeds,
    get_compatible_envs,
    normalize_score,
)


# ---------------------------------------------------------------------------
# TestEnvConfig
# ---------------------------------------------------------------------------


class TestEnvConfig:
    """Tests for env_config module: seed generation, normalize_score, registry."""

    def test_generate_seeds_deterministic(self):
        """Same call twice produces identical seeds."""
        assert generate_seeds(5) == generate_seeds(5)

    def test_generate_seeds_unique(self):
        seeds = generate_seeds(N_SEEDS)
        assert len(seeds) == len(set(seeds)), "seeds must be unique"

    def test_generate_seeds_count(self):
        for n in (1, 5, 10):
            assert len(generate_seeds(n)) == n

    def test_n_seeds_is_10(self):
        assert N_SEEDS == 10

    def test_master_seed_differs_from_exp1(self):
        assert MASTER_SEED == 20260215
        assert MASTER_SEED != 20260212  # Experiment 1's seed

    def test_normalize_score_boundaries(self):
        """random -> 0, max -> 1."""
        assert normalize_score(10.0, 10.0, 100.0) == pytest.approx(0.0)
        assert normalize_score(100.0, 10.0, 100.0) == pytest.approx(1.0)

    def test_normalize_score_negative_baseline(self):
        # Pong: random baseline is -20.5, max is 21
        result = normalize_score(-20.5, -20.5, 21.0)
        assert result == pytest.approx(0.0)
        result = normalize_score(21.0, -20.5, 21.0)
        assert result == pytest.approx(1.0)

    def test_normalize_score_division_by_zero_scalar(self):
        result = normalize_score(42.0, 100.0, 100.0)
        assert result == 0.0

    def test_normalize_score_array(self):
        arr = np.array([10.0, 55.0, 100.0])
        result = normalize_score(arr, 10.0, 100.0)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_registry_structure(self):
        """Every ENV_ORDER entry is in ENV_REGISTRY and has required fields."""
        for env_name in ENV_ORDER:
            spec = ENV_REGISTRY[env_name]
            assert isinstance(spec, EnvSpec)
            assert spec.total_timesteps == 5_000_000
            assert spec.slug
            assert spec.action_space_type == "discrete"
            assert spec.n_envs == 16

    def test_only_two_envs(self):
        assert len(ENV_ORDER) == 2
        assert set(ENV_ORDER) == {"Pong", "Breakout"}


# ---------------------------------------------------------------------------
# TestAlgoEnvCompat
# ---------------------------------------------------------------------------


class TestAlgoEnvCompat:
    """Tests for algorithm-environment compatibility checks."""

    @pytest.mark.parametrize("algo", ["dqn", "ppo"])
    def test_algos_accept_discrete(self, algo):
        for env_name in ENV_ORDER:
            check_algo_env_compat(algo, env_name)

    def test_unknown_algo_raises(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            check_algo_env_compat("nonexistent", "Pong")

    @pytest.mark.parametrize("algo", ["dqn", "ppo"])
    def test_get_compatible_envs_all(self, algo):
        envs = get_compatible_envs(algo)
        assert set(envs) == set(ENV_ORDER)

    def test_only_two_algos(self):
        assert set(ALGO_ENV_COMPAT.keys()) == {"dqn", "ppo"}


# ---------------------------------------------------------------------------
# TestAlgoClasses
# ---------------------------------------------------------------------------


class TestAlgoClasses:
    """Tests for algo class mapping."""

    def test_algo_class_mapping(self):
        from train import ALGO_MAP

        from stable_baselines3 import DQN, PPO

        assert ALGO_MAP["dqn"] is DQN
        assert ALGO_MAP["ppo"] is PPO
        assert len(ALGO_MAP) == 2

    def test_evaluate_algo_classes(self):
        from evaluate import ALGO_CLASSES

        from stable_baselines3 import DQN, PPO

        assert ALGO_CLASSES["dqn"] is DQN
        assert ALGO_CLASSES["ppo"] is PPO
        assert len(ALGO_CLASSES) == 2


# ---------------------------------------------------------------------------
# TestAtariEnvSmoke
# ---------------------------------------------------------------------------


class TestAtariEnvSmoke:
    """Verify Atari environments can be created with proper wrappers."""

    @pytest.mark.parametrize("env_name", ENV_ORDER)
    def test_atari_env_create(self, env_name):
        import ale_py
        import gymnasium as gym
        from stable_baselines3.common.env_util import make_atari_env
        from stable_baselines3.common.vec_env import VecFrameStack

        gym.register_envs(ale_py)

        spec = ENV_REGISTRY[env_name]
        env = make_atari_env(spec.env_id, n_envs=1, seed=42)
        env = VecFrameStack(env, n_stack=4)
        obs = env.reset()
        assert obs.shape == (1, 84, 84, 4)  # (n_envs, H, W, n_stack) — channels-last
        action = [env.action_space.sample()]
        obs2, reward, done, info = env.step(action)
        assert obs2.shape == (1, 84, 84, 4)
        env.close()
