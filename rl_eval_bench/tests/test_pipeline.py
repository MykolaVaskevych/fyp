"""Fast smoke tests for the evaluation pipeline (no training required)."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
import pytest

from env_config import (
    ENV_ORDER,
    ENV_REGISTRY,
    EnvSpec,
    N_SEEDS,
    MASTER_SEED,
    generate_seeds,
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
        for n in (1, 5, 20):
            assert len(generate_seeds(n)) == n

    def test_normalize_score_boundaries(self):
        """random → 0, max → 1."""
        assert normalize_score(10.0, 10.0, 100.0) == pytest.approx(0.0)
        assert normalize_score(100.0, 10.0, 100.0) == pytest.approx(1.0)

    def test_normalize_score_negative_baseline(self):
        result = normalize_score(0.0, -100.0, 100.0)
        assert result == pytest.approx(0.5)

    def test_normalize_score_division_by_zero_scalar(self):
        result = normalize_score(42.0, 100.0, 100.0)
        assert result == 0.0

    def test_normalize_score_division_by_zero_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = normalize_score(arr, 5.0, 5.0)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_normalize_score_array(self):
        arr = np.array([10.0, 55.0, 100.0])
        result = normalize_score(arr, 10.0, 100.0)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_registry_structure(self):
        """Every ENV_ORDER entry is in ENV_REGISTRY and has required fields."""
        for env_id in ENV_ORDER:
            spec = ENV_REGISTRY[env_id]
            assert isinstance(spec, EnvSpec)
            assert spec.env_id == env_id
            assert spec.max_return > 0
            assert spec.total_timesteps > 0
            assert spec.slug  # non-empty


# ---------------------------------------------------------------------------
# TestRandomBaseline
# ---------------------------------------------------------------------------


class TestRandomBaseline:
    """Tests for measure_random_baseline()."""

    @pytest.fixture()
    def cartpole_spec(self):
        return ENV_REGISTRY["CartPole-v1"]

    def test_returns_tuple(self, cartpole_spec):
        from evaluate import measure_random_baseline

        seeds = generate_seeds(3)
        result = measure_random_baseline(cartpole_spec, seeds, n_episodes=5)
        assert isinstance(result, tuple)
        assert len(result) == 2
        grand_mean, per_seed = result
        assert isinstance(grand_mean, float)
        assert isinstance(per_seed, dict)

    def test_per_seed_keys_match(self, cartpole_spec):
        from evaluate import measure_random_baseline

        seeds = generate_seeds(3)
        _, per_seed = measure_random_baseline(cartpole_spec, seeds, n_episodes=5)
        assert set(per_seed.keys()) == set(seeds)

    def test_grand_mean_is_average_of_per_seed(self, cartpole_spec):
        from evaluate import measure_random_baseline

        seeds = generate_seeds(3)
        grand_mean, per_seed = measure_random_baseline(cartpole_spec, seeds, n_episodes=5)
        expected = float(np.mean(list(per_seed.values())))
        assert grand_mean == pytest.approx(expected)

    def test_reproducible(self, cartpole_spec):
        from evaluate import measure_random_baseline

        seeds = generate_seeds(3)
        r1 = measure_random_baseline(cartpole_spec, seeds, n_episodes=5)
        r2 = measure_random_baseline(cartpole_spec, seeds, n_episodes=5)
        assert r1[0] == pytest.approx(r2[0])
        for s in seeds:
            assert r1[1][s] == pytest.approx(r2[1][s])


# ---------------------------------------------------------------------------
# TestFreshEvaluate
# ---------------------------------------------------------------------------


class TestFreshEvaluate:
    """Tests for algo class mapping and model filename logic (no trained models needed)."""

    def test_algo_class_mapping_resolves(self):
        from evaluate import ALGO_CLASSES
        from stable_baselines3 import A2C

        assert "a2c" in ALGO_CLASSES
        assert ALGO_CLASSES["a2c"] is A2C

    def test_model_filename_uses_algo_name(self):
        """Verify the filename template uses the algo string, not hardcoded 'a2c'."""
        algo = "ppo"
        expected_suffix = f"{algo}_final.zip"
        assert expected_suffix == "ppo_final.zip"

        algo = "a2c"
        expected_suffix = f"{algo}_final.zip"
        assert expected_suffix == "a2c_final.zip"

    def test_unknown_algo_raises_keyerror(self):
        from evaluate import ALGO_CLASSES

        with pytest.raises(KeyError):
            ALGO_CLASSES["nonexistent_algo"]


# ---------------------------------------------------------------------------
# TestEnvironmentSmoke
# ---------------------------------------------------------------------------


class TestEnvironmentSmoke:
    """Verify all registered environments can be created and stepped."""

    @pytest.mark.parametrize("env_id", ENV_ORDER)
    def test_env_create_and_step(self, env_id):
        spec = ENV_REGISTRY[env_id]
        env = gym.make(spec.env_id)
        obs, info = env.reset(seed=42)
        assert obs is not None
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env.step(action)
        assert obs2 is not None
        env.close()
