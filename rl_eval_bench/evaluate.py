"""Evaluate trained A2C models and compute rliable metrics (multi-environment).

Usage:
    uv run python evaluate.py                     # all envs in registry
    uv run python evaluate.py --envs CartPole-v1  # single env only
    uv run python evaluate.py --episodes 100      # more eval episodes
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from rliable import library as rly
from rliable import metrics
from stable_baselines3 import A2C

from env_config import ENV_ORDER, ENV_REGISTRY, EnvSpec, normalize_score, generate_seeds

ALGO_CLASSES: dict[str, type] = {"a2c": A2C}

RESULTS_DIR = Path("results")
METRICS_DIR = RESULTS_DIR / "metrics"


def _scalar(x) -> float:
    """Extract a Python float from a numpy scalar or 0-d/1-element array."""
    return float(np.asarray(x).flat[0])


def load_config(algo: str, env_spec: EnvSpec) -> dict:
    config_path = RESULTS_DIR / algo / env_spec.slug / "config.json"
    with open(config_path) as f:
        return json.load(f)


def measure_random_baseline(
    env_spec: EnvSpec, seeds: list[int], n_episodes: int
) -> tuple[float, dict[int, float]]:
    """Run a uniform-random agent across seeds and return grand mean + per-seed breakdown."""
    per_seed: dict[int, float] = {}

    for seed in seeds:
        env = gym.make(env_spec.env_id)
        env.action_space.seed(seed)
        returns = []
        for ep in range(n_episodes):
            if ep == 0:
                obs, _ = env.reset(seed=seed)
            else:
                obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            returns.append(total_reward)
        env.close()
        seed_mean = float(np.mean(returns))
        per_seed[seed] = seed_mean
        print(f"  seed {seed:>10d}: random baseline = {seed_mean:.2f} ({n_episodes} episodes)")

    grand_mean = float(np.mean(list(per_seed.values())))
    print(f"  Grand mean random baseline: {grand_mean:.2f}")
    return grand_mean, per_seed


def fresh_evaluate(algo: str, env_spec: EnvSpec, n_episodes: int) -> np.ndarray:
    """Load each seed's final model, run n_episodes. Returns (n_seeds,) score array."""
    config = load_config(algo, env_spec)
    seeds = config["seeds"]
    algo_cls = ALGO_CLASSES[algo]
    scores = []

    for seed in seeds:
        model_path = RESULTS_DIR / algo / env_spec.slug / f"seed_{seed}" / f"{algo}_final.zip"
        model = algo_cls.load(str(model_path), device="cpu")

        env = gym.make(env_spec.env_id)
        episode_rewards = []
        for ep in range(n_episodes):
            if ep == 0:
                obs, _ = env.reset(seed=seed)
            else:
                obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            episode_rewards.append(total_reward)
        env.close()

        mean_reward = np.mean(episode_rewards)
        scores.append(mean_reward)
        print(f"  seed {seed:>2d}: mean={mean_reward:.1f} over {n_episodes} episodes")

    return np.array(scores)


def load_learning_curves(algo: str, env_spec: EnvSpec) -> tuple[np.ndarray, np.ndarray]:
    """Load evaluations.npz from all seeds. Returns (timesteps, reward_matrix).

    reward_matrix shape: (n_seeds, n_checkpoints) — mean reward per checkpoint.
    """
    config = load_config(algo, env_spec)
    seeds = config["seeds"]

    all_means = []
    timesteps = None

    for seed in seeds:
        npz_path = (
            RESULTS_DIR / algo / env_spec.slug / f"seed_{seed}" / "logs" / "evaluations.npz"
        )
        data = np.load(npz_path)
        ts = data["timesteps"]
        results = data["results"]  # (n_checkpoints, n_eval_episodes)

        if timesteps is None:
            timesteps = ts
        else:
            min_len = min(len(timesteps), len(ts))
            timesteps = timesteps[:min_len]
            results = results[:min_len]

        mean_per_checkpoint = results.mean(axis=1)
        all_means.append(mean_per_checkpoint[: len(timesteps)])

    reward_matrix = np.array(all_means)  # (n_seeds, n_checkpoints)
    return timesteps, reward_matrix


def compute_per_environment_metrics(
    env_spec: EnvSpec,
    random_baseline: float,
    scores_1d: np.ndarray,
    timesteps: np.ndarray,
    reward_matrix: np.ndarray,
) -> dict:
    """Compute per-environment metrics using env-specific max_return."""
    results = {}

    # Learning curves summary stats
    mean_curve = reward_matrix.mean(axis=0)
    std_curve = reward_matrix.std(axis=0)
    median_curve = np.median(reward_matrix, axis=0)
    p25_curve = np.percentile(reward_matrix, 25, axis=0)
    p75_curve = np.percentile(reward_matrix, 75, axis=0)

    results["learning_curves"] = {
        "mean": mean_curve.tolist(),
        "std": std_curve.tolist(),
        "median": median_curve.tolist(),
        "p25": p25_curve.tolist(),
        "p75": p75_curve.tolist(),
        "timesteps": timesteps.tolist(),
    }

    # Final performance with 95% stratified bootstrap CI
    score_matrix = scores_1d.reshape(-1, 1)
    score_dict = {"a2c": normalize_score(score_matrix, random_baseline, env_spec.max_return)}

    iqm_fn = lambda x: metrics.aggregate_iqm(x)
    mean_fn = lambda x: metrics.aggregate_mean(x)
    median_fn = lambda x: metrics.aggregate_median(x)

    for name, fn in [("iqm", iqm_fn), ("mean", mean_fn), ("median", median_fn)]:
        point, ci = rly.get_interval_estimates(score_dict, fn, reps=50_000)
        results[f"final_{name}"] = {
            "point": _scalar(point["a2c"]),
            "ci_low": _scalar(ci["a2c"][0]),
            "ci_high": _scalar(ci["a2c"][1]),
        }

    # Sample efficiency AUC (normalized, using IQM per checkpoint)
    n_checkpoints = reward_matrix.shape[1]
    iqm_curve = np.zeros(n_checkpoints)
    n_seeds = reward_matrix.shape[0]
    q1 = n_seeds // 4
    q3 = n_seeds - q1
    for i in range(n_checkpoints):
        col = np.sort(reward_matrix[:, i])
        iqm_curve[i] = np.mean(col[q1:q3])
    iqm_curve_norm = normalize_score(iqm_curve, random_baseline, env_spec.max_return)
    auc = float(np.trapezoid(iqm_curve_norm, timesteps) / timesteps[-1])
    results["sample_efficiency_auc"] = auc

    # Reliability across seeds
    final_scores = scores_1d.flatten()
    q25, q75 = np.percentile(final_scores, [25, 75])
    iqr = float(q75 - q25)

    n_seeds = len(final_scores)
    k = max(1, int(np.ceil(0.1 * n_seeds)))
    sorted_scores = np.sort(final_scores)
    cvar_01 = float(sorted_scores[:k].mean())

    results["reliability"] = {
        "iqr": iqr,
        "cvar_01": cvar_01,
        "min_score": float(sorted_scores[0]),
        "max_score": float(sorted_scores[-1]),
    }

    return results


def compute_cross_environment_metrics(score_matrix: np.ndarray) -> dict:
    """Compute cross-environment metrics. score_matrix: (n_seeds, M) already normalized."""
    results = {}
    score_dict = {"a2c": score_matrix}

    # IQM + CI
    point, ci = rly.get_interval_estimates(
        score_dict, lambda x: metrics.aggregate_iqm(x), reps=50_000
    )
    results["iqm"] = {
        "point": _scalar(point["a2c"]),
        "ci_low": _scalar(ci["a2c"][0]),
        "ci_high": _scalar(ci["a2c"][1]),
    }

    # Aggregate mean/median + CI
    for name, fn in [
        ("mean", lambda x: metrics.aggregate_mean(x)),
        ("median", lambda x: metrics.aggregate_median(x)),
    ]:
        point, ci = rly.get_interval_estimates(score_dict, fn, reps=50_000)
        results[name] = {
            "point": _scalar(point["a2c"]),
            "ci_low": _scalar(ci["a2c"][0]),
            "ci_high": _scalar(ci["a2c"][1]),
        }

    # Performance profile
    tau_list = np.linspace(0.0, 1.0, 51)
    perf_profile, perf_ci = rly.create_performance_profile(
        score_dict, tau_list, reps=2000
    )
    results["performance_profile"] = {
        "tau": tau_list.tolist(),
        "values": perf_profile["a2c"].tolist(),
        "ci_low": perf_ci["a2c"][0].tolist(),
        "ci_high": perf_ci["a2c"][1].tolist(),
    }

    # Optimality gap
    point, ci = rly.get_interval_estimates(
        score_dict,
        lambda x: metrics.aggregate_optimality_gap(x, gamma=1.0),
        reps=50_000,
    )
    results["optimality_gap"] = {
        "point": _scalar(point["a2c"]),
        "ci_low": _scalar(ci["a2c"][0]),
        "ci_high": _scalar(ci["a2c"][1]),
    }

    return results


def compute_sample_efficiency_curves(
    env_spec: EnvSpec,
    random_baseline: float,
    timesteps: np.ndarray,
    reward_matrix: np.ndarray,
) -> dict:
    """IQM at each checkpoint via rliable bootstrap."""
    n_checkpoints = len(timesteps)
    iqm_values = np.zeros(n_checkpoints)
    ci_low = np.zeros(n_checkpoints)
    ci_high = np.zeros(n_checkpoints)

    for i in range(n_checkpoints):
        scores_at_t = normalize_score(
            reward_matrix[:, i : i + 1], random_baseline, env_spec.max_return
        )
        score_dict = {"a2c": scores_at_t}
        point, ci = rly.get_interval_estimates(
            score_dict, lambda x: metrics.aggregate_iqm(x), reps=2000
        )
        iqm_values[i] = _scalar(point["a2c"])
        ci_low[i] = _scalar(ci["a2c"][0])
        ci_high[i] = _scalar(ci["a2c"][1])

    return {
        "timesteps": timesteps,
        "iqm": iqm_values,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained A2C models")
    parser.add_argument("--algo", type=str, default="a2c", help="algorithm name")
    parser.add_argument(
        "--envs",
        type=str,
        nargs="*",
        default=None,
        help="environments to evaluate (default: all in registry)",
    )
    parser.add_argument(
        "--episodes", type=int, default=50, help="eval episodes per seed"
    )
    args = parser.parse_args()

    env_ids = args.envs if args.envs else ENV_ORDER
    env_specs = [ENV_REGISTRY[eid] for eid in env_ids]

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    lc_dir = METRICS_DIR / "learning_curves"
    se_dir = METRICS_DIR / "sample_efficiency"
    lc_dir.mkdir(parents=True, exist_ok=True)
    se_dir.mkdir(parents=True, exist_ok=True)

    per_env_results: dict[str, dict] = {}
    per_env_scores: dict[str, np.ndarray] = {}  # raw scores (n_seeds,)
    random_baselines: dict[str, float] = {}
    random_baseline_details: dict[str, dict[int, float]] = {}
    env_timing: dict[str, float] = {}

    seeds = generate_seeds()

    # Measure random baselines
    print("Measuring random baselines...")
    for env_spec in env_specs:
        grand_mean, per_seed = measure_random_baseline(env_spec, seeds, args.episodes)
        random_baselines[env_spec.slug] = grand_mean
        random_baseline_details[env_spec.slug] = {str(k): v for k, v in per_seed.items()}

    t_eval_total = time.perf_counter()

    for env_spec in env_specs:
        print(f"\n{'='*60}")
        print(f"Evaluating {args.algo} on {env_spec.env_id} ({args.episodes} episodes/seed)...")
        print(f"{'='*60}")

        t_env = time.perf_counter()

        # Fresh evaluation
        scores_1d = fresh_evaluate(args.algo, env_spec, args.episodes)
        per_env_scores[env_spec.slug] = scores_1d

        # Learning curves
        print(f"Loading learning curves for {env_spec.slug}...")
        timesteps, reward_matrix = load_learning_curves(args.algo, env_spec)
        np.savez(
            lc_dir / f"{env_spec.slug}.npz",
            timesteps=timesteps,
            reward_matrix=reward_matrix,
            mean=reward_matrix.mean(axis=0),
            std=reward_matrix.std(axis=0),
            median=np.median(reward_matrix, axis=0),
            p25=np.percentile(reward_matrix, 25, axis=0),
            p75=np.percentile(reward_matrix, 75, axis=0),
        )

        # Per-environment metrics
        _baseline = random_baselines[env_spec.slug]
        print(f"Computing per-environment metrics for {env_spec.slug}...")
        per_env = compute_per_environment_metrics(
            env_spec, _baseline, scores_1d, timesteps, reward_matrix
        )

        # Sample efficiency curves
        print(f"Computing sample efficiency curves for {env_spec.slug}...")
        se_curves = compute_sample_efficiency_curves(
            env_spec, _baseline, timesteps, reward_matrix
        )
        np.savez(
            se_dir / f"{env_spec.slug}.npz",
            timesteps=se_curves["timesteps"],
            iqm=se_curves["iqm"],
            ci_low=se_curves["ci_low"],
            ci_high=se_curves["ci_high"],
        )

        env_elapsed = time.perf_counter() - t_env
        env_timing[env_spec.slug] = round(env_elapsed, 2)
        per_env_results[env_spec.slug] = per_env

    total_eval_elapsed = time.perf_counter() - t_eval_total

    # Build normalized score matrix (n_seeds, M)
    normalized_cols = []
    for env_spec in env_specs:
        _baseline = random_baselines[env_spec.slug]
        normalized_cols.append(
            normalize_score(per_env_scores[env_spec.slug], _baseline, env_spec.max_return)
        )
    score_matrix = np.column_stack(normalized_cols)  # (n_seeds, M)
    np.save(METRICS_DIR / "score_matrix.npy", score_matrix)

    # Cross-environment metrics
    print(f"\n{'='*60}")
    print(f"Computing cross-environment metrics on {score_matrix.shape} matrix...")
    print(f"{'='*60}")
    cross_env = compute_cross_environment_metrics(score_matrix)

    # Load training timing from each env's config.json
    training_timing: dict[str, dict] = {}
    for env_spec in env_specs:
        try:
            config = load_config(args.algo, env_spec)
            training_timing[env_spec.slug] = config.get("timing", {})
        except FileNotFoundError:
            training_timing[env_spec.slug] = {}

    # Determine actual seed count from first env's config
    actual_n_seeds = len(load_config(args.algo, env_specs[0])["seeds"])

    # Save combined results
    all_metrics = {
        "algorithm": args.algo,
        "n_seeds": actual_n_seeds,
        "n_eval_episodes": args.episodes,
        "environments": [es.slug for es in env_specs],
        "random_baselines": random_baselines,
        "random_baseline_details": random_baseline_details,
        "per_environment": per_env_results,
        "cross_environment": cross_env,
        "timing": {
            "evaluation": {
                "per_env_seconds": env_timing,
                "total_seconds": round(total_eval_elapsed, 2),
            },
            "training": training_timing,
        },
    }
    with open(METRICS_DIR / "evaluation_results.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nMetrics saved to {METRICS_DIR}/")
    print(f"  evaluation_results.json — all metrics")
    print(f"  score_matrix.npy — {score_matrix.shape}")
    for env_spec in env_specs:
        print(f"  learning_curves/{env_spec.slug}.npz")
        print(f"  sample_efficiency/{env_spec.slug}.npz")


if __name__ == "__main__":
    main()
