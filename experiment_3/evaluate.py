"""Evaluate trained Atari models and compute rliable metrics (Experiment 2).

Usage:
    uv run python evaluate.py --algo dqn                    # all envs
    uv run python evaluate.py --algo ppo --envs Pong        # single env
    uv run python evaluate.py --algo dqn --episodes 100     # more eval episodes
    uv run python evaluate.py --pairwise-only               # recompute P(X>Y) only
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
from rliable import library as rly
from rliable import metrics
from scipy.stats import trim_mean
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

from env_config import ENV_ORDER, ENV_REGISTRY, EnvSpec, generate_seeds, normalize_score

ALGO_CLASSES: dict[str, type] = {
    "dqn": DQN,
    "ppo": PPO,
}

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
    """Run a uniform-random agent across seeds and return grand mean + per-seed breakdown.

    Uses raw gym.make() (no Atari wrappers) to get true unclipped episode returns.
    """
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
        print(
            f"  seed {seed:>10d}: random baseline = {seed_mean:.2f} ({n_episodes} episodes)"
        )

    grand_mean = float(np.mean(list(per_seed.values())))
    print(f"  Grand mean random baseline: {grand_mean:.2f}")
    return grand_mean, per_seed


def fresh_evaluate(algo: str, env_spec: EnvSpec, n_episodes: int) -> np.ndarray:
    """Load each seed's final model, run n_episodes. Returns (n_seeds,) score array.

    Uses SB3's evaluate_policy which reads true episode returns from the Monitor
    wrapper, bypassing ClipRewardEnv and EpisodicLifeEnv.
    """
    config = load_config(algo, env_spec)
    seeds = config["seeds"]
    algo_cls = ALGO_CLASSES[algo]
    scores = []

    for seed in seeds:
        model_path = (
            RESULTS_DIR / algo / env_spec.slug / f"seed_{seed}" / f"{algo}_final.zip"
        )
        model = algo_cls.load(str(model_path), device="cpu")

        env = make_atari_env(env_spec.env_id, n_envs=1, seed=seed)
        env = VecFrameStack(env, n_stack=4)

        episode_rewards, episode_lengths = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_episodes,
            deterministic=True,
            return_episode_rewards=True,
        )
        env.close()

        mean_reward = float(np.mean(episode_rewards))
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
            RESULTS_DIR
            / algo
            / env_spec.slug
            / f"seed_{seed}"
            / "logs"
            / "evaluations.npz"
        )
        data = np.load(npz_path)
        ts = data["timesteps"]
        results = data["results"]  # (n_checkpoints, n_eval_episodes)

        if timesteps is None:
            timesteps = ts
        else:
            if len(ts) != len(timesteps):
                warnings.warn(
                    f"Seed {seed} has {len(ts)} checkpoints vs {len(timesteps)} "
                    f"expected — truncating to min length",
                    stacklevel=2,
                )
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
    algo: str = "dqn",
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
    score_dict = {
        algo: normalize_score(score_matrix, random_baseline, env_spec.max_return)
    }

    iqm_fn = lambda x: metrics.aggregate_iqm(x)
    mean_fn = lambda x: metrics.aggregate_mean(x)
    median_fn = lambda x: metrics.aggregate_median(x)

    for name, fn in [("iqm", iqm_fn), ("mean", mean_fn), ("median", median_fn)]:
        point, ci = rly.get_interval_estimates(score_dict, fn, reps=50_000)
        results[f"final_{name}"] = {
            "point": _scalar(point[algo]),
            "ci_low": _scalar(ci[algo][0]),
            "ci_high": _scalar(ci[algo][1]),
        }

    # Sample efficiency AUC (normalized, using IQM per checkpoint)
    n_checkpoints = reward_matrix.shape[1]
    iqm_curve = np.zeros(n_checkpoints)
    for i in range(n_checkpoints):
        iqm_curve[i] = trim_mean(reward_matrix[:, i], proportiontocut=0.25)
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


def compute_cross_environment_metrics(
    score_matrix: np.ndarray, algo: str = "dqn"
) -> dict:
    """Compute cross-environment metrics. score_matrix: (n_seeds, M) already normalized."""
    results = {}
    score_dict = {algo: score_matrix}

    # IQM + CI
    point, ci = rly.get_interval_estimates(
        score_dict, lambda x: metrics.aggregate_iqm(x), reps=50_000
    )
    results["iqm"] = {
        "point": _scalar(point[algo]),
        "ci_low": _scalar(ci[algo][0]),
        "ci_high": _scalar(ci[algo][1]),
    }

    # Aggregate mean/median + CI
    for name, fn in [
        ("mean", lambda x: metrics.aggregate_mean(x)),
        ("median", lambda x: metrics.aggregate_median(x)),
    ]:
        point, ci = rly.get_interval_estimates(score_dict, fn, reps=50_000)
        results[name] = {
            "point": _scalar(point[algo]),
            "ci_low": _scalar(ci[algo][0]),
            "ci_high": _scalar(ci[algo][1]),
        }

    # Performance profile
    tau_list = np.linspace(0.0, 1.0, 51)
    perf_profile, perf_ci = rly.create_performance_profile(
        score_dict, tau_list, reps=2000
    )
    results["performance_profile"] = {
        "tau": tau_list.tolist(),
        "values": perf_profile[algo].tolist(),
        "ci_low": perf_ci[algo][0].tolist(),
        "ci_high": perf_ci[algo][1].tolist(),
    }

    # Optimality gap
    point, ci = rly.get_interval_estimates(
        score_dict,
        lambda x: metrics.aggregate_optimality_gap(x, gamma=1.0),
        reps=50_000,
    )
    results["optimality_gap"] = {
        "point": _scalar(point[algo]),
        "ci_low": _scalar(ci[algo][0]),
        "ci_high": _scalar(ci[algo][1]),
    }

    return results


def compute_sample_efficiency_curves(
    env_spec: EnvSpec,
    random_baseline: float,
    timesteps: np.ndarray,
    reward_matrix: np.ndarray,
    algo: str = "dqn",
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
        score_dict = {algo: scores_at_t}
        point, ci = rly.get_interval_estimates(
            score_dict, lambda x: metrics.aggregate_iqm(x), reps=2000
        )
        iqm_values[i] = _scalar(point[algo])
        ci_low[i] = _scalar(ci[algo][0])
        ci_high[i] = _scalar(ci[algo][1])

    return {
        "timesteps": timesteps,
        "iqm": iqm_values,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def compute_pairwise_poi(metrics_dir: Path) -> dict:
    """Compute P(X>Y) for all algorithm pairs using rliable."""
    algo_scores = {}
    for d in sorted(metrics_dir.iterdir()):
        sm_path = d / "score_matrix.npy"
        if d.is_dir() and sm_path.exists():
            algo_scores[d.name] = np.load(sm_path)

    if len(algo_scores) < 2:
        return {}

    algos = sorted(algo_scores.keys())
    poi_results = {}
    for i, a in enumerate(algos):
        for j, b in enumerate(algos):
            if i >= j:
                continue
            poi = metrics.probability_of_improvement(algo_scores[a], algo_scores[b])
            poi_results[f"{a}_vs_{b}"] = float(poi)
    return poi_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained Atari models")
    parser.add_argument(
        "--algo",
        type=str,
        default="dqn",
        choices=list(ALGO_CLASSES.keys()),
        help="algorithm name (default: dqn)",
    )
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
    parser.add_argument(
        "--pairwise-only",
        action="store_true",
        help="only compute pairwise P(X>Y) across all algos (skip per-algo evaluation)",
    )
    args = parser.parse_args()

    if args.pairwise_only:
        print("Computing pairwise P(X>Y) across all algorithms...")
        poi = compute_pairwise_poi(METRICS_DIR)
        if poi:
            out_path = METRICS_DIR / "pairwise_poi.json"
            with open(out_path, "w") as f:
                json.dump(poi, f, indent=2)
            print(f"Saved pairwise P(X>Y) to {out_path}")
            for pair, val in poi.items():
                print(f"  {pair}: {val:.4f}")
        else:
            print("Need >= 2 algos with score_matrix.npy to compute pairwise POI.")
        return

    algo = args.algo

    env_names = args.envs if args.envs else ENV_ORDER
    env_specs = [ENV_REGISTRY[name] for name in env_names]

    # Per-algo metrics directory
    algo_metrics_dir = METRICS_DIR / algo
    algo_metrics_dir.mkdir(parents=True, exist_ok=True)
    lc_dir = algo_metrics_dir / "learning_curves"
    se_dir = algo_metrics_dir / "sample_efficiency"
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
        random_baseline_details[env_spec.slug] = {
            str(k): v for k, v in per_seed.items()
        }

    t_eval_total = time.perf_counter()

    for env_spec in env_specs:
        print(f"\n{'=' * 60}")
        print(
            f"Evaluating {algo} on {env_spec.env_id} ({args.episodes} episodes/seed)..."
        )
        print(f"{'=' * 60}")

        t_env = time.perf_counter()

        # Fresh evaluation
        scores_1d = fresh_evaluate(algo, env_spec, args.episodes)
        per_env_scores[env_spec.slug] = scores_1d

        # Learning curves
        print(f"Loading learning curves for {env_spec.slug}...")
        timesteps, reward_matrix = load_learning_curves(algo, env_spec)
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
            env_spec, _baseline, scores_1d, timesteps, reward_matrix, algo=algo
        )

        # Sample efficiency curves
        print(f"Computing sample efficiency curves for {env_spec.slug}...")
        se_curves = compute_sample_efficiency_curves(
            env_spec, _baseline, timesteps, reward_matrix, algo=algo
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
            normalize_score(
                per_env_scores[env_spec.slug], _baseline, env_spec.max_return
            )
        )
    score_matrix = np.column_stack(normalized_cols)  # (n_seeds, M)
    np.save(algo_metrics_dir / "score_matrix.npy", score_matrix)

    # Raw (unnormalized) score matrix
    raw_score_matrix = np.column_stack(
        [per_env_scores[ENV_REGISTRY[name].slug] for name in env_names]
    )
    np.save(algo_metrics_dir / "raw_score_matrix.npy", raw_score_matrix)

    # Cross-environment metrics (only meaningful with >= 2 envs)
    cross_env: dict = {}
    if len(env_specs) >= 2:
        print(f"\n{'=' * 60}")
        print(f"Computing cross-environment metrics on {score_matrix.shape} matrix...")
        print(f"{'=' * 60}")
        cross_env = compute_cross_environment_metrics(score_matrix, algo=algo)
    else:
        print("\nSkipping cross-environment metrics (need >= 2 environments).")

    # Load training timing from each env's config.json
    training_timing: dict[str, dict] = {}
    for env_spec in env_specs:
        try:
            config = load_config(algo, env_spec)
            training_timing[env_spec.slug] = config.get("timing", {})
        except FileNotFoundError:
            training_timing[env_spec.slug] = {}

    # Determine actual seed count from first env's config
    actual_n_seeds = len(load_config(algo, env_specs[0])["seeds"])

    # Save combined results
    all_metrics = {
        "algorithm": algo,
        "n_seeds": actual_n_seeds,
        "seeds": seeds,
        "n_eval_episodes": args.episodes,
        "environments": [es.slug for es in env_specs],
        "random_baselines": random_baselines,
        "random_baseline_details": random_baseline_details,
        "per_seed_raw_scores": {
            slug: scores.tolist() for slug, scores in per_env_scores.items()
        },
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
    with open(algo_metrics_dir / "evaluation_results.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nMetrics saved to {algo_metrics_dir}/")
    print("  evaluation_results.json — all metrics")
    print(f"  score_matrix.npy — {score_matrix.shape}")
    print(f"  raw_score_matrix.npy — {raw_score_matrix.shape}")
    for env_spec in env_specs:
        print(f"  learning_curves/{env_spec.slug}.npz")
        print(f"  sample_efficiency/{env_spec.slug}.npz")

    # Compute pairwise P(X>Y) if multiple algos have been evaluated
    poi = compute_pairwise_poi(METRICS_DIR)
    if poi:
        poi_path = METRICS_DIR / "pairwise_poi.json"
        with open(poi_path, "w") as f:
            json.dump(poi, f, indent=2)
        print(f"\nPairwise P(X>Y) saved to {poi_path}")
        for pair, val in poi.items():
            print(f"  {pair}: {val:.4f}")


if __name__ == "__main__":
    main()
