"""A2C training with deterministic seeding — supports multiple environments.

Usage:
    uv run python train.py --env CartPole-v1              # 15 seeds, 200K steps
    uv run python train.py --env LunarLander-v3           # 15 seeds, 500K steps
    uv run python train.py --env LunarLander-v3 --timesteps 750000
    uv run python train.py --env CartPole-v1 --seeds 3 --device cpu
    uv run python train.py --env CartPole-v1 --verify     # determinism check
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from env_config import ENV_REGISTRY, EnvSpec, N_SEEDS, generate_seeds


def seed_everything(seed: int) -> None:
    """Seed all RNGs before each training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_seed(
    seed: int, env_spec: EnvSpec, total_timesteps: int, device: str
) -> tuple[Path, float]:
    """Train A2C for one seed. Returns (seed_dir, elapsed_seconds)."""
    results_dir = Path("results") / "a2c" / env_spec.slug
    seed_dir = results_dir / f"seed_{seed}"
    log_dir = seed_dir / "logs"
    best_model_dir = seed_dir / "best_model"
    log_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)

    train_env = make_vec_env(env_spec.env_id, n_envs=env_spec.n_envs, seed=seed)
    train_env.reset()

    eval_env = make_vec_env(env_spec.env_id, n_envs=1, seed=seed + 10_000)
    eval_env.reset()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(log_dir),
        eval_freq=env_spec.eval_freq,
        n_eval_episodes=env_spec.n_eval_episodes,
        deterministic=True,
        verbose=0,
    )

    model = A2C("MlpPolicy", train_env, seed=seed, device=device, verbose=0)

    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    elapsed = time.perf_counter() - t0

    model.save(str(seed_dir / "a2c_final"))

    train_env.close()
    eval_env.close()

    print(f"  seed {seed} done in {elapsed:.1f}s")
    return seed_dir, elapsed


def save_config(
    env_spec: EnvSpec,
    seeds: list[int],
    total_timesteps: int,
    device: str,
    per_seed_seconds: list[float],
    total_seconds: float,
) -> None:
    """Save training config for reproducibility."""
    import stable_baselines3

    from env_config import MASTER_SEED

    results_dir = Path("results") / "a2c" / env_spec.slug
    config = {
        "algorithm": "A2C",
        "environment": env_spec.env_id,
        "policy": "MlpPolicy",
        "total_timesteps": total_timesteps,
        "max_return": env_spec.max_return,
        "n_envs": env_spec.n_envs,
        "eval_freq": env_spec.eval_freq,
        "n_eval_episodes": env_spec.n_eval_episodes,
        "master_seed": MASTER_SEED,
        "seeds": seeds,
        "device": device,
        "sb3_version": stable_baselines3.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "timing": {
            "per_seed_seconds": [round(t, 2) for t in per_seed_seconds],
            "total_seconds": round(total_seconds, 2),
        },
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def verify_determinism(env_spec: EnvSpec, device: str) -> bool:
    """Train a seed twice and assert identical evaluation results."""
    results_dir = Path("results") / "a2c" / env_spec.slug
    test_seed = generate_seeds(1)[0]
    seed_dir_name = f"seed_{test_seed}"
    print(f"Verifying determinism on {env_spec.env_id}: training seed {test_seed} twice...")

    train_seed(test_seed, env_spec, total_timesteps=50_000, device=device)
    npz1 = np.load(results_dir / seed_dir_name / "logs" / "evaluations.npz")
    ts1, res1 = npz1["timesteps"].copy(), npz1["results"].copy()

    shutil.rmtree(results_dir / seed_dir_name)
    train_seed(test_seed, env_spec, total_timesteps=50_000, device=device)
    npz2 = np.load(results_dir / seed_dir_name / "logs" / "evaluations.npz")
    ts2, res2 = npz2["timesteps"].copy(), npz2["results"].copy()

    shutil.rmtree(results_dir / seed_dir_name)

    ts_match = np.array_equal(ts1, ts2)
    res_match = np.array_equal(res1, res2)

    if ts_match and res_match:
        print("PASS: both runs produced identical evaluations.npz")
        return True
    else:
        print("FAIL: runs differ!")
        if not ts_match:
            print(f"  timesteps differ: {ts1[:5]}... vs {ts2[:5]}...")
        if not res_match:
            diff_idx = np.where(res1 != res2)
            print(f"  results differ at {len(diff_idx[0])} positions")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train A2C on a gymnasium environment")
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=list(ENV_REGISTRY.keys()),
        help="environment to train on",
    )
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help="number of seeds")
    parser.add_argument(
        "--timesteps", type=int, default=None, help="timesteps per seed (overrides env default)"
    )
    parser.add_argument("--device", type=str, default="cpu", help="torch device")
    parser.add_argument(
        "--verify", action="store_true", help="train seed 0 twice to check determinism"
    )
    args = parser.parse_args()

    env_spec = ENV_REGISTRY[args.env]
    total_timesteps = args.timesteps if args.timesteps is not None else env_spec.total_timesteps

    if args.verify:
        ok = verify_determinism(env_spec, args.device)
        sys.exit(0 if ok else 1)

    seeds = generate_seeds(args.seeds)
    print(
        f"Training A2C on {env_spec.env_id}: {len(seeds)} seeds, {total_timesteps} steps each"
    )

    per_seed_seconds: list[float] = []
    t_total = time.perf_counter()
    for seed in seeds:
        _, elapsed = train_seed(seed, env_spec, total_timesteps, args.device)
        per_seed_seconds.append(elapsed)
    total_elapsed = time.perf_counter() - t_total

    save_config(env_spec, seeds, total_timesteps, args.device, per_seed_seconds, total_elapsed)

    results_dir = Path("results") / "a2c" / env_spec.slug
    print(f"\nAll done in {total_elapsed:.1f}s. Results in {results_dir}/")


if __name__ == "__main__":
    main()
