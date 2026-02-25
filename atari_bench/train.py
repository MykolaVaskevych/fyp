"""Atari training — DQN vs PPO with CnnPolicy on GPU (Experiment 2).

Usage:
    uv run python train.py --algo dqn --env Pong
    uv run python train.py --algo ppo --env Breakout
    uv run python train.py --algo dqn --env Pong --device cuda --seeds 10
    uv run python train.py --algo dqn --env Pong --resume   # skip completed seeds
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

gym.register_envs(ale_py)

from env_config import ENV_REGISTRY, EnvSpec, N_SEEDS, generate_seeds

ALGO_MAP: dict[str, type] = {"dqn": DQN, "ppo": PPO}


def seed_everything(seed: int) -> None:
    """Seed all RNGs before each training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_train_env(env_spec: EnvSpec, seed: int):
    """Create vectorized Atari training environment with frame stacking.

    Uses SubprocVecEnv to distribute environments across CPU cores.
    """
    env = make_atari_env(
        env_spec.env_id,
        n_envs=env_spec.n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
    )
    env = VecFrameStack(env, n_stack=4)
    return env


def make_eval_env(env_spec: EnvSpec, seed: int):
    """Create single Atari eval environment with frame stacking."""
    env = make_atari_env(env_spec.env_id, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    return env


def train_seed(
    seed: int, env_spec: EnvSpec, total_timesteps: int, device: str, algo: str = "dqn"
) -> tuple[Path, float]:
    """Train one seed with the given algorithm. Returns (seed_dir, elapsed_seconds)."""
    results_dir = Path("results") / algo / env_spec.slug
    seed_dir = results_dir / f"seed_{seed}"
    log_dir = seed_dir / "logs"
    best_model_dir = seed_dir / "best_model"
    log_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)

    train_env = make_train_env(env_spec, seed)
    eval_env = make_eval_env(env_spec, seed + 10_000)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(log_dir),
        eval_freq=max(env_spec.eval_freq // env_spec.n_envs, 1),
        n_eval_episodes=env_spec.n_eval_episodes,
        deterministic=True,
        verbose=0,
    )

    # DQN: reduce buffer_size from default 1M to 100K for Atari
    # (1M × 84×84 × uint8 = ~56GB RAM; 100K = ~5.6GB, sufficient for 5M steps)
    algo_kwargs = {}
    if algo == "dqn":
        algo_kwargs["buffer_size"] = 100_000

    model = ALGO_MAP[algo](
        "CnnPolicy",
        train_env,
        seed=seed,
        device=device,
        verbose=0,
        **algo_kwargs,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    elapsed = time.perf_counter() - t0

    model.save(str(seed_dir / f"{algo}_final"))

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
    algo: str = "dqn",
) -> None:
    """Save training config for reproducibility."""
    import stable_baselines3

    from env_config import MASTER_SEED

    results_dir = Path("results") / algo / env_spec.slug
    config = {
        "algorithm": algo,
        "environment": env_spec.env_id,
        "policy": "CnnPolicy",
        "total_timesteps": total_timesteps,
        "max_return": env_spec.max_return,
        "random_baseline": env_spec.random_baseline,
        "n_envs": env_spec.n_envs,
        "n_stack": 4,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN/PPO on Atari environments")
    parser.add_argument(
        "--algo",
        type=str,
        default="dqn",
        choices=list(ALGO_MAP.keys()),
        help="algorithm to train (default: dqn)",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=list(ENV_REGISTRY.keys()),
        help="environment to train on (Pong or Breakout)",
    )
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help="number of seeds")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="timesteps per seed (overrides env default)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="torch device (default: cuda)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip seeds that already have a completed final model",
    )
    args = parser.parse_args()

    algo = args.algo
    env_spec = ENV_REGISTRY[args.env]
    total_timesteps = (
        args.timesteps if args.timesteps is not None else env_spec.total_timesteps
    )

    seeds = generate_seeds(args.seeds)
    results_dir = Path("results") / algo / env_spec.slug

    # Determine which seeds to run
    if args.resume:
        seeds_to_run = []
        for seed in seeds:
            seed_dir = results_dir / f"seed_{seed}"
            final_model = seed_dir / f"{algo}_final.zip"
            if final_model.exists():
                print(f"  seed {seed}: SKIP (final model exists)")
            else:
                # Clean up partial results before rerunning
                if seed_dir.exists():
                    import shutil

                    shutil.rmtree(seed_dir)
                    print(f"  seed {seed}: cleaned up partial results, will retrain")
                seeds_to_run.append(seed)
        if not seeds_to_run:
            print(f"All seeds already completed for {algo}/{env_spec.slug}.")
            save_config(
                env_spec, seeds, total_timesteps, args.device, [], 0.0, algo=algo
            )
            return
    else:
        seeds_to_run = seeds

    print(
        f"Training {algo.upper()} on {env_spec.env_id}: {len(seeds_to_run)}/{len(seeds)} seeds, "
        f"{total_timesteps} steps each, device={args.device}"
    )

    per_seed_seconds: list[float] = []
    t_total = time.perf_counter()
    for seed in seeds_to_run:
        _, elapsed = train_seed(seed, env_spec, total_timesteps, args.device, algo=algo)
        per_seed_seconds.append(elapsed)
    total_elapsed = time.perf_counter() - t_total

    save_config(
        env_spec,
        seeds,
        total_timesteps,
        args.device,
        per_seed_seconds,
        total_elapsed,
        algo=algo,
    )

    print(f"\nAll done in {total_elapsed:.1f}s. Results in {results_dir}/")


if __name__ == "__main__":
    main()
