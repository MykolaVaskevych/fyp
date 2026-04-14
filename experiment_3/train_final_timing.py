"""Train the final 7 missing PPO timing seeds (Experiment 3, run 3).

Missing seeds (confirmed by audit):
  PPO Pong:    indices 7-9 → seeds 812540611, 2544781736, 3667090056
  PPO Breakout: indices 6-9 → seeds 3891539213, 812540611, 2544781736, 3667090056

Output: run3_final_timing/<algo>/<env>/
  - seed_<N>/   — model checkpoints and eval logs (timing only; eval scores NOT re-evaluated)
  - config.json — all 10 seeds listed; per_seed_wall_seconds has 10 entries
                  (0.0 for already-covered indices, actual time for newly trained ones)

Usage:
    uv run python train_final_timing.py --env Pong
    uv run python train_final_timing.py --env Breakout
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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

gym.register_envs(ale_py)

from env_config import ENV_REGISTRY, EnvSpec, generate_seeds

OUTPUT_BASE = Path("run3_final_timing")

# Exactly which seed indices are missing per env (0-based into the 10-seed list)
MISSING_INDICES: dict[str, list[int]] = {
    "Pong": [7, 8, 9],
    "Breakout": [6, 7, 8, 9],
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_train_env(env_spec: EnvSpec, seed: int):
    env = make_atari_env(
        env_spec.env_id,
        n_envs=env_spec.n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
    )
    return VecFrameStack(env, n_stack=4)


def make_eval_env(env_spec: EnvSpec, seed: int):
    env = make_atari_env(env_spec.env_id, n_envs=1, seed=seed)
    return VecFrameStack(env, n_stack=4)


def train_seed(seed: int, env_spec: EnvSpec, device: str) -> tuple[float, float]:
    """Train one PPO seed. Returns (wall_seconds, cpu_seconds)."""
    results_dir = OUTPUT_BASE / "ppo" / env_spec.slug
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

    model = PPO(
        "CnnPolicy",
        train_env,
        seed=seed,
        device=device,
        verbose=0,
    )

    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    model.learn(total_timesteps=env_spec.total_timesteps, callback=eval_callback)
    elapsed_wall = time.perf_counter() - t0_wall
    elapsed_cpu = time.process_time() - t0_cpu

    model.save(str(seed_dir / "ppo_final"))
    train_env.close()
    eval_env.close()

    print(
        f"  seed {seed} done: {elapsed_wall / 3600:.2f}h wall, {elapsed_cpu / 3600:.2f}h cpu"
    )
    return elapsed_wall, elapsed_cpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        required=True,
        choices=list(MISSING_INDICES.keys()),
        help="environment to train (Pong or Breakout)",
    )
    parser.add_argument("--device", default="cuda", help="torch device")
    args = parser.parse_args()

    env_spec = ENV_REGISTRY[args.env]
    all_seeds = generate_seeds(10)
    missing_indices = MISSING_INDICES[args.env]
    seeds_to_train = [all_seeds[i] for i in missing_indices]

    # Safety: refuse to overwrite any already-completed seed in run3
    results_dir = OUTPUT_BASE / "ppo" / env_spec.slug
    for seed in seeds_to_train:
        existing = results_dir / f"seed_{seed}" / "ppo_final.zip"
        if existing.exists():
            print(f"ERROR: seed {seed} already has a final model in {results_dir}.")
            print("Remove it manually if you want to retrain.")
            raise SystemExit(1)

    print(f"PPO {args.env}: training {len(seeds_to_train)} missing seeds")
    print(f"  Seeds:   {seeds_to_train}")
    print(f"  Indices: {missing_indices}")
    print(f"  Output:  {results_dir}")
    print(f"  Steps:   {env_spec.total_timesteps:,} per seed")
    print()

    # Build 10-element timing arrays (0.0 for already-covered indices)
    wall_arr = [0.0] * 10
    cpu_arr = [0.0] * 10

    t_total = time.perf_counter()
    for idx, seed in zip(missing_indices, seeds_to_train):
        print(
            f"[{seeds_to_train.index(seed) + 1}/{len(seeds_to_train)}] seed {seed} (index {idx})"
        )
        w, c = train_seed(seed, env_spec, args.device)
        wall_arr[idx] = round(w, 2)
        cpu_arr[idx] = round(c, 2)
    total_wall = time.perf_counter() - t_total

    # Save config
    import stable_baselines3
    from env_config import MASTER_SEED

    config = {
        "algorithm": "ppo",
        "environment": env_spec.env_id,
        "policy": "CnnPolicy",
        "total_timesteps": env_spec.total_timesteps,
        "max_return": env_spec.max_return,
        "random_baseline": env_spec.random_baseline,
        "n_envs": env_spec.n_envs,
        "n_stack": 4,
        "eval_freq": env_spec.eval_freq,
        "n_eval_episodes": env_spec.n_eval_episodes,
        "master_seed": MASTER_SEED,
        "seeds": all_seeds,
        "trained_indices": missing_indices,
        "device": args.device,
        "sb3_version": stable_baselines3.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "timing": {
            "per_seed_wall_seconds": wall_arr,
            "per_seed_cpu_seconds": cpu_arr,
            "total_wall_seconds": round(total_wall, 2),
            "note": "Only indices listed in trained_indices have nonzero timing. "
            "Combine with run1_original and run2_timing_retrain for complete coverage.",
        },
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Total: {total_wall / 3600:.2f}h")
    print(f"Config saved to: {results_dir / 'config.json'}")
    print()
    print("Timing written at indices:")
    for idx, seed in zip(missing_indices, seeds_to_train):
        print(f"  [{idx}] seed {seed}: {wall_arr[idx] / 3600:.2f}h")


if __name__ == "__main__":
    main()
