"""Finish the interrupted Breakout timing run.

seed_3667090056 (index 9) was interrupted at 59%. The other 3 Breakout seeds
(indices 6-8) completed; their timing is estimated from file modification times.

This script:
1. Trains seed_3667090056 from scratch and records its timing.
2. Saves run3_final_timing/ppo/breakout/config.json with all 4 timing values.

Usage:
    uv run python finish_breakout_timing.py
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from datetime import datetime, timezone
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

from env_config import ENV_REGISTRY, MASTER_SEED, generate_seeds

OUTPUT_BASE = Path("run3_final_timing")
ENV_NAME = "Breakout"
MISSING_INDEX = 9  # seed_3667090056 — the only one that needs training
COMPLETED_INDICES = [6, 7, 8]  # estimated from file timestamps


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def estimate_completed_timing(
    all_seeds: list[int], pong_config_path: Path
) -> list[float]:
    """Estimate wall-clock seconds for the 3 completed seeds from file mtimes.

    Breakout start = script_start + pong_total.
    Script start = Apr 14 21:00:30 IST (UTC+1) = Apr 14 20:00:30 UTC.
    Consecutive seed durations = diff between consecutive ppo_final.zip mtimes.
    """
    env_spec = ENV_REGISTRY[ENV_NAME]
    results_dir = OUTPUT_BASE / "ppo" / env_spec.slug

    # Load Pong config to get pong_total
    pong_cfg = json.loads(pong_config_path.read_text())
    pong_total = sum(pong_cfg["timing"]["per_seed_wall_seconds"])

    # Script start (UTC)
    script_start_utc = datetime(2026, 4, 14, 20, 0, 30, tzinfo=timezone.utc)
    script_start_ts = script_start_utc.timestamp()
    breakout_start_ts = script_start_ts + pong_total

    seed_dirs = ["seed_3891539213", "seed_812540611", "seed_2544781736"]
    mtimes = [os.path.getmtime(results_dir / s / "ppo_final.zip") for s in seed_dirs]

    durations = [
        mtimes[0] - breakout_start_ts,  # index 6
        mtimes[1] - mtimes[0],  # index 7
        mtimes[2] - mtimes[1],  # index 8
    ]

    print("Estimated timing for completed seeds (from file timestamps):")
    for idx, s, d in zip(COMPLETED_INDICES, seed_dirs, durations):
        print(f"  [{idx}] {s}: {d:.0f}s = {d / 3600:.2f}h")

    return durations


def train_seed(seed: int, env_spec, device: str) -> tuple[float, float]:
    results_dir = OUTPUT_BASE / "ppo" / env_spec.slug
    seed_dir = results_dir / f"seed_{seed}"

    # Clean up the partial run (was at 59%, no ppo_final.zip)
    if seed_dir.exists():
        shutil.rmtree(seed_dir)
        print(f"  Cleaned up partial seed_{seed} directory.")

    log_dir = seed_dir / "logs"
    best_model_dir = seed_dir / "best_model"
    log_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)

    train_env = make_atari_env(
        env_spec.env_id, n_envs=env_spec.n_envs, seed=seed, vec_env_cls=SubprocVecEnv
    )
    train_env = VecFrameStack(train_env, n_stack=4)
    eval_env = make_atari_env(env_spec.env_id, n_envs=1, seed=seed + 10_000)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(log_dir),
        eval_freq=max(env_spec.eval_freq // env_spec.n_envs, 1),
        n_eval_episodes=env_spec.n_eval_episodes,
        deterministic=True,
        verbose=0,
    )

    model = PPO("CnnPolicy", train_env, seed=seed, device=device, verbose=0)

    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    model.learn(total_timesteps=env_spec.total_timesteps, callback=eval_cb)
    elapsed_wall = time.perf_counter() - t0_wall
    elapsed_cpu = time.process_time() - t0_cpu

    model.save(str(seed_dir / "ppo_final"))
    train_env.close()
    eval_env.close()

    print(f"  seed {seed} done: {elapsed_wall / 3600:.2f}h wall")
    return elapsed_wall, elapsed_cpu


def main() -> None:
    env_spec = ENV_REGISTRY[ENV_NAME]
    all_seeds = generate_seeds(10)
    results_dir = OUTPUT_BASE / "ppo" / env_spec.slug

    pong_config = OUTPUT_BASE / "ppo" / "pong" / "config.json"
    if not pong_config.exists():
        raise SystemExit("ERROR: run3_final_timing/ppo/pong/config.json not found.")

    # Verify the 3 completed seeds are there
    for idx in COMPLETED_INDICES:
        seed = all_seeds[idx]
        final = results_dir / f"seed_{seed}" / "ppo_final.zip"
        if not final.exists():
            raise SystemExit(f"ERROR: Expected completed model at {final} — not found.")

    # Verify seed_3667090056 did NOT complete
    seed_last = all_seeds[MISSING_INDEX]
    final_last = results_dir / f"seed_{seed_last}" / "ppo_final.zip"
    if final_last.exists():
        raise SystemExit(
            f"ERROR: seed_{seed_last} already has ppo_final.zip — nothing to do."
        )

    estimated_durations = estimate_completed_timing(all_seeds, pong_config)

    print()
    print(f"Training seed_{seed_last} (index {MISSING_INDEX}) from scratch (~3.7h)...")
    t_total = time.perf_counter()
    measured_wall, measured_cpu = train_seed(seed_last, env_spec, device="cuda")
    total_wall = time.perf_counter() - t_total

    # Build 10-element arrays (0.0 for seeds not in this run)
    wall_arr = [0.0] * 10
    cpu_arr = [0.0] * 10
    for i, d in zip(COMPLETED_INDICES, estimated_durations):
        wall_arr[i] = round(d, 2)
    wall_arr[MISSING_INDEX] = round(measured_wall, 2)
    cpu_arr[MISSING_INDEX] = round(measured_cpu, 2)

    # Save config
    import stable_baselines3

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
        "trained_indices": COMPLETED_INDICES + [MISSING_INDEX],
        "device": "cuda",
        "sb3_version": stable_baselines3.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "timing": {
            "per_seed_wall_seconds": wall_arr,
            "per_seed_cpu_seconds": cpu_arr,
            "total_wall_seconds": round(total_wall, 2),
            "note": (
                "Indices 6-8 timing estimated from ppo_final.zip file modification times "
                "(seed interrupted at 59% on first run). Index 9 timing measured directly."
            ),
        },
    }
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Config saved to {results_dir / 'config.json'}")
    print("Timing summary:")
    for i in COMPLETED_INDICES + [MISSING_INDEX]:
        src = "estimated" if i in COMPLETED_INDICES else "measured"
        print(f"  [{i}] seed {all_seeds[i]}: {wall_arr[i] / 3600:.2f}h ({src})")


if __name__ == "__main__":
    main()
