#!/usr/bin/env bash
# Continue training from where we left off.
# Skips seeds that already have a final model; cleans up and reruns partial seeds.
set -euo pipefail

ALGOS=("dqn" "ppo")
ENVS=("Pong" "Breakout")

for algo in "${ALGOS[@]}"; do
    echo "=== Training ${algo^^} on Atari (resume mode) ==="
    for env in "${ENVS[@]}"; do
        echo "  Starting $env ..."
        uv run python train.py --algo "$algo" --env "$env" --device cuda --resume
        echo "  Done $env"
    done

    echo "=== Evaluating $algo ==="
    uv run python evaluate.py --algo "$algo"
    echo
done

echo "=== Computing pairwise P(X>Y) ==="
uv run python evaluate.py --pairwise-only

echo "=== Generating figures ==="
uv run python generate_figures.py

echo "Done."
