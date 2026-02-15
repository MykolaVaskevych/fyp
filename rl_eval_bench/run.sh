#!/usr/bin/env bash
set -euo pipefail

ALGOS=("a2c" "dqn" "ppo" "qrdqn" "rppo")
ENVS=("CartPole-v1" "LunarLander-v3" "Acrobot-v1")

for algo in "${ALGOS[@]}"; do
    echo "=== Training ${algo^^} on compatible environments ==="
    for env in "${ENVS[@]}"; do
        uv run python train.py --algo "$algo" --env "$env" 2>/dev/null \
            && echo "  Trained $env" \
            || echo "  Skipped $env (incompatible with $algo)"
    done

    echo "=== Evaluating $algo ==="
    uv run python evaluate.py --algo "$algo"
    echo
done

echo "=== Launching marimo notebook ==="
exec uv run marimo edit notebook/report.py
