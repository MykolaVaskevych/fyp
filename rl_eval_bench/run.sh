#!/usr/bin/env bash
set -euo pipefail

echo "=== Training A2C on CartPole-v1 ==="
uv run python train.py --env CartPole-v1

echo "=== Training A2C on LunarLander-v3 ==="
uv run python train.py --env LunarLander-v3

echo "=== Evaluating all environments ==="
uv run python evaluate.py

echo "=== Launching marimo notebook ==="
exec uv run marimo edit notebook/report.py
