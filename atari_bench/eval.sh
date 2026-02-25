#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

uv run python evaluate.py --algo dqn --episodes 50
uv run python evaluate.py --algo ppo --episodes 50
uv run python evaluate.py --pairwise-only
uv run python generate_figures.py
