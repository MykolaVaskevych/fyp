#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Clearing results..."

# Trained models, checkpoints, logs, configs (all algorithms)
for algo in a2c dqn ppo qrdqn rppo; do
    rm -rf "results/$algo"
done

# Evaluation metrics (all algorithms)
for algo in a2c dqn ppo qrdqn rppo; do
    rm -rf "results/metrics/$algo"
done

# Legacy flat metrics (pre-multi-algo)
rm -rf results/metrics/learning_curves
rm -rf results/metrics/sample_efficiency
rm -f results/metrics/*.json results/metrics/*.npy results/metrics/*.npz

# Figures
rm -f results/figures/*.png results/figures/*.pdf

echo "Done."
