#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Clearing results..."

# Trained models, checkpoints, logs
rm -rf results/a2c/*/seed_*
rm -f results/a2c/*/config.json

# Evaluation metrics
rm -rf results/metrics/learning_curves
rm -rf results/metrics/sample_efficiency
rm -f results/metrics/*.json results/metrics/*.npy results/metrics/*.npz

# Figures
rm -f results/figures/*.png results/figures/*.pdf

echo "Done."
