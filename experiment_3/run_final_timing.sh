#!/usr/bin/env bash
# Run the final 7 missing PPO timing seeds.
#
# PPO Pong:     3 seeds (indices 7-9) — ~11h
# PPO Breakout: 4 seeds (indices 6-9) — ~13h
# Total:        ~24h on RTX 4090
#
# Output goes to run3_final_timing/ppo/{pong,breakout}/
# Run from the experiment_3/ directory.

set -euo pipefail
cd "$(dirname "$0")"

echo "=========================================="
echo "  Final timing recovery — 7 PPO seeds"
echo "  Started: $(date)"
echo "=========================================="
echo ""

# --- Pre-flight: confirm nothing already exists in run3 ---
if [ -d "run3_final_timing" ]; then
    echo "WARNING: run3_final_timing/ already exists."
    echo "Contents:"
    find run3_final_timing -name "config.json" | sort
    echo ""
    read -r -p "Continue anyway? (yes/no): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Aborted."
        exit 0
    fi
fi

echo "--- Step 1/2: PPO Pong (3 seeds, ~11h) ---"
echo "Started: $(date)"
uv run python train_final_timing.py --env Pong --device cuda
echo "Pong done: $(date)"
echo ""

echo "--- Step 2/2: PPO Breakout (4 seeds, ~13h) ---"
echo "Started: $(date)"
uv run python train_final_timing.py --env Breakout --device cuda
echo "Breakout done: $(date)"
echo ""

echo "=========================================="
echo "  ALL DONE: $(date)"
echo "=========================================="
echo ""
echo "Results in run3_final_timing/ppo/pong/config.json"
echo "              run3_final_timing/ppo/breakout/config.json"
echo ""
echo "Next step: regenerate notebook figures"
echo "  cd ../analysis_results_experiment_3"
echo "  uv run marimo export html notebook.py -o /dev/null"
