#!/usr/bin/env bash
# Training progress monitor for Experiment 3
# Usage: bash monitor.sh [interval_seconds]  (default: 3600 = 1 hour)
set -euo pipefail

INTERVAL="${1:-3600}"
RESULTS_DIR="$(cd "$(dirname "$0")" && pwd)/results"
ALGOS=("dqn" "ppo")
ENVS=("pong" "breakout" "seaquest")
SEEDS_TOTAL=10
TOTAL_TIMESTEPS=20000000
TOTAL_JOBS=60  # 2 algos * 3 envs * 10 seeds

# Colors
R='\033[0m'    # reset
G='\033[32m'   # green (done)
Y='\033[33m'   # yellow (active)
D='\033[90m'   # dim (pending)
B='\033[1m'    # bold
C='\033[36m'   # cyan

bar() {
    # bar <filled> <total> <width>
    local filled=$1 total=$2 width=${3:-20}
    local pct=0
    if [ "$total" -gt 0 ]; then
        pct=$((filled * 100 / total))
    fi
    local done_w=$((filled * width / total))
    local left_w=$((width - done_w))
    local color="$D"
    if [ "$pct" -ge 100 ]; then color="$G"
    elif [ "$pct" -gt 0 ]; then color="$Y"
    fi
    printf "${color}["
    printf '%0.s#' $(seq 1 "$done_w" 2>/dev/null) || true
    printf '%0.s-' $(seq 1 "$left_w" 2>/dev/null) || true
    printf "]${R} %3d%%" "$pct"
}

fmt_time() {
    # fmt_time <seconds>
    local s="${1%.*}"
    [ -z "$s" ] || [ "$s" -le 0 ] 2>/dev/null && { echo "--:--"; return; }
    local h=$((s / 3600)) m=$(( (s % 3600) / 60 ))
    printf "%dh%02dm" "$h" "$m"
}

render() {
    clear
    printf "${B}${C}  Experiment 3 — Training Monitor${R}\n"
    printf "${D}  %s${R}\n\n" "$(date '+%Y-%m-%d %H:%M:%S')"

    local jobs_done=0
    local total_wall=0
    local active_algo="" active_env="" active_seed="" active_pct=0

    for algo in "${ALGOS[@]}"; do
        local algo_upper="${algo^^}"
        printf "${B}  %s${R}\n" "$algo_upper"

        for env in "${ENVS[@]}"; do
            local dir="$RESULTS_DIR/$algo/$env"
            local completed=0
            local env_wall=0

            # Count completed seeds
            if [ -d "$dir" ]; then
                for sd in "$dir"/seed_*/; do
                    [ -d "$sd" ] || continue
                    if [ -f "${sd}${algo}_final.zip" ]; then
                        completed=$((completed + 1))
                    else
                        # Active seed — get progress from evaluations.npz
                        local npz="${sd}logs/evaluations.npz"
                        if [ -f "$npz" ]; then
                            local ts
                            ts=$(python3 -c "import numpy as np; print(int(np.load('$npz')['timesteps'][-1]))" 2>/dev/null || echo 0)
                            active_pct=$((ts * 100 / TOTAL_TIMESTEPS))
                            active_algo="$algo_upper"
                            active_env="$env"
                            active_seed="$(basename "$sd")"
                        fi
                    fi
                done
            fi

            # Read avg seed time from config if available
            local avg_seed_s=0
            local config="$dir/config.json"
            if [ -f "$config" ]; then
                avg_seed_s=$(python3 -c "
import json
with open('$config') as f:
    c = json.load(f)
w = c['timing']['per_seed_wall_seconds']
print(int(sum(w)/len(w)) if w else 0)
" 2>/dev/null || echo 0)
                env_wall=$(python3 -c "
import json
with open('$config') as f:
    c = json.load(f)
print(int(c['timing']['total_wall_seconds']))
" 2>/dev/null || echo 0)
            fi

            jobs_done=$((jobs_done + completed))
            total_wall=$((total_wall + env_wall))

            # Print env row
            printf "    %-10s " "$env"
            bar "$completed" "$SEEDS_TOTAL" 20
            printf "  %2d/%d" "$completed" "$SEEDS_TOTAL"
            if [ "$avg_seed_s" -gt 0 ]; then
                printf "  ~%s/seed" "$(fmt_time "$avg_seed_s")"
            fi
            printf "\n"
        done
        printf "\n"
    done

    # Active seed info
    if [ -n "$active_algo" ]; then
        printf "${Y}  Active: %s %s %s — %d%%${R}\n" "$active_algo" "$active_env" "$active_seed" "$active_pct"
    fi

    # Overall progress
    local overall_pct=$((jobs_done * 100 / TOTAL_JOBS))
    printf "\n${B}  Overall${R}  "
    bar "$jobs_done" "$TOTAL_JOBS" 30
    printf "  %d/%d seeds\n" "$jobs_done" "$TOTAL_JOBS"

    # ETA estimate
    if [ "$jobs_done" -gt 0 ] && [ "$total_wall" -gt 0 ] && [ "$jobs_done" -lt "$TOTAL_JOBS" ]; then
        local avg_job=$((total_wall / jobs_done))
        local remaining=$(( (TOTAL_JOBS - jobs_done) * avg_job ))
        printf "${D}  Elapsed: %s  |  ETA: ~%s${R}\n" "$(fmt_time "$total_wall")" "$(fmt_time "$remaining")"
    elif [ "$jobs_done" -ge "$TOTAL_JOBS" ]; then
        printf "${G}  Done! Total: %s${R}\n" "$(fmt_time "$total_wall")"
    fi

    printf "\n${D}  Refreshing every %ds. Ctrl+C to exit.${R}\n" "$INTERVAL"
}

# Main loop
while true; do
    render
    sleep "$INTERVAL"
done
