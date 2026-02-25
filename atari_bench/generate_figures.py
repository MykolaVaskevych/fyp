"""Generate publication-quality figures for the Atari experiment (Experiment 2).

Produces 13 PNGs:
  1.  learning_curves_pong.png        — mean±std + median±IQR (1×2 subplots)
  2.  learning_curves_breakout.png    — same for Breakout
  3.  combined_learning_curves.png    — 2-row stack (DQN vs PPO, both envs)
  4.  score_distribution.png          — 2-row boxplot+swarm
  5.  per_seed_heatmap.png            — 1×2 algo panels
  6.  per_seed_boxswarm.png           — 2-row box+swarm with anomaly marks
  7.  final_performance.png           — grouped bars (IQM/Mean/Median + 95% CI)
  8.  performance_profile.png         — DQN vs PPO curves + CI bands
  9.  optimality_gap.png              — horizontal bars + CI
  10. poi_heatmap.png                 — 2×2 annotated heatmap
  11. timing_analysis.png             — per-env training time bars
  12. sample_efficiency_pong.png      — IQM over training + CI bands
  13. sample_efficiency_breakout.png  — same for Breakout

Saves to results/figures/ and copies to paper_overleaf/5_chapter/assets/.
"""

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# -- Paths --
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
ASSETS_DIR = BASE_DIR.parent / "paper_overleaf" / "5_chapter" / "assets"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# -- Constants --
ALGO_ORDER = ["dqn", "ppo"]
ALGO_COLORS = {
    "dqn": "#ff7f0e",
    "ppo": "#2ca02c",
}
ALGO_LABELS = {
    "dqn": "DQN",
    "ppo": "PPO",
}
ENV_SLUGS = ["pong", "breakout"]
ENV_LABELS = {
    "pong": "Pong",
    "breakout": "Breakout",
}

# -- Data loading --
algo_eval = {}
algo_lc = {}
algo_se = {}
algo_score = {}
algo_raw_score = {}

for algo in ALGO_ORDER:
    algo_dir = METRICS_DIR / algo
    with open(algo_dir / "evaluation_results.json") as f:
        algo_eval[algo] = json.load(f)
    algo_score[algo] = np.load(algo_dir / "score_matrix.npy")
    algo_raw_score[algo] = np.load(algo_dir / "raw_score_matrix.npy")
    algo_lc[algo] = {}
    algo_se[algo] = {}
    for slug in ENV_SLUGS:
        algo_lc[algo][slug] = np.load(algo_dir / "learning_curves" / f"{slug}.npz")
        algo_se[algo][slug] = np.load(algo_dir / "sample_efficiency" / f"{slug}.npz")

random_baselines = algo_eval[ALGO_ORDER[0]]["random_baselines"]

# Load pairwise POI
poi_path = METRICS_DIR / "pairwise_poi.json"
pairwise_poi = {}
if poi_path.exists():
    with open(poi_path) as f:
        pairwise_poi = json.load(f)

# -- Shared style --
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def save_and_copy(fig: plt.Figure, name: str) -> None:
    src = FIGURES_DIR / name
    fig.savefig(src)
    if ASSETS_DIR.exists():
        shutil.copy2(src, ASSETS_DIR / name)
        print(f"  Saved {src} -> {ASSETS_DIR / name}")
    else:
        print(f"  Saved {src}")
    plt.close(fig)


# ── 1 & 2. Per-env learning curves (mean±std + median±IQR) ──────────────
def plot_learning_curves_single(slug: str) -> None:
    fname = f"learning_curves_{slug}.png"
    print(f"Generating {fname} ...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for algo in ALGO_ORDER:
        lc = algo_lc[algo][slug]
        ts = lc["timesteps"] / 1_000_000
        color = ALGO_COLORS[algo]
        label = ALGO_LABELS[algo]

        # Left: mean ± std
        ax = axes[0]
        ax.plot(ts, lc["mean"], color=color, label=label, linewidth=1.5)
        ax.fill_between(
            ts,
            lc["mean"] - lc["std"],
            lc["mean"] + lc["std"],
            color=color,
            alpha=0.15,
        )

        # Right: median ± IQR
        ax = axes[1]
        ax.plot(ts, lc["median"], color=color, label=label, linewidth=1.5)
        ax.fill_between(ts, lc["p25"], lc["p75"], color=color, alpha=0.15)

    rb = random_baselines.get(slug, 0)
    for i, (ax, title) in enumerate(
        zip(
            axes,
            [f"{ENV_LABELS[slug]} — Mean ± Std", f"{ENV_LABELS[slug]} — Median ± IQR"],
        )
    ):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Environment Steps (millions)")
        ax.set_ylabel("Return")
        ax.axhline(rb, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(
            ax.get_xlim()[1] * 0.98,
            rb,
            "random",
            ha="right",
            va="bottom",
            fontsize=8,
            color="gray",
        )
        ax.legend(loc="lower right", ncol=2, framealpha=0.9)

    fig.tight_layout()
    save_and_copy(fig, fname)


# ── 3. Combined learning curves (2-row stack) ───────────────────────────
def plot_combined_learning_curves() -> None:
    print("Generating combined_learning_curves.png ...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    for i, slug in enumerate(ENV_SLUGS):
        ax = axes[i]
        for algo in ALGO_ORDER:
            lc = algo_lc[algo][slug]
            ts = lc["timesteps"] / 1_000_000
            color = ALGO_COLORS[algo]
            label = ALGO_LABELS[algo]
            ax.plot(ts, lc["median"], color=color, label=label, linewidth=1.5)
            ax.fill_between(ts, lc["p25"], lc["p75"], color=color, alpha=0.15)
        ax.set_title(ENV_LABELS[slug], fontweight="bold")
        ax.set_ylabel("Return (median ± IQR)")
        if i == 1:
            ax.set_xlabel("Environment Steps (millions)")
        rb = random_baselines.get(slug, 0)
        ax.axhline(rb, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(
            ax.get_xlim()[1] * 0.98,
            rb,
            "random",
            ha="right",
            va="bottom",
            fontsize=8,
            color="gray",
        )
    axes[0].legend(loc="lower right", ncol=2, framealpha=0.9)
    fig.tight_layout()
    save_and_copy(fig, "combined_learning_curves.png")


# ── 4. Score distribution (boxplot + swarm) ──────────────────────────────
def plot_score_distribution() -> None:
    print("Generating score_distribution.png ...")
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for i, slug in enumerate(ENV_SLUGS):
        ax = axes[i]
        data = []
        labels = []
        colors = []
        for algo in ALGO_ORDER:
            col_idx = ENV_SLUGS.index(slug)
            scores = algo_score[algo][:, col_idx]
            data.append(scores)
            labels.append(ALGO_LABELS[algo])
            colors.append(ALGO_COLORS[algo])

        positions = np.arange(len(ALGO_ORDER))
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)

        rng = np.random.default_rng(42)
        for j, (d, color) in enumerate(zip(data, colors)):
            jitter = rng.uniform(-0.15, 0.15, size=len(d))
            ax.scatter(
                positions[j] + jitter,
                d,
                color=color,
                s=18,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.3,
                zorder=3,
            )

        ax.set_title(ENV_LABELS[slug], fontweight="bold")
        ax.set_ylabel("Normalised Score")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[-1].set_xlabel("Algorithm")
    fig.tight_layout()
    save_and_copy(fig, "score_distribution.png")


# ── 5. Per-seed heatmap (1×2 algo panels) ───────────────────────────────
def plot_per_seed_heatmap() -> None:
    print("Generating per_seed_heatmap.png ...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    n_seeds = algo_score[ALGO_ORDER[0]].shape[0]

    for idx, algo in enumerate(ALGO_ORDER):
        ax = axes[idx]
        mat = algo_score[algo]
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="RdYlGn",
            vmin=-0.1,
            vmax=1.1,
            interpolation="nearest",
        )
        ax.set_title(ALGO_LABELS[algo], fontweight="bold")
        ax.set_xticks(range(len(ENV_SLUGS)))
        ax.set_xticklabels(
            [ENV_LABELS[s] for s in ENV_SLUGS],
            rotation=30,
            ha="right",
            fontsize=9,
        )
        ax.set_yticks(range(n_seeds))
        ax.set_yticklabels([f"S{i}" for i in range(n_seeds)], fontsize=8)
        if idx == 0:
            ax.set_ylabel("Seed")

        for si in range(n_seeds):
            for ei in range(len(ENV_SLUGS)):
                val = mat[si, ei]
                text_color = "white" if val < 0.3 or val > 0.85 else "black"
                ax.text(
                    ei,
                    si,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )

    fig.colorbar(
        im,
        ax=axes,
        location="right",
        fraction=0.05,
        pad=0.02,
        shrink=0.8,
        label="Normalised Score",
    )
    fig.tight_layout()
    save_and_copy(fig, "per_seed_heatmap.png")


# ── 6. Per-seed box+swarm with anomaly detection ────────────────────────
def plot_per_seed_boxswarm() -> None:
    print("Generating per_seed_boxswarm.png ...")
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for i, slug in enumerate(ENV_SLUGS):
        ax = axes[i]
        col_idx = ENV_SLUGS.index(slug)
        data = []
        labels = []
        colors = []
        for algo in ALGO_ORDER:
            scores = algo_score[algo][:, col_idx]
            data.append(scores)
            labels.append(ALGO_LABELS[algo])
            colors.append(ALGO_COLORS[algo])

        positions = np.arange(len(ALGO_ORDER))
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)

        rng = np.random.default_rng(42)
        for j, (d, color) in enumerate(zip(data, colors)):
            jitter = rng.uniform(-0.15, 0.15, size=len(d))
            ax.scatter(
                positions[j] + jitter,
                d,
                color=color,
                s=25,
                alpha=0.8,
                edgecolors="white",
                linewidth=0.4,
                zorder=3,
            )
            anomalies = d < 0
            if anomalies.any():
                ax.scatter(
                    positions[j] + jitter[anomalies],
                    d[anomalies],
                    color="red",
                    s=50,
                    marker="x",
                    linewidth=1.5,
                    zorder=4,
                )

        ax.set_title(ENV_LABELS[slug], fontweight="bold")
        ax.set_ylabel("Normalised Score")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.axhline(
            0,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
            label="Random baseline",
        )
        ax.axhline(
            1,
            color="gray",
            linestyle=":",
            linewidth=0.8,
            alpha=0.5,
            label="Max return",
        )

    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    axes[-1].set_xlabel("Algorithm")
    fig.tight_layout()
    save_and_copy(fig, "per_seed_boxswarm.png")


# ── 7. Final performance (grouped bars + 95% CI) ────────────────────────
def plot_final_performance() -> None:
    print("Generating final_performance.png ...")
    metric_names = ["iqm", "mean", "median"]
    metric_labels = ["IQM", "Mean", "Median"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

    for mi, (mname, mlabel) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[mi]
        x = np.arange(len(ALGO_ORDER))
        width = 0.5

        for ai, algo in enumerate(ALGO_ORDER):
            ce = algo_eval[algo]["cross_environment"]
            point = ce[mname]["point"]
            ci_lo = ce[mname]["ci_low"]
            ci_hi = ce[mname]["ci_high"]
            err_lo = point - ci_lo
            err_hi = ci_hi - point

            ax.bar(
                x[ai],
                point,
                width,
                color=ALGO_COLORS[algo],
                label=ALGO_LABELS[algo] if mi == 0 else None,
                alpha=0.8,
            )
            ax.errorbar(
                x[ai],
                point,
                yerr=[[err_lo], [err_hi]],
                fmt="none",
                ecolor="black",
                capsize=5,
                linewidth=1.5,
            )
            ax.text(
                x[ai],
                point - 0.03,
                f"{point:.3f}",
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
                color="white",
            )

        ax.set_title(mlabel, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([ALGO_LABELS[a] for a in ALGO_ORDER])
        ax.set_ylim(0, 1.1)
        if mi == 0:
            ax.set_ylabel("Normalised Score")

    axes[0].legend(loc="upper right", framealpha=0.9)
    fig.suptitle(
        "Cross-Environment Final Performance (95% Bootstrap CI)", fontweight="bold"
    )
    fig.tight_layout()
    save_and_copy(fig, "final_performance.png")


# ── 8. Performance profile ──────────────────────────────────────────────
def plot_performance_profile() -> None:
    print("Generating performance_profile.png ...")
    fig, ax = plt.subplots(figsize=(8, 5))

    for algo in ALGO_ORDER:
        pp = algo_eval[algo]["cross_environment"]["performance_profile"]
        tau = np.array(pp["tau"])
        vals = np.array(pp["values"])
        ci_lo = np.array(pp["ci_low"])
        ci_hi = np.array(pp["ci_high"])
        color = ALGO_COLORS[algo]

        ax.plot(tau, vals, color=color, label=ALGO_LABELS[algo], linewidth=2)
        ax.fill_between(tau, ci_lo, ci_hi, color=color, alpha=0.12)

    ax.set_xlabel("Normalised Score Threshold (τ)")
    ax.set_ylabel("Fraction of Runs ≥ τ")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("Performance Profile", fontweight="bold")
    fig.tight_layout()
    save_and_copy(fig, "performance_profile.png")


# ── 9. Optimality gap ───────────────────────────────────────────────────
def plot_optimality_gap() -> None:
    print("Generating optimality_gap.png ...")
    fig, ax = plt.subplots(figsize=(8, 3.5))

    y_pos = np.arange(len(ALGO_ORDER))
    for ai, algo in enumerate(ALGO_ORDER):
        og = algo_eval[algo]["cross_environment"]["optimality_gap"]
        point = og["point"]
        ci_lo = og["ci_low"]
        ci_hi = og["ci_high"]

        ax.barh(
            y_pos[ai],
            point,
            height=0.5,
            color=ALGO_COLORS[algo],
            alpha=0.8,
            label=ALGO_LABELS[algo],
        )
        ax.errorbar(
            point,
            y_pos[ai],
            xerr=[[point - ci_lo], [ci_hi - point]],
            fmt="none",
            ecolor="black",
            capsize=5,
            linewidth=1.5,
        )
        ax.text(
            ci_hi + 0.01,
            y_pos[ai],
            f"{point:.3f}",
            ha="left",
            va="center",
            fontsize=10,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([ALGO_LABELS[a] for a in ALGO_ORDER])
    ax.set_xlabel("Optimality Gap (lower is better)")
    ax.set_title("Optimality Gap — 95% Bootstrap CI", fontweight="bold")
    ax.set_xlim(0, 1.1)
    fig.tight_layout()
    save_and_copy(fig, "optimality_gap.png")


# ── 10. POI heatmap ─────────────────────────────────────────────────────
def plot_poi_heatmap() -> None:
    print("Generating poi_heatmap.png ...")
    if not pairwise_poi:
        print("  Skipped — no pairwise POI data.")
        return

    algos = sorted(ALGO_ORDER)
    n = len(algos)
    matrix = np.full((n, n), 0.5)
    for pair, val in pairwise_poi.items():
        parts = pair.split("_vs_")
        if len(parts) != 2:
            continue
        a, b = parts
        if a in algos and b in algos:
            i, j = algos.index(a), algos.index(b)
            matrix[i, j] = val
            matrix[j, i] = 1 - val

    labels = [ALGO_LABELS[a] for a in algos]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="RdBu", vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Y")
    ax.set_ylabel("X")

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            text_color = "white" if abs(val - 0.5) > 0.25 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=text_color,
            )

    ax.set_title("P(X > Y)", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="P(X > Y)")
    fig.tight_layout()
    save_and_copy(fig, "poi_heatmap.png")


# ── 11. Timing analysis ─────────────────────────────────────────────────
def plot_timing_analysis() -> None:
    print("Generating timing_analysis.png ...")
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(ENV_SLUGS))
    width = 0.35
    offset = [-width / 2, width / 2]

    has_data = False
    for ai, algo in enumerate(ALGO_ORDER):
        timing = algo_eval[algo].get("timing", {}).get("training", {})
        times = []
        for slug in ENV_SLUGS:
            t = timing.get(slug, {}).get("total_seconds", 0)
            times.append(t)
            if t > 0:
                has_data = True

        bars = ax.bar(
            x + offset[ai],
            times,
            width,
            color=ALGO_COLORS[algo],
            label=ALGO_LABELS[algo],
            alpha=0.8,
        )
        for bar, t in zip(bars, times):
            if t > 0:
                hours = t / 3600
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{hours:.1f}h",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.5,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="gray",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABELS[s] for s in ENV_SLUGS])
    ax.set_ylabel("Training Time (seconds)")
    ax.set_title("Training Time per Environment", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)

    if not has_data:
        ax.text(
            0.5,
            0.5,
            "No training timing data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )

    fig.tight_layout()
    save_and_copy(fig, "timing_analysis.png")


# ── 12 & 13. Sample efficiency per env ───────────────────────────────────
def plot_sample_efficiency_single(slug: str) -> None:
    fname = f"sample_efficiency_{slug}.png"
    print(f"Generating {fname} ...")
    fig, ax = plt.subplots(figsize=(8, 5))

    for algo in ALGO_ORDER:
        se = algo_se[algo][slug]
        ts = se["timesteps"] / 1_000_000
        color = ALGO_COLORS[algo]

        ax.plot(ts, se["iqm"], color=color, label=ALGO_LABELS[algo], linewidth=2)
        ax.fill_between(ts, se["ci_low"], se["ci_high"], color=color, alpha=0.12)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Environment Steps (millions)")
    ax.set_ylabel("IQM (normalised)")
    ax.set_title(f"Sample Efficiency — {ENV_LABELS[slug]}", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    save_and_copy(fig, fname)


# -- Main --
if __name__ == "__main__":
    plot_learning_curves_single("pong")
    plot_learning_curves_single("breakout")
    plot_combined_learning_curves()
    plot_score_distribution()
    plot_per_seed_heatmap()
    plot_per_seed_boxswarm()
    plot_final_performance()
    plot_performance_profile()
    plot_optimality_gap()
    plot_poi_heatmap()
    plot_timing_analysis()
    plot_sample_efficiency_single("pong")
    plot_sample_efficiency_single("breakout")
    print("\nDone — 13 figures regenerated.")
