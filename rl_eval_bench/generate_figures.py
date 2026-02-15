"""Regenerate publication-quality figures for the paper.

Produces 4 multi-panel PNGs that were too small in the original marimo export:
  1. combined_learning_curves.png  — 3 envs stacked vertically
  2. score_distribution.png        — 3 envs stacked vertically
  3. per_seed_heatmap.png          — 5 algo heatmaps in 2x3 grid
  4. per_seed_boxswarm.png         — 3 envs stacked vertically

Saves to results/figures/ and copies to paper_overleaf/4_chapter/assets/.
"""

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
ASSETS_DIR = BASE_DIR.parent / "paper_overleaf" / "4_chapter" / "assets"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
ALGO_ORDER = ["ppo", "a2c", "rppo", "dqn", "qrdqn"]
ALGO_COLORS = {
    "a2c": "#1f77b4",
    "dqn": "#ff7f0e",
    "ppo": "#2ca02c",
    "qrdqn": "#d62728",
    "rppo": "#9467bd",
}
ALGO_LABELS = {
    "a2c": "A2C",
    "dqn": "DQN",
    "ppo": "PPO",
    "qrdqn": "QR-DQN",
    "rppo": "RecurrentPPO",
}
ENV_SLUGS = ["cartpole-v1", "lunarlander-v3", "acrobot-v1"]
ENV_LABELS = {
    "cartpole-v1": "CartPole-v1",
    "lunarlander-v3": "LunarLander-v3",
    "acrobot-v1": "Acrobot-v1",
}

# ── Data loading ───────────────────────────────────────────────────────────
algo_eval = {}
algo_lc = {}
algo_score = {}
algo_raw_score = {}

for algo in ALGO_ORDER:
    algo_dir = METRICS_DIR / algo
    with open(algo_dir / "evaluation_results.json") as f:
        algo_eval[algo] = json.load(f)
    algo_score[algo] = np.load(algo_dir / "score_matrix.npy")
    algo_raw_score[algo] = np.load(algo_dir / "raw_score_matrix.npy")
    algo_lc[algo] = {}
    for slug in ENV_SLUGS:
        algo_lc[algo][slug] = np.load(algo_dir / "learning_curves" / f"{slug}.npz")

random_baselines = algo_eval[ALGO_ORDER[0]]["random_baselines"]

# ── Shared style ───────────────────────────────────────────────────────────
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
    shutil.copy2(src, ASSETS_DIR / name)
    print(f"  Saved {src} -> {ASSETS_DIR / name}")
    plt.close(fig)


# ── 1. Combined Learning Curves (3 rows x 1 col) ──────────────────────────
def plot_combined_learning_curves() -> None:
    print("Generating combined_learning_curves.png ...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
    for i, slug in enumerate(ENV_SLUGS):
        ax = axes[i]
        for algo in ALGO_ORDER:
            lc = algo_lc[algo][slug]
            ts = lc["timesteps"] / 1000  # kilo-steps
            median = lc["median"]
            p25, p75 = lc["p25"], lc["p75"]
            color = ALGO_COLORS[algo]
            label = ALGO_LABELS[algo]
            ax.plot(ts, median, color=color, label=label, linewidth=1.5)
            ax.fill_between(ts, p25, p75, color=color, alpha=0.15)
        ax.set_title(ENV_LABELS[slug], fontweight="bold")
        ax.set_ylabel("Return (median ± IQR)")
        if i == 2:
            ax.set_xlabel("Environment Steps (×1000)")
        # Random baseline
        rb = random_baselines[slug]
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
    axes[0].legend(loc="lower right", ncol=3, framealpha=0.9)
    fig.tight_layout()
    save_and_copy(fig, "combined_learning_curves.png")


# ── 2. Score Distribution (3 rows x 1 col) ────────────────────────────────
def plot_score_distribution() -> None:
    print("Generating score_distribution.png ...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
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

        # Overlay individual seeds as scatter
        for j, (d, color) in enumerate(zip(data, colors)):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(d))
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


# ── 3. Per-Seed Heatmap (2x3 grid, 5 algo panels) ────────────────────────
def plot_per_seed_heatmap() -> None:
    print("Generating per_seed_heatmap.png ...")
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes_flat = axes.flatten()

    n_seeds = algo_score[ALGO_ORDER[0]].shape[0]

    for idx, algo in enumerate(ALGO_ORDER):
        ax = axes_flat[idx]
        mat = algo_score[algo]  # (15, 3)
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
            [ENV_LABELS[s] for s in ENV_SLUGS], rotation=30, ha="right", fontsize=9
        )
        ax.set_yticks(range(n_seeds))
        ax.set_yticklabels([f"S{i}" for i in range(n_seeds)], fontsize=8)
        if idx % 3 == 0:
            ax.set_ylabel("Seed")

        # Annotate cells
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
                    fontsize=6.5,
                    color=text_color,
                )

    # Hide the 6th subplot
    axes_flat[5].set_visible(False)

    # Colorbar
    cbar = fig.colorbar(
        im, ax=axes_flat[5], location="left", fraction=0.6, pad=0.05, shrink=0.9
    )
    cbar.set_label("Normalised Score")

    fig.tight_layout()
    save_and_copy(fig, "per_seed_heatmap.png")


# ── 4. Per-Seed Box+Swarm (3 rows x 1 col) ────────────────────────────────
def plot_per_seed_boxswarm() -> None:
    print("Generating per_seed_boxswarm.png ...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

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

        # Swarm-like scatter with jitter
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
            # Mark anomalies (below 0)
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
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Random baseline")
        ax.axhline(1, color="gray", linestyle=":", linewidth=0.8, alpha=0.5, label="Max return")

    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    axes[-1].set_xlabel("Algorithm")
    fig.tight_layout()
    save_and_copy(fig, "per_seed_boxswarm.png")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_combined_learning_curves()
    plot_score_distribution()
    plot_per_seed_heatmap()
    plot_per_seed_boxswarm()
    print("\nDone — 4 figures regenerated.")
