import marimo

__generated_with = "0.19.10"
app = marimo.App(
    width="medium",
    app_title="A2C Evaluation Report",
    layout_file="layouts/report.slides.json",
)


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    tex_fonts = {
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update(tex_fonts)
    return Path, json, mo, np, plt, sns


@app.cell
def _(Path):
    # --- Config ---
    BASE_DIR = Path(__file__).parent.parent
    RESULTS_DIR = BASE_DIR / "results"
    METRICS_DIR = RESULTS_DIR / "metrics"
    FIGURES_DIR = RESULTS_DIR / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ALGO_COLORS = {
        "a2c": "#1f77b4",
        "dqn": "#ff7f0e",
        "ppo": "#2ca02c",
    }
    TEX_WIDTH = 412.56496
    # Scale factor so figures are readable in marimo — saved PNGs
    # can be rescaled for LaTeX later via \includegraphics[width=...]
    FIG_SCALE = 1.6
    DPI = 600


    def set_size(width_pt, fraction=1, subplots=(1, 1)):
        """Set figure dimensions, scaled up for notebook readability."""
        fig_width_pt = width_pt * fraction
        inches_per_pt = 1 / 72.27
        golden_ratio = (5**0.5 - 1) / 2
        fig_width_in = fig_width_pt * inches_per_pt * FIG_SCALE
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
        return (fig_width_in, fig_height_in)

    return (
        ALGO_COLORS,
        DPI,
        FIGURES_DIR,
        METRICS_DIR,
        RESULTS_DIR,
        TEX_WIDTH,
        set_size,
    )


@app.cell
def _(METRICS_DIR, RESULTS_DIR, json, np):
    # --- Load evaluation results ---
    with open(METRICS_DIR / "evaluation_results.json") as _f:
        eval_results = json.load(_f)

    env_slugs = eval_results["environments"]
    score_matrix = np.load(METRICS_DIR / "score_matrix.npy")
    random_baselines = eval_results["random_baselines"]  # slug → float

    # Per-env data dicts keyed by slug
    lc_data = {}
    se_data = {}
    env_configs = {}
    for _slug in env_slugs:
        lc_data[_slug] = np.load(METRICS_DIR / "learning_curves" / f"{_slug}.npz")
        se_data[_slug] = np.load(
            METRICS_DIR / "sample_efficiency" / f"{_slug}.npz"
        )
        _config_path = RESULTS_DIR / "a2c" / _slug / "config.json"
        with open(_config_path) as _f:
            env_configs[_slug] = json.load(_f)
    return (
        env_configs,
        env_slugs,
        eval_results,
        lc_data,
        random_baselines,
        score_matrix,
        se_data,
    )


@app.cell
def _(
    env_configs,
    env_slugs,
    eval_results,
    mo,
    random_baselines,
    score_matrix,
):
    n_seeds = score_matrix.shape[0]
    n_envs = score_matrix.shape[1]

    _env_rows = ""
    for _slug in env_slugs:
        _cfg = env_configs[_slug]
        _rand = random_baselines[_slug]
        _env_rows += (
            f"| `{_cfg['environment']}` | {_cfg['total_timesteps']:,} "
            f"| {_cfg['n_envs']} | {_cfg['eval_freq'] * _cfg['n_envs']:,} "
            f"| {_rand:.1f} | {_cfg['max_return']:.0f} |\n"
        )

    mo.md(f"""
    # A2C Evaluation Report

    **Algorithm:** A2C (Advantage Actor-Critic) with MlpPolicy (SB3 defaults)

    | Parameter | Value |
    |-----------|-------|
    | Seeds | {n_seeds} |
    | Environments | {n_envs} |
    | Eval episodes/seed | {eval_results["n_eval_episodes"]} |

    **Per-environment training config:**

    | Environment | Timesteps | VecEnvs | Eval every N steps | Random baseline | Max return |
    |-------------|-----------|---------|---------------------|-----------------|------------|
    {_env_rows}

    Metrics follow Agarwal et al. (2021) and Patterson et al. (2023).
    """)
    return


@app.cell
def _(
    ALGO_COLORS,
    DPI,
    FIGURES_DIR,
    TEX_WIDTH,
    env_configs,
    env_slugs,
    lc_data,
    mo,
    np,
    plt,
    random_baselines,
    set_size,
):
    # --- Per-env learning curves: 2-panel (mean±std, median+IQR) ---
    _figs = []
    for _slug in env_slugs:
        _ld = lc_data[_slug]
        _max_ret = env_configs[_slug]["max_return"]
        _rand = random_baselines[_slug]
        _ts = _ld["timesteps"] / 1000
        _mean = _ld["mean"]
        _std = _ld["std"]
        _median = _ld["median"]
        _p25 = _ld["p25"]
        _p75 = _ld["p75"]
        _c = ALGO_COLORS["a2c"]

        _fig, (_ax1, _ax2) = plt.subplots(
            1, 2, figsize=set_size(TEX_WIDTH, subplots=(1, 2)), sharey=True
        )

        _ax1.plot(_ts, _mean, color=_c, linewidth=1.0, label="Mean")
        _ax1.fill_between(
            _ts,
            _mean - _std,
            np.minimum(_mean + _std, _max_ret),
            alpha=0.2,
            color=_c,
        )
        _ax1.axhline(
            _max_ret,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"Max ({_max_ret:.0f})",
        )
        _ax1.axhline(
            _rand,
            color="orange",
            linestyle="--",
            alpha=0.6,
            label=f"Random ({_rand:.0f})",
        )
        _ax1.set_xlabel(r"Timesteps ($\times 10^3$)")
        _ax1.set_ylabel("Eval Return")
        _ax1.set_title(r"Mean $\pm$ Std")
        _ax1.legend()

        _ax2.plot(_ts, _median, color=_c, linewidth=1.0, label="Median")
        _ax2.fill_between(
            _ts,
            _p25,
            np.minimum(_p75, _max_ret),
            alpha=0.2,
            color=_c,
            label=r"IQR (25--75\%)",
        )
        _ax2.axhline(_max_ret, color="gray", linestyle="--", alpha=0.5)
        _ax2.axhline(_rand, color="orange", linestyle="--", alpha=0.6)
        _ax2.set_xlabel(r"Timesteps ($\times 10^3$)")
        _ax2.set_title("Median + IQR")
        _ax2.legend()

        _env_name = env_configs[_slug]["environment"]
        _fig.suptitle(f"Learning Curves — {_env_name}", y=1.02)
        _fig.tight_layout()
        _fig.savefig(
            FIGURES_DIR / f"learning_curves_{_slug}.png",
            format="png",
            dpi=DPI,
            bbox_inches="tight",
        )
        _figs.append(_fig)
    mo.vstack(_figs)
    return


@app.cell
def _(
    ALGO_COLORS,
    DPI,
    FIGURES_DIR,
    TEX_WIDTH,
    env_configs,
    env_slugs,
    plt,
    random_baselines,
    score_matrix,
    set_size,
    sns,
):
    # --- Score distribution: violin + strip plot (one panel per env) ---
    _n_envs = len(env_slugs)
    _fig, _axes = plt.subplots(
        1, _n_envs, figsize=set_size(TEX_WIDTH, subplots=(1, _n_envs))
    )
    if _n_envs == 1:
        _axes = [_axes]

    for _i, _slug in enumerate(env_slugs):
        _scores_norm = score_matrix[:, _i]
        _max_ret = env_configs[_slug]["max_return"]
        _rand = random_baselines[_slug]
        _raw = _scores_norm * (_max_ret - _rand) + _rand
        _env_name = env_configs[_slug]["environment"]
        _ax = _axes[_i]

        sns.violinplot(
            y=_raw, ax=_ax, inner=None, color=ALGO_COLORS["a2c"], alpha=0.6
        )
        sns.swarmplot(
            y=_raw, ax=_ax, color="black", edgecolor="white", linewidth=0.5,
            size=6, alpha=0.8, label="Seed" if _i == 0 else None,
        )
        _ax.axhline(
            _max_ret,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Max ({_max_ret:.0f})",
        )
        _ax.axhline(
            _rand,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label=f"Random ({_rand:.0f})",
        )
        _ax.set_title(_env_name)
        _ax.set_ylabel("Mean Eval Return")
        _ax.set_xlabel("")
        _ax.set_xticks([])
        _ax.legend(fontsize=8)

    _fig.suptitle("Score Distribution", y=1.02)
    _fig.tight_layout()
    _fig.savefig(
        FIGURES_DIR / "score_distribution.png",
        format="png",
        dpi=DPI,
        bbox_inches="tight",
    )
    _fig
    return


@app.cell
def _(
    ALGO_COLORS,
    DPI,
    FIGURES_DIR,
    TEX_WIDTH,
    env_configs,
    env_slugs,
    eval_results,
    mo,
    plt,
    se_data,
    set_size,
):
    # --- Per-env sample efficiency: IQM over training ---
    _figs = []
    for _slug in env_slugs:
        _sd = se_data[_slug]
        _ts = _sd["timesteps"] / 1000
        _iqm = _sd["iqm"]
        _ci_lo = _sd["ci_low"]
        _ci_hi = _sd["ci_high"]
        _auc = eval_results["per_environment"][_slug]["sample_efficiency_auc"]
        _env_name = env_configs[_slug]["environment"]
        _c = ALGO_COLORS["a2c"]

        _fig, _ax = plt.subplots(figsize=set_size(TEX_WIDTH))
        _ax.plot(_ts, _iqm, color=_c, linewidth=1.0, label="IQM")
        _ax.fill_between(_ts, _ci_lo, _ci_hi, alpha=0.2, color=_c)
        _ax.axhline(
            1.0, color="gray", linestyle="--", alpha=0.5, label="Max normalized"
        )
        _ax.set_xlabel(r"Timesteps ($\times 10^3$)")
        _ax.set_ylabel("Normalized IQM")
        _ax.set_title(f"Sample Efficiency — {_env_name} (AUC $= {_auc:.3f}$)")
        _ax.legend()
        _fig.tight_layout()
        _fig.savefig(
            FIGURES_DIR / f"sample_efficiency_{_slug}.png",
            format="png",
            dpi=DPI,
            bbox_inches="tight",
        )
        _figs.append(_fig)
    mo.vstack(_figs)
    return


@app.cell
def _(ALGO_COLORS, DPI, FIGURES_DIR, TEX_WIDTH, eval_results, plt, set_size):
    # --- Cross-env: Final performance bar chart with 95% CI ---
    _ce = eval_results["cross_environment"]
    _metric_names = ["mean", "median", "iqm"]
    _labels = ["Mean", "Median", "IQM"]
    _points = [_ce[_m]["point"] for _m in _metric_names]
    _ci_lows = [_ce[_m]["ci_low"] for _m in _metric_names]
    _ci_highs = [_ce[_m]["ci_high"] for _m in _metric_names]
    _errors_low = [_p - _lo for _p, _lo in zip(_points, _ci_lows)]
    _errors_high = [_hi - _p for _p, _hi in zip(_points, _ci_highs)]

    _fig, _ax = plt.subplots(figsize=set_size(TEX_WIDTH, fraction=0.75))
    _x = range(len(_labels))
    _bars = _ax.bar(
        _x,
        _points,
        yerr=[_errors_low, _errors_high],
        capsize=4,
        color=ALGO_COLORS["a2c"],
        alpha=0.8,
        width=0.5,
    )
    for _bar, _val in zip(_bars, _points):
        _ax.text(
            _bar.get_x() + _bar.get_width() / 2,
            _bar.get_height() / 2,
            f"{_val:.3f}",
            ha="center",
            va="center",
            fontweight="bold",
            color="white",
        )
    _ax.set_xticks(list(_x))
    _ax.set_xticklabels(_labels)
    _ax.set_ylabel("Normalized Score")
    _ax.set_title(r"Cross-Environment Final Performance (95\% Bootstrap CI)")
    _ax.set_ylim(0, 1.1)
    _fig.tight_layout()
    _fig.savefig(
        FIGURES_DIR / "final_performance.png",
        format="png",
        dpi=DPI,
        bbox_inches="tight",
    )
    _fig
    return


@app.cell
def _(
    ALGO_COLORS,
    DPI,
    FIGURES_DIR,
    TEX_WIDTH,
    eval_results,
    np,
    plt,
    set_size,
):
    # --- Cross-env: Performance profile ---
    _pp = eval_results["cross_environment"]["performance_profile"]
    _tau = np.array(_pp["tau"])
    _vals = np.array(_pp["values"])
    _ci_low = np.array(_pp["ci_low"])
    _ci_high = np.array(_pp["ci_high"])

    _fig, _ax = plt.subplots(figsize=set_size(TEX_WIDTH, fraction=0.75))
    _c = ALGO_COLORS["a2c"]
    _ax.plot(_tau, _vals, color=_c, linewidth=1.0, label="A2C")
    _ax.fill_between(_tau, _ci_low, _ci_high, alpha=0.2, color=_c)
    _ax.set_xlabel(r"Normalized Score Threshold ($\tau$)")
    _ax.set_ylabel(r"Fraction of Runs $\geq \tau$")
    _ax.set_title("Cross-Environment Performance Profile")
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1.05)
    _ax.legend()
    _fig.tight_layout()
    _fig.savefig(
        FIGURES_DIR / "performance_profile.png",
        format="png",
        dpi=DPI,
        bbox_inches="tight",
    )
    _fig
    return


@app.cell
def _(env_slugs, eval_results, mo):
    # --- Per-environment metrics tables ---
    _tables = ""
    for _slug in env_slugs:
        _pe = eval_results["per_environment"][_slug]
        _rows = [
            (
                "IQM",
                f"{_pe['final_iqm']['point']:.4f}",
                f"[{_pe['final_iqm']['ci_low']:.4f}, {_pe['final_iqm']['ci_high']:.4f}]",
            ),
            (
                "Mean",
                f"{_pe['final_mean']['point']:.4f}",
                f"[{_pe['final_mean']['ci_low']:.4f}, {_pe['final_mean']['ci_high']:.4f}]",
            ),
            (
                "Median",
                f"{_pe['final_median']['point']:.4f}",
                f"[{_pe['final_median']['ci_low']:.4f}, {_pe['final_median']['ci_high']:.4f}]",
            ),
            ("IQR (raw)", f"{_pe['reliability']['iqr']:.1f}", "---"),
            ("CVaR 0.1 (raw)", f"{_pe['reliability']['cvar_01']:.1f}", "---"),
            ("Min (raw)", f"{_pe['reliability']['min_score']:.1f}", "---"),
            ("Max (raw)", f"{_pe['reliability']['max_score']:.1f}", "---"),
            ("Sample Eff. AUC", f"{_pe['sample_efficiency_auc']:.4f}", "---"),
        ]
        _tbl = "| Metric | Value | 95\\% CI |\n|--------|-------|--------|\n"
        for _name, _val, _ci in _rows:
            _tbl += f"| {_name} | {_val} | {_ci} |\n"
        _tables += f"\n### `{_slug}`\n\n{_tbl}\n"

    mo.md(f"""
    ## Per-Environment Metrics

    {_tables}

    *Normalized: (raw − random) / (max − random), per Agarwal et al. (2021). Bootstrap CIs use 50,000 resamples.*
    """)
    return


@app.cell
def _(eval_results, mo):
    # --- Cross-environment aggregate metrics table ---
    _ce = eval_results["cross_environment"]
    _rows = [
        (
            "IQM",
            f"{_ce['iqm']['point']:.4f}",
            f"[{_ce['iqm']['ci_low']:.4f}, {_ce['iqm']['ci_high']:.4f}]",
        ),
        (
            "Mean",
            f"{_ce['mean']['point']:.4f}",
            f"[{_ce['mean']['ci_low']:.4f}, {_ce['mean']['ci_high']:.4f}]",
        ),
        (
            "Median",
            f"{_ce['median']['point']:.4f}",
            f"[{_ce['median']['ci_low']:.4f}, {_ce['median']['ci_high']:.4f}]",
        ),
        (
            "Optimality Gap",
            f"{_ce['optimality_gap']['point']:.4f}",
            f"[{_ce['optimality_gap']['ci_low']:.4f}, {_ce['optimality_gap']['ci_high']:.4f}]",
        ),
    ]
    _tbl = "| Metric | Value | 95\\% CI |\n|--------|-------|--------|\n"
    for _name, _val, _ci in _rows:
        _tbl += f"| {_name} | {_val} | {_ci} |\n"

    _n_envs = len(eval_results["environments"])
    _n_seeds = eval_results["n_seeds"]
    mo.md(f"""
    ## Cross-Environment Aggregate Metrics

    Score matrix shape: ({_n_seeds}, {_n_envs}) — {_n_seeds * _n_envs} data points.

    {_tbl}

    *Bootstrap CIs use 50,000 resamples across all environment-seed pairs.*

    **Caveat:** With only M=2 environments, cross-environment aggregates have limited statistical power.
    These metrics become more informative as more environments are added.
    """)
    return


@app.cell
def _(env_slugs, eval_results, mo, np):
    # --- Timing / Reproducibility ---
    _timing = eval_results.get("timing", {})
    _train_timing = _timing.get("training", {})
    _eval_timing = _timing.get("evaluation", {})

    _rows = ""
    for _slug in env_slugs:
        _tt = _train_timing.get(_slug, {})
        _per_seed = _tt.get("per_seed_seconds", [])
        _total_train = _tt.get("total_seconds", 0)
        _eval_sec = _eval_timing.get("per_env_seconds", {}).get(_slug, 0)

        if _per_seed:
            _arr = np.array(_per_seed)
            _mean_s = f"{_arr.mean():.1f}"
            _std_s = f"{_arr.std():.1f}"
        else:
            _mean_s = "---"
            _std_s = "---"

        _rows += f"| `{_slug}` | {_total_train:.1f} | {_mean_s} | {_std_s} | {_eval_sec:.1f} |\n"

    _total_eval = _eval_timing.get("total_seconds", 0)

    mo.md(f"""
    ## Reproducibility & Timing

    | Environment | Train Total (s) | Train Mean/Seed (s) | Train Std/Seed (s) | Eval (s) |
    |-------------|-----------------|---------------------|--------------------|----------|
    {_rows}
    | **Total eval** | | | | **{_total_eval:.1f}** |

    **Note:** Same-seed results are reproducible on same hardware + platform only.
    CPU vs GPU results will differ (PyTorch limitation).
    """)
    return


if __name__ == "__main__":
    app.run()
