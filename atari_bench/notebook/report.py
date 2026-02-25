import marimo

__generated_with = "0.19.10"
app = marimo.App(
    width="full",
    app_title="Atari DQN vs PPO Evaluation Report",
)


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    return Path, alt, json, mo, np, pd


@app.cell
def _(Path, alt):
    # --- Config ---
    BASE_DIR = Path(__file__).parent.parent
    RESULTS_DIR = BASE_DIR / "results"
    METRICS_DIR = RESULTS_DIR / "metrics"

    ALGO_COLORS = {
        "dqn": "#ff7f0e",
        "ppo": "#2ca02c",
    }
    ALGO_LABELS = {
        "dqn": "DQN",
        "ppo": "PPO",
    }

    ALGO_COLOR_SCALE = alt.Scale(
        domain=list(ALGO_LABELS.values()), range=list(ALGO_COLORS.values())
    )

    def algo_selection():
        """Reusable legend-bound selection for algo toggle."""
        return alt.selection_point(fields=["Algorithm"], bind="legend")

    return (
        ALGO_COLORS,
        ALGO_COLOR_SCALE,
        ALGO_LABELS,
        METRICS_DIR,
        RESULTS_DIR,
        algo_selection,
    )


@app.cell
def _(ALGO_COLORS, ALGO_LABELS, METRICS_DIR, RESULTS_DIR, json, np):
    # --- Discover algos from metrics dirs ---
    all_algos = sorted(
        d.name
        for d in METRICS_DIR.iterdir()
        if d.is_dir() and (d / "evaluation_results.json").exists()
    )

    algo_eval_results = {}
    algo_lc_data = {}
    algo_se_data = {}
    algo_score_matrices = {}
    algo_raw_score_matrices = {}
    algo_configs = {}

    for _algo in all_algos:
        _algo_metrics = METRICS_DIR / _algo

        with open(_algo_metrics / "evaluation_results.json") as _f:
            algo_eval_results[_algo] = json.load(_f)

        algo_score_matrices[_algo] = np.load(_algo_metrics / "score_matrix.npy")

        _raw_path = _algo_metrics / "raw_score_matrix.npy"
        if _raw_path.exists():
            algo_raw_score_matrices[_algo] = np.load(_raw_path)

        _slugs = algo_eval_results[_algo]["environments"]
        algo_lc_data[_algo] = {}
        algo_se_data[_algo] = {}
        algo_configs[_algo] = {}
        for _slug in _slugs:
            algo_lc_data[_algo][_slug] = np.load(
                _algo_metrics / "learning_curves" / f"{_slug}.npz"
            )
            algo_se_data[_algo][_slug] = np.load(
                _algo_metrics / "sample_efficiency" / f"{_slug}.npz"
            )
            _config_path = RESULTS_DIR / _algo / _slug / "config.json"
            with open(_config_path) as _f:
                algo_configs[_algo][_slug] = json.load(_f)

    # Pairwise P(X>Y)
    _poi_path = METRICS_DIR / "pairwise_poi.json"
    pairwise_poi = {}
    if _poi_path.exists():
        with open(_poi_path) as _f:
            pairwise_poi = json.load(_f)

    # Shared across all algos (from first algo)
    _first = all_algos[0]
    env_slugs = algo_eval_results[_first]["environments"]
    random_baselines = algo_eval_results[_first]["random_baselines"]
    env_configs = algo_configs[_first]

    # Altair color helpers — filtered to only discovered algos
    algo_color_domain = [ALGO_LABELS[a] for a in all_algos]
    algo_color_range = [ALGO_COLORS[a] for a in all_algos]
    return (
        algo_color_domain,
        algo_color_range,
        algo_configs,
        algo_eval_results,
        algo_lc_data,
        algo_raw_score_matrices,
        algo_score_matrices,
        algo_se_data,
        all_algos,
        env_configs,
        env_slugs,
        pairwise_poi,
        random_baselines,
    )


@app.cell
def _(
    algo_eval_results,
    all_algos,
    env_configs,
    env_slugs,
    mo,
    random_baselines,
):
    # --- Header + overview table ---
    _first = all_algos[0]
    _n_seeds = algo_eval_results[_first]["n_seeds"]
    _n_envs = len(env_slugs)
    _n_eval_episodes = algo_eval_results[_first]["n_eval_episodes"]
    _algo_list = ", ".join(a.upper() for a in all_algos)

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
    # Atari DQN vs PPO — Evaluation Report

    **Algorithms:** {_algo_list} | **Policy:** CnnPolicy | **Device:** GPU

    | Parameter | Value |
    |-----------|-------|
    | Seeds | {_n_seeds} |
    | Environments | {_n_envs} |
    | Training budget | 5M steps/env |
    | Eval episodes/seed | {_n_eval_episodes} |

    **Per-environment training config:**

    | Environment | Timesteps | VecEnvs | Eval every N steps | Random baseline | Max return |
    |-------------|-----------|---------|---------------------|-----------------|------------|
    {_env_rows}

    Metrics follow Agarwal et al. (2021) and Patterson et al. (2023).
    """)
    return


@app.cell
def _(
    ALGO_COLOR_SCALE,
    ALGO_LABELS,
    algo_lc_data,
    algo_selection,
    all_algos,
    alt,
    env_configs,
    mo,
    pd,
    random_baselines,
):
    # --- Learning curves — Pong: mean±std ---
    _slug = "pong"
    _env_name = env_configs[_slug]["environment"]
    _rand = random_baselines[_slug]

    _records = []
    for _algo in all_algos:
        if _slug not in algo_lc_data[_algo]:
            continue
        _ld = algo_lc_data[_algo][_slug]
        _ts = _ld["timesteps"] / 1_000_000
        for _k in range(len(_ts)):
            _records.append(
                {
                    "Steps (M)": float(_ts[_k]),
                    "Return": float(_ld["mean"][_k]),
                    "Upper": float(_ld["mean"][_k] + _ld["std"][_k]),
                    "Lower": float(_ld["mean"][_k] - _ld["std"][_k]),
                    "Algorithm": ALGO_LABELS[_algo],
                }
            )

    _df = pd.DataFrame(_records)
    _sel = algo_selection()
    _line = (
        alt.Chart(_df)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("Steps (M):Q"),
            y=alt.Y("Return:Q"),
            color=alt.Color("Algorithm:N", scale=ALGO_COLOR_SCALE),
            opacity=alt.condition(_sel, alt.value(1), alt.value(0.15)),
            tooltip=["Algorithm:N", "Steps (M):Q", "Return:Q"],
        )
        .add_params(_sel)
    )
    _band = (
        alt.Chart(_df)
        .mark_area()
        .encode(
            x="Steps (M):Q",
            y="Lower:Q",
            y2="Upper:Q",
            color=alt.Color("Algorithm:N", scale=ALGO_COLOR_SCALE),
            opacity=alt.condition(_sel, alt.value(0.15), alt.value(0.03)),
        )
        .add_params(_sel)
    )
    _ref = (
        alt.Chart(pd.DataFrame([{"y": _rand, "label": f"Random ({_rand:.0f})"}]))
        .mark_rule(strokeDash=[4, 4], opacity=0.6)
        .encode(y="y:Q")
    )

    _chart_pong_mean = (
        (_band + _line + _ref)
        .properties(
            title=f"{_env_name} — Mean ± Std",
            width=550,
            height=300,
        )
        .interactive()
    )

    mo.md("## Learning Curves — Pong")
    _chart_pong_mean
    return


@app.cell
def _(
    ALGO_COLOR_SCALE,
    ALGO_LABELS,
    algo_lc_data,
    algo_selection,
    all_algos,
    alt,
    env_configs,
    mo,
    pd,
    random_baselines,
):
    # --- Learning curves — Breakout: mean±std ---
    _slug = "breakout"
    _env_name = env_configs[_slug]["environment"]
    _rand = random_baselines[_slug]

    _records = []
    for _algo in all_algos:
        if _slug not in algo_lc_data[_algo]:
            continue
        _ld = algo_lc_data[_algo][_slug]
        _ts = _ld["timesteps"] / 1_000_000
        for _k in range(len(_ts)):
            _records.append(
                {
                    "Steps (M)": float(_ts[_k]),
                    "Return": float(_ld["mean"][_k]),
                    "Upper": float(_ld["mean"][_k] + _ld["std"][_k]),
                    "Lower": float(_ld["mean"][_k] - _ld["std"][_k]),
                    "Algorithm": ALGO_LABELS[_algo],
                }
            )

    _df = pd.DataFrame(_records)
    _sel = algo_selection()
    _line = (
        alt.Chart(_df)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("Steps (M):Q"),
            y=alt.Y("Return:Q"),
            color=alt.Color("Algorithm:N", scale=ALGO_COLOR_SCALE),
            opacity=alt.condition(_sel, alt.value(1), alt.value(0.15)),
            tooltip=["Algorithm:N", "Steps (M):Q", "Return:Q"],
        )
        .add_params(_sel)
    )
    _band = (
        alt.Chart(_df)
        .mark_area()
        .encode(
            x="Steps (M):Q",
            y="Lower:Q",
            y2="Upper:Q",
            color=alt.Color("Algorithm:N", scale=ALGO_COLOR_SCALE),
            opacity=alt.condition(_sel, alt.value(0.15), alt.value(0.03)),
        )
        .add_params(_sel)
    )
    _ref = (
        alt.Chart(pd.DataFrame([{"y": _rand, "label": f"Random ({_rand:.0f})"}]))
        .mark_rule(strokeDash=[4, 4], opacity=0.6)
        .encode(y="y:Q")
    )

    _chart_breakout_mean = (
        (_band + _line + _ref)
        .properties(
            title=f"{_env_name} — Mean ± Std",
            width=550,
            height=300,
        )
        .interactive()
    )

    mo.md("## Learning Curves — Breakout")
    _chart_breakout_mean
    return


@app.cell
def _(
    ALGO_COLOR_SCALE,
    ALGO_LABELS,
    algo_lc_data,
    algo_selection,
    all_algos,
    alt,
    env_configs,
    env_slugs,
    pd,
    random_baselines,
):
    # --- Learning curves — median±IQR variant (both envs, faceted) ---
    _all_records = []
    for _slug in env_slugs:
        _env_name = env_configs[_slug]["environment"]
        for _algo in all_algos:
            if _slug not in algo_lc_data[_algo]:
                continue
            _ld = algo_lc_data[_algo][_slug]
            _ts = _ld["timesteps"] / 1_000_000
            for _k in range(len(_ts)):
                _all_records.append(
                    {
                        "Steps (M)": float(_ts[_k]),
                        "Return": float(_ld["median"][_k]),
                        "Upper": float(_ld["p75"][_k]),
                        "Lower": float(_ld["p25"][_k]),
                        "Algorithm": ALGO_LABELS[_algo],
                        "Environment": _env_name,
                    }
                )

    _df = pd.DataFrame(_all_records)
    _sel = algo_selection()
    _env_order = [env_configs[s]["environment"] for s in env_slugs]

    _line = (
        alt.Chart(_df)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("Steps (M):Q"),
            y=alt.Y("Return:Q"),
            color=alt.Color("Algorithm:N", scale=ALGO_COLOR_SCALE),
            opacity=alt.condition(_sel, alt.value(1), alt.value(0.15)),
            tooltip=["Algorithm:N", "Steps (M):Q", "Return:Q"],
        )
        .add_params(_sel)
    )
    _band = (
        alt.Chart(_df)
        .mark_area()
        .encode(
            x="Steps (M):Q",
            y="Lower:Q",
            y2="Upper:Q",
            color=alt.Color("Algorithm:N", scale=ALGO_COLOR_SCALE),
            opacity=alt.condition(_sel, alt.value(0.15), alt.value(0.03)),
        )
        .add_params(_sel)
    )
    _chart = (
        alt.layer(_line, _band)
        .properties(width=350, height=250, title="Median ± IQR")
        .interactive()
        .facet(column=alt.Column("Environment:N", sort=_env_order))
        .resolve_scale(y="independent")
    )
    _chart
    return


@app.cell
def _(
    ALGO_COLOR_SCALE,
    ALGO_LABELS,
    algo_score_matrices,
    all_algos,
    alt,
    env_configs,
    env_slugs,
    pd,
    random_baselines,
):
    # --- Score distribution: boxplot + strip per env ---
    _all_records = []
    for _i, _slug in enumerate(env_slugs):
        _max_ret = env_configs[_slug]["max_return"]
        _rand = random_baselines[_slug]
        _env_name = env_configs[_slug]["environment"]
        for _algo in all_algos:
            _sm = algo_score_matrices[_algo]
            _scores_norm = _sm[:, _i]
            _raw = _scores_norm * (_max_ret - _rand) + _rand
            for _si, _s in enumerate(_raw):
                _all_records.append(
                    {
                        "Score": float(_s),
                        "Algorithm": ALGO_LABELS[_algo],
                        "Environment": _env_name,
                        "Seed": _si,
                    }
                )

    _adf = pd.DataFrame(_all_records)
    _base = alt.Chart(_adf)
    _box = _base.mark_boxplot(
        extent="min-max", size=30, median={"color": "black"}
    ).encode(
        x=alt.X("Algorithm:N"),
        y=alt.Y("Score:Q"),
        color=alt.Color("Algorithm:N", scale=ALGO_COLOR_SCALE),
    )
    _points = _base.mark_circle(size=30, opacity=0.6).encode(
        x=alt.X("Algorithm:N"),
        y=alt.Y("Score:Q"),
        color=alt.Color("Algorithm:N", scale=ALGO_COLOR_SCALE),
        tooltip=["Algorithm:N", "Score:Q", "Seed:Q"],
    )
    _chart = (
        alt.layer(_box, _points)
        .properties(width=250, height=280)
        .interactive()
        .facet(column=alt.Column("Environment:N"))
        .resolve_scale(y="independent")
    )
    _chart
    return


@app.cell
def _(
    ALGO_LABELS,
    algo_color_domain,
    algo_score_matrices,
    all_algos,
    alt,
    env_slugs,
    pd,
):
    # --- Per-seed normalized score heatmap ---
    _n_seeds = algo_score_matrices[all_algos[0]].shape[0]

    _heat_records = []
    for _algo in all_algos:
        _sm = algo_score_matrices[_algo]
        for _si in range(_n_seeds):
            for _ei, _slug in enumerate(env_slugs):
                _val = float(_sm[_si, _ei])
                _heat_records.append(
                    {
                        "Seed": _si,
                        "Environment": _slug,
                        "Algorithm": ALGO_LABELS[_algo],
                        "Normalized Score": _val,
                        "label": f"{_val:.2f}",
                    }
                )

    _hdf = pd.DataFrame(_heat_records)
    _rect = (
        alt.Chart(_hdf)
        .mark_rect()
        .encode(
            x=alt.X("Environment:N"),
            y=alt.Y("Seed:O"),
            color=alt.Color(
                "Normalized Score:Q",
                scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]),
            ),
            tooltip=["Algorithm:N", "Environment:N", "Seed:O", "Normalized Score:Q"],
        )
    )
    _text = (
        alt.Chart(_hdf)
        .mark_text(fontSize=9)
        .encode(
            x=alt.X("Environment:N"),
            y=alt.Y("Seed:O"),
            text="label:N",
            color=alt.condition(
                'datum["Normalized Score"] < 0.4 || datum["Normalized Score"] > 0.8',
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )
    _chart = (
        (_rect + _text)
        .properties(width=120, height=300)
        .facet(column=alt.Column("Algorithm:N", sort=algo_color_domain))
    )
    _chart
    return


@app.cell
def _(
    ALGO_COLOR_SCALE,
    ALGO_LABELS,
    algo_score_matrices,
    all_algos,
    alt,
    env_configs,
    env_slugs,
    mo,
    pd,
):
    # --- Per-seed box/swarm with anomaly detection ---
    _all_records = []
    _anomaly_records = []
    for _i, _slug in enumerate(env_slugs):
        _env_name = env_configs[_slug]["environment"]
        for _algo in all_algos:
            _sm = algo_score_matrices[_algo]
            _norm_scores = _sm[:, _i]
            for _si, _score in enumerate(_norm_scores):
                _all_records.append(
                    {
                        "Normalized Score": float(_score),
                        "Algorithm": ALGO_LABELS[_algo],
                        "Environment": _env_name,
                        "Seed": _si,
                    }
                )
                if _score < 0:
                    _anomaly_records.append(
                        {
                            "Algorithm": ALGO_LABELS[_algo],
                            "Environment": _slug,
                            "Seed Index": _si,
                            "Normalized Score": f"{float(_score):.4f}",
                        }
                    )

    _df = pd.DataFrame(_all_records)
    _df["Baseline"] = 0.0

    _base = alt.Chart(_df)
    _box = _base.mark_boxplot(
        extent="min-max", size=30, median={"color": "black"}
    ).encode(
        x=alt.X("Algorithm:N"),
        y=alt.Y("Normalized Score:Q"),
        color=alt.Color("Algorithm:N", scale=ALGO_COLOR_SCALE),
    )
    _points = _base.mark_circle(size=25, opacity=0.6).encode(
        x=alt.X("Algorithm:N"),
        y=alt.Y("Normalized Score:Q"),
        color=alt.Color("Algorithm:N", scale=ALGO_COLOR_SCALE),
        tooltip=["Algorithm:N", "Normalized Score:Q", "Seed:Q"],
    )
    _baseline = _base.mark_rule(strokeDash=[4, 4], color="orange", opacity=0.7).encode(
        y="Baseline:Q"
    )

    _chart = (
        alt.layer(_box, _points, _baseline)
        .properties(width=250, height=250)
        .interactive()
        .facet(column=alt.Column("Environment:N"))
        .resolve_scale(y="independent")
    )

    if _anomaly_records:
        _anom_df = pd.DataFrame(_anomaly_records)
        _anom_md = f"\n\n**Anomalous seeds** (score below random baseline):\n\n{_anom_df.to_markdown(index=False)}"
    else:
        _anom_md = (
            "\n\n*No anomalous seeds detected (all scores above random baseline).*"
        )

    mo.vstack([_chart, mo.md(_anom_md)])
    return


@app.cell
def _(
    ALGO_COLOR_SCALE,
    ALGO_LABELS,
    algo_se_data,
    algo_selection,
    all_algos,
    alt,
    env_configs,
    mo,
    pd,
):
    # --- Sample efficiency — Pong: IQM + CI bands ---
    _slug = "pong"
    _env_name = env_configs[_slug]["environment"]
    _records = []
    for _algo in all_algos:
        if _slug not in algo_se_data[_algo]:
            continue
        _sd = algo_se_data[_algo][_slug]
        _ts = _sd["timesteps"] / 1_000_000
        for _k in range(len(_ts)):
            _records.append(
                {
                    "Steps (M)": float(_ts[_k]),
                    "IQM": float(_sd["iqm"][_k]),
                    "CI Low": float(_sd["ci_low"][_k]),
                    "CI High": float(_sd["ci_high"][_k]),
                    "Algorithm": ALGO_LABELS[_algo],
                }
            )

    _df = pd.DataFrame(_records)
    _sel = algo_selection()
    _line = (
        alt.Chart(_df)
        .mark_line()
        .encode(
            x=alt.X("Steps (M):Q"),
            y=alt.Y("IQM:Q"),
            color=alt.Color("Algorithm:N", scale=ALGO_COLOR_SCALE),
            opacity=alt.condition(_sel, alt.value(1), alt.value(0.15)),
            tooltip=["Algorithm:N", "Steps (M):Q", "IQM:Q"],
        )
        .add_params(_sel)
    )
    _band = (
        alt.Chart(_df)
        .mark_area(opacity=0.12)
        .encode(
            x="Steps (M):Q",
            y="CI Low:Q",
            y2="CI High:Q",
            color=alt.Color("Algorithm:N", scale=ALGO_COLOR_SCALE),
            opacity=alt.condition(_sel, alt.value(0.12), alt.value(0.02)),
        )
        .add_params(_sel)
    )
    _ref = (
        alt.Chart(pd.DataFrame([{"y": 1.0}]))
        .mark_rule(strokeDash=[4, 4], color="gray", opacity=0.5)
        .encode(y="y:Q")
    )
    _chart = (
        (_band + _line + _ref)
        .properties(
            title=f"Sample Efficiency — {_env_name}",
            width=550,
            height=300,
        )
        .interactive()
    )

    mo.md("## Sample Efficiency")
    _chart
    return


@app.cell
def _(
    ALGO_COLOR_SCALE,
    ALGO_LABELS,
    algo_se_data,
    algo_selection,
    all_algos,
    alt,
    env_configs,
    pd,
):
    # --- Sample efficiency — Breakout: IQM + CI bands ---
    _slug = "breakout"
    _env_name = env_configs[_slug]["environment"]
    _records = []
    for _algo in all_algos:
        if _slug not in algo_se_data[_algo]:
            continue
        _sd = algo_se_data[_algo][_slug]
        _ts = _sd["timesteps"] / 1_000_000
        for _k in range(len(_ts)):
            _records.append(
                {
                    "Steps (M)": float(_ts[_k]),
                    "IQM": float(_sd["iqm"][_k]),
                    "CI Low": float(_sd["ci_low"][_k]),
                    "CI High": float(_sd["ci_high"][_k]),
                    "Algorithm": ALGO_LABELS[_algo],
                }
            )

    _df = pd.DataFrame(_records)
    _sel = algo_selection()
    _line = (
        alt.Chart(_df)
        .mark_line()
        .encode(
            x=alt.X("Steps (M):Q"),
            y=alt.Y("IQM:Q"),
            color=alt.Color("Algorithm:N", scale=ALGO_COLOR_SCALE),
            opacity=alt.condition(_sel, alt.value(1), alt.value(0.15)),
            tooltip=["Algorithm:N", "Steps (M):Q", "IQM:Q"],
        )
        .add_params(_sel)
    )
    _band = (
        alt.Chart(_df)
        .mark_area(opacity=0.12)
        .encode(
            x="Steps (M):Q",
            y="CI Low:Q",
            y2="CI High:Q",
            color=alt.Color("Algorithm:N", scale=ALGO_COLOR_SCALE),
            opacity=alt.condition(_sel, alt.value(0.12), alt.value(0.02)),
        )
        .add_params(_sel)
    )
    _ref = (
        alt.Chart(pd.DataFrame([{"y": 1.0}]))
        .mark_rule(strokeDash=[4, 4], color="gray", opacity=0.5)
        .encode(y="y:Q")
    )
    _chart = (
        (_band + _line + _ref)
        .properties(
            title=f"Sample Efficiency — {_env_name}",
            width=550,
            height=300,
        )
        .interactive()
    )
    _chart
    return


@app.cell
def _(
    ALGO_COLOR_SCALE,
    ALGO_LABELS,
    algo_color_domain,
    algo_eval_results,
    all_algos,
    alt,
    mo,
    pd,
):
    # --- Cross-env final performance: grouped bar (IQM/Mean/Median + CI) ---
    _algos_with_ce = [
        a for a in all_algos if algo_eval_results[a].get("cross_environment")
    ]
    mo.stop(
        not _algos_with_ce,
        mo.md("*Cross-environment metrics skipped (need >= 2 environments).*"),
    )

    _metric_names = ["mean", "median", "iqm"]
    _labels = ["Mean", "Median", "IQM"]
    _bar_records = []
    for _algo in _algos_with_ce:
        _ce = algo_eval_results[_algo]["cross_environment"]
        for _mi, _m in enumerate(_metric_names):
            _bar_records.append(
                {
                    "Metric": _labels[_mi],
                    "Algorithm": ALGO_LABELS[_algo],
                    "Value": _ce[_m]["point"],
                    "CI Low": _ce[_m]["ci_low"],
                    "CI High": _ce[_m]["ci_high"],
                }
            )

    _bdf = pd.DataFrame(_bar_records)

    _bars = (
        alt.Chart(_bdf)
        .mark_bar()
        .encode(
            x=alt.X("Algorithm:N", sort=algo_color_domain),
            y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1.1])),
            color=alt.Color("Algorithm:N", scale=ALGO_COLOR_SCALE),
            tooltip=["Algorithm:N", "Metric:N", "Value:Q", "CI Low:Q", "CI High:Q"],
        )
    )
    _errorbars = (
        alt.Chart(_bdf)
        .mark_rule(strokeWidth=1.5)
        .encode(
            x=alt.X("Algorithm:N", sort=algo_color_domain),
            y=alt.Y("CI Low:Q"),
            y2=alt.Y2("CI High:Q"),
        )
    )
    _val_labels = (
        alt.Chart(_bdf)
        .mark_text(color="white", fontWeight="bold", fontSize=10, dy=12)
        .encode(
            x=alt.X("Algorithm:N", sort=algo_color_domain),
            y=alt.Y("Value:Q"),
            text=alt.Text("Value:Q", format=".2f"),
        )
    )
    _chart = (
        (_bars + _errorbars + _val_labels)
        .properties(
            width=150, height=280, title="Cross-Environment Final Performance (95% CI)"
        )
        .facet(column=alt.Column("Metric:N", sort=_labels))
    )

    mo.md("## Cross-Environment Metrics")
    _chart
    return


@app.cell
def _(
    ALGO_COLOR_SCALE,
    ALGO_LABELS,
    algo_eval_results,
    algo_selection,
    all_algos,
    alt,
    mo,
    np,
    pd,
):
    # --- Performance profile: curves + CI bands ---
    _algos_with_pp = [
        a
        for a in all_algos
        if algo_eval_results[a].get("cross_environment", {}).get("performance_profile")
    ]
    mo.stop(
        not _algos_with_pp,
        mo.md("*Performance profile skipped (need >= 2 environments).*"),
    )

    _pp_records = []
    for _algo in _algos_with_pp:
        _pp = algo_eval_results[_algo]["cross_environment"]["performance_profile"]
        _tau = np.array(_pp["tau"])
        _vals = np.array(_pp["values"])
        _ci_low = np.array(_pp["ci_low"])
        _ci_high = np.array(_pp["ci_high"])
        for _k in range(len(_tau)):
            _pp_records.append(
                {
                    "Threshold": float(_tau[_k]),
                    "Fraction": float(_vals[_k]),
                    "CI Low": float(_ci_low[_k]),
                    "CI High": float(_ci_high[_k]),
                    "Algorithm": ALGO_LABELS[_algo],
                }
            )

    _pdf = pd.DataFrame(_pp_records)
    _sel = algo_selection()
    _line = (
        alt.Chart(_pdf)
        .mark_line()
        .encode(
            x=alt.X("Threshold:Q", title="Normalized Score Threshold (τ)"),
            y=alt.Y(
                "Fraction:Q",
                title="Fraction of Runs ≥ τ",
                scale=alt.Scale(domain=[0, 1.05]),
            ),
            color=alt.Color("Algorithm:N", scale=ALGO_COLOR_SCALE),
            opacity=alt.condition(_sel, alt.value(1), alt.value(0.15)),
            tooltip=["Algorithm:N", "Threshold:Q", "Fraction:Q"],
        )
        .add_params(_sel)
    )
    _band = (
        alt.Chart(_pdf)
        .mark_area(opacity=0.1)
        .encode(
            x="Threshold:Q",
            y="CI Low:Q",
            y2="CI High:Q",
            color=alt.Color("Algorithm:N", scale=ALGO_COLOR_SCALE),
        )
    )
    _chart = (
        (_band + _line)
        .properties(width=500, height=300, title="Performance Profile")
        .interactive()
    )
    _chart
    return


@app.cell
def _(
    ALGO_COLOR_SCALE,
    ALGO_LABELS,
    algo_color_domain,
    algo_eval_results,
    all_algos,
    alt,
    mo,
    pd,
):
    # --- Optimality gap: horizontal bars + CI ---
    _algos_with_og = [
        a
        for a in all_algos
        if "optimality_gap" in algo_eval_results[a].get("cross_environment", {})
    ]
    mo.stop(
        not _algos_with_og,
        mo.md("*Optimality gap skipped (need cross-environment metrics).*"),
    )

    _og_records = []
    for _algo in _algos_with_og:
        _og = algo_eval_results[_algo]["cross_environment"]["optimality_gap"]
        _og_records.append(
            {
                "Algorithm": ALGO_LABELS[_algo],
                "Optimality Gap": _og["point"],
                "CI Low": _og["ci_low"],
                "CI High": _og["ci_high"],
            }
        )

    _odf = pd.DataFrame(_og_records)
    _bars = (
        alt.Chart(_odf)
        .mark_bar()
        .encode(
            y=alt.Y("Algorithm:N", sort=algo_color_domain),
            x=alt.X("Optimality Gap:Q"),
            color=alt.Color("Algorithm:N", scale=ALGO_COLOR_SCALE),
            tooltip=["Algorithm:N", "Optimality Gap:Q", "CI Low:Q", "CI High:Q"],
        )
    )
    _errorbars = (
        alt.Chart(_odf)
        .mark_rule(strokeWidth=1.5)
        .encode(
            y=alt.Y("Algorithm:N", sort=algo_color_domain),
            x=alt.X("CI Low:Q"),
            x2=alt.X2("CI High:Q"),
        )
    )
    _val_labels = (
        alt.Chart(_odf)
        .mark_text(align="left", dx=5, fontSize=10)
        .encode(
            y=alt.Y("Algorithm:N", sort=algo_color_domain),
            x=alt.X("CI High:Q"),
            text=alt.Text("Optimality Gap:Q", format=".3f"),
        )
    )
    _chart = (_bars + _errorbars + _val_labels).properties(
        width=400, height=200, title="Optimality Gap (lower is better) — 95% CI"
    )
    _chart
    return


@app.cell
def _(ALGO_LABELS, all_algos, alt, mo, np, pairwise_poi, pd):
    # --- POI heatmap + markdown table ---
    mo.stop(
        not pairwise_poi,
        mo.md("*Pairwise P(X>Y) skipped (run `evaluate.py --pairwise-only` first).*"),
    )

    _algos_sorted = sorted(all_algos)
    _n = len(_algos_sorted)
    _matrix = np.full((_n, _n), 0.5)
    for _pair, _val in pairwise_poi.items():
        _parts = _pair.split("_vs_")
        if len(_parts) != 2:
            continue
        _a, _b = _parts
        if _a in _algos_sorted and _b in _algos_sorted:
            _i = _algos_sorted.index(_a)
            _j = _algos_sorted.index(_b)
            _matrix[_i, _j] = _val
            _matrix[_j, _i] = 1 - _val

    _labels = [ALGO_LABELS[a] for a in _algos_sorted]

    _poi_records = []
    for _i in range(_n):
        for _j in range(_n):
            _poi_records.append(
                {
                    "Row (X)": _labels[_i],
                    "Column (Y)": _labels[_j],
                    "P(X > Y)": float(_matrix[_i, _j]),
                }
            )
    _poi_df = pd.DataFrame(_poi_records)

    _rect = (
        alt.Chart(_poi_df)
        .mark_rect()
        .encode(
            x=alt.X("Column (Y):N", sort=_labels),
            y=alt.Y("Row (X):N", sort=_labels),
            color=alt.Color(
                "P(X > Y):Q",
                scale=alt.Scale(scheme="redblue", domain=[0, 1]),
            ),
            tooltip=["Row (X):N", "Column (Y):N", "P(X > Y):Q"],
        )
        .properties(width=300, height=300, title="Probability of Improvement")
    )
    _text = (
        alt.Chart(_poi_df)
        .mark_text(fontSize=14, fontWeight="bold")
        .encode(
            x=alt.X("Column (Y):N", sort=_labels),
            y=alt.Y("Row (X):N", sort=_labels),
            text=alt.Text("P(X > Y):Q", format=".2f"),
            color=alt.condition(
                'abs(datum["P(X > Y)"] - 0.5) > 0.25',
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )

    # Markdown table
    _tbl = "| |" + "".join(f" {_l} |" for _l in _labels) + "\n"
    _tbl += "|---|" + "".join("---|" for _ in _labels) + "\n"
    for _i in range(_n):
        _tbl += f"| **{_labels[_i]}** |"
        for _j in range(_n):
            _v = _matrix[_i, _j]
            _bold = "**" if abs(_v - 0.5) > 0.25 else ""
            _tbl += f" {_bold}{_v:.2f}{_bold} |"
        _tbl += "\n"

    mo.vstack(
        [
            _rect + _text,
            mo.md(
                f"### Pairwise P(X > Y) Table\n\n{_tbl}\n\n"
                "*Values > 0.5 mean the row algorithm is more likely to outperform the column algorithm.*"
            ),
        ]
    )
    return


@app.cell
def _(
    ALGO_LABELS,
    algo_color_domain,
    algo_color_range,
    algo_eval_results,
    all_algos,
    alt,
    env_configs,
    env_slugs,
    mo,
    pd,
):
    # --- Timing analysis: training time bars ---
    _time_records = []
    for _algo in all_algos:
        _timing = algo_eval_results[_algo].get("timing", {}).get("training", {})
        for _slug in env_slugs:
            _t = _timing.get(_slug, {}).get("total_seconds", 0)
            _time_records.append(
                {
                    "Algorithm": ALGO_LABELS[_algo],
                    "Environment": env_configs[_slug]["environment"],
                    "Training Time (s)": _t,
                    "Hours": f"{_t / 3600:.1f}h" if _t > 0 else "N/A",
                }
            )

    _tdf = pd.DataFrame(_time_records)
    _bar = (
        alt.Chart(_tdf)
        .mark_bar()
        .encode(
            x=alt.X("Algorithm:N", sort=algo_color_domain),
            y=alt.Y("Training Time (s):Q"),
            color=alt.Color(
                "Algorithm:N",
                scale=alt.Scale(domain=algo_color_domain, range=algo_color_range),
            ),
            tooltip=["Algorithm:N", "Environment:N", "Training Time (s):Q", "Hours:N"],
        )
        .properties(width=150, height=250, title="Training Time per Environment")
        .facet(column=alt.Column("Environment:N"))
    )

    mo.md("## Timing")
    _bar
    return


@app.cell
def _(algo_eval_results, all_algos, env_slugs, mo):
    # --- Per-environment metrics tables ---
    _tables = ""
    for _slug in env_slugs:
        _header = "| Metric |" + "".join(f" {a.upper()} |" for a in all_algos) + "\n"
        _sep = "|--------|" + "".join("--------|" for _ in all_algos) + "\n"

        _metric_defs = [
            ("IQM", "final_iqm"),
            ("Mean", "final_mean"),
            ("Median", "final_median"),
        ]
        _reliability_defs = [
            ("IQR (raw)", "iqr"),
            ("CVaR 0.1 (raw)", "cvar_01"),
            ("Min (raw)", "min_score"),
            ("Max (raw)", "max_score"),
        ]

        _tbl = _header + _sep
        for _label, _key in _metric_defs:
            _row = f"| {_label} |"
            for _algo in all_algos:
                _pe = algo_eval_results[_algo]["per_environment"].get(_slug)
                if _pe:
                    _m = _pe[_key]
                    _row += f" {_m['point']:.4f} [{_m['ci_low']:.4f}, {_m['ci_high']:.4f}] |"
                else:
                    _row += " --- |"
            _tbl += _row + "\n"

        for _label, _key in _reliability_defs:
            _row = f"| {_label} |"
            for _algo in all_algos:
                _pe = algo_eval_results[_algo]["per_environment"].get(_slug)
                if _pe:
                    _row += f" {_pe['reliability'][_key]:.1f} |"
                else:
                    _row += " --- |"
            _tbl += _row + "\n"

        _row = "| Sample Eff. AUC |"
        for _algo in all_algos:
            _pe = algo_eval_results[_algo]["per_environment"].get(_slug)
            if _pe:
                _row += f" {_pe['sample_efficiency_auc']:.4f} |"
            else:
                _row += " --- |"
        _tbl += _row + "\n"

        _tables += f"\n### `{_slug}`\n\n{_tbl}\n"

    mo.md(f"""
    ## Per-Environment Metrics

    {_tables}

    *Normalized: (raw - random) / (max - random), per Agarwal et al. (2021). Bootstrap CIs use 50,000 resamples.*
    """)
    return


@app.cell
def _(algo_eval_results, all_algos, env_slugs, mo):
    # --- Cross-env aggregate metrics + summary rankings ---
    _algos_with_ce = [
        a for a in all_algos if algo_eval_results[a].get("cross_environment")
    ]
    mo.stop(
        not _algos_with_ce,
        mo.md(
            "*Cross-environment aggregate metrics skipped (need >= 2 environments).*"
        ),
    )

    _header = "| Metric |" + "".join(f" {a.upper()} |" for a in _algos_with_ce) + "\n"
    _sep = "|--------|" + "".join("--------|" for _ in _algos_with_ce) + "\n"

    _metric_defs = [
        ("IQM", "iqm"),
        ("Mean", "mean"),
        ("Median", "median"),
        ("Optimality Gap", "optimality_gap"),
    ]

    _tbl = _header + _sep
    for _label, _key in _metric_defs:
        _row = f"| {_label} |"
        for _algo in _algos_with_ce:
            _m = algo_eval_results[_algo]["cross_environment"][_key]
            _row += f" {_m['point']:.4f} [{_m['ci_low']:.4f}, {_m['ci_high']:.4f}] |"
        _tbl += _row + "\n"

    # Summary ranking
    _rows = []
    for _algo in _algos_with_ce:
        _ce = algo_eval_results[_algo]["cross_environment"]
        _iqm = _ce.get("iqm", {}).get("point", 0)
        _og = _ce.get("optimality_gap", {}).get("point", 1)
        _timing = algo_eval_results[_algo].get("timing", {}).get("training", {})
        _total_time = sum(
            _timing.get(_slug, {}).get("total_seconds", 0) for _slug in env_slugs
        )
        _rows.append(
            {"algo": _algo.upper(), "iqm": _iqm, "og": _og, "time": _total_time}
        )

    _rows.sort(key=lambda r: r["iqm"], reverse=True)

    _rank_tbl = (
        "| Rank | Algorithm | Cross-Env IQM | Optimality Gap | Training Time (s) |\n"
    )
    _rank_tbl += (
        "|------|-----------|---------------|----------------|-------------------|\n"
    )
    for _i, _r in enumerate(_rows):
        _rank_tbl += (
            f"| {_i + 1} | {_r['algo']} | {_r['iqm']:.4f} | {_r['og']:.4f} "
            f"| {_r['time']:.1f} |\n"
        )

    _first = _algos_with_ce[0]
    _n_envs = len(algo_eval_results[_first]["environments"])
    _n_seeds = algo_eval_results[_first]["n_seeds"]
    mo.md(f"""
    ## Cross-Environment Aggregate Metrics

    Score matrix shape: ({_n_seeds}, {_n_envs}) — {_n_seeds * _n_envs} data points.

    {_tbl}

    ### Summary Rankings

    {_rank_tbl}

    *Ranked by cross-environment IQM (higher is better). Optimality gap: lower is better. Bootstrap CIs use 50,000 resamples.*
    """)
    return


@app.cell
def _(mo):
    # --- Reproducibility notes ---
    mo.md("""
    ## Reproducibility Notes

    - **n_envs mismatch:** DQN × Pong seeds 1–7 were trained with `n_envs=8`,
      remaining runs use `n_envs=16`. This affects wall-clock time but not
      per-step learning dynamics (same total timesteps).
    - **GPU non-determinism:** CUDA operations are inherently non-deterministic.
      Results are statistically valid across seeds but not bit-for-bit reproducible.
    - **DQN training timing:** `N/A` — DQN training was started before timing
      instrumentation was added to `train.py`.
    - **Evaluation fix:** `fresh_evaluate()` now uses SB3's `evaluate_policy`
      to read true episode returns from the Monitor wrapper, bypassing
      `ClipRewardEnv` and `EpisodicLifeEnv`. Prior Breakout scores were
      ~10× underreported due to reward clipping.
    """)
    return


if __name__ == "__main__":
    app.run()
