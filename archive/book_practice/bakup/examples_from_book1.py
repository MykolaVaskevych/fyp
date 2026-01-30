import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo  # for this notebook
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Example 6.1 (ride home MV vs TD)""")
    return


@app.cell
def _():
    # Core Imports
    import numpy as np
    import pandas as pd
    import polars as pl
    import altair as alt

    # RL parameters
    gamma = 1.0  # no discounting in the driving-home example
    alpha = 0.5  # step-size parameter
    return alpha, alt, gamma, np, pd, pl


@app.cell
def _(alpha, gamma, np):
    # Example 6.1 data (from the book)
    # States (descriptions)
    states = [
        "leaving office (Fri 6pm)",
        "reach car (raining)",
        "exiting highway",
        "secondary road (behind truck)",
        "entering home street",
        "arrive home",
    ]

    # Elapsed times (minutes) at each state
    elapsed = np.array([0, 5, 20, 30, 40, 43], dtype=float)

    # Initial value estimates (predicted time-to-go at each state)
    V_init = np.array([30, 35, 15, 10, 3, 0], dtype=float)

    # Predicted total time = elapsed + predicted time-to-go
    pred_total = elapsed + V_init

    # Rewards are the time differences between successive states
    rewards = np.diff(elapsed)  # [5, 15, 10, 10, 3]

    print("✅ Driving-home scenario loaded")
    print(f"\nScenario: {len(states)} states, γ={gamma}, α={alpha}")
    print(f"Rewards (elapsed times between states): {rewards}")
    return V_init, elapsed, pred_total, rewards, states


@app.cell
def _(V_init, elapsed, mo, pl, pred_total, states):
    # Create initial data table
    df_states = pl.DataFrame(
        {
            "State": states,
            "Elapsed Time (min)": elapsed,
            "Predicted Time to Go": V_init,
            "Predicted Total Time": pred_total,
        }
    )

    mo.md(f"""
    ### Initial State Values

    This table shows your predictions at each stage of the drive home:
    """)

    df_states
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Monte Carlo Method

    **Update Rule:** `V(S) ← V(S) + α[G - V(S)]`

    - Waits until episode ends to get actual return G
    - G = actual time-to-go from each state
    - Updates all states based on **final outcome**
    """
    )
    return


@app.cell
def _(V_init, alpha, elapsed, mo, pl, states):
    # Monte Carlo Analysis
    # Actual return from each state = total time to reach home - elapsed time
    actual_returns = 43 - elapsed  # actual time-to-go from each state

    # MC error: difference between actual return and predicted value
    mc_errors = actual_returns - V_init

    # MC updates (what would be added to V with α=1)
    mc_updates_full = mc_errors.copy()
    mc_updates_alpha = alpha * mc_errors

    # Apply MC updates
    V_mc = V_init.copy()
    V_mc = V_mc + mc_updates_alpha
    V_mc[-1] = 0  # terminal state always 0

    df_mc = pl.DataFrame(
        {
            "State": states,
            "V_init": V_init,
            "Actual Return (G)": actual_returns,
            "MC Error (G - V)": mc_errors,
            "MC Update (α=0.5)": mc_updates_alpha,
            "V_after_MC": V_mc,
        }
    )

    mo.md("### Monte Carlo Updates")
    df_mc
    return (mc_errors,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## TD(0) Method

    **Update Rule:** `V(S) ← V(S) + α[R + γV(S') - V(S)]`

    - Updates immediately at each transition
    - Uses bootstrapped estimate from next state
    - Updates are based on **temporal differences** between successive predictions
    """
    )
    return


@app.cell
def _(V_init, alpha, gamma, mo, np, pl, rewards, states):
    # TD(0) Analysis - FIX: use _i instead of i
    td_errors = []
    td_updates = []

    for _i in range(len(states) - 1):
        R = rewards[_i]  # reward for transition
        V_s = V_init[_i]  # current state value
        V_s_next = V_init[_i + 1]  # next state value

        # TD error
        delta = R + gamma * V_s_next - V_s
        td_errors.append(delta)

        # TD update
        td_updates.append(alpha * delta)

    td_errors = np.array(td_errors)
    td_updates = np.array(td_updates)

    # Apply TD updates sequentially
    V_td = V_init.copy()
    for _j in range(len(states) - 1):
        V_td[_j] = V_td[_j] + td_updates[_j]

    df_td = pl.DataFrame(
        {
            "Transition": [
                f"{states[_i][:20]}... → {states[_i + 1][:20]}..."
                for _i in range(len(states) - 1)
            ],
            "Reward (R)": rewards,
            "V(S)": V_init[:-1],
            "V(S')": V_init[1:],
            "TD Error (δ)": td_errors,
            "TD Update (α=0.5)": td_updates,
        }
    )

    mo.md("### TD(0) Updates")
    df_td
    return (td_errors,)


@app.cell
def _(V_init, alpha, elapsed, mo, pl, states):
    # Monte Carlo Analysis - FIX: use _i instead of i
    actual_returns = 43 - elapsed
    mc_errors = actual_returns - V_init
    mc_updates_alpha = alpha * mc_errors

    # Apply MC updates
    V_mc = V_init.copy()
    V_mc = V_mc + mc_updates_alpha
    V_mc[-1] = 0  # terminal state always 0

    df_mc = pl.DataFrame(
        {
            "State": states,
            "V_init": V_init,
            "Actual Return (G)": actual_returns,
            "MC Error (G - V)": mc_errors,
            "MC Update (α=0.5)": mc_updates_alpha,
            "V_after_MC": V_mc,
        }
    )

    mo.md("### Monte Carlo Updates")
    df_mc
    return (mc_errors,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Figure 6.1: Visualization of Updates

    Left chart shows **Monte Carlo**: arrows point from each prediction to the actual outcome (43 minutes).

    Right chart shows **TD(0)**: arrows point from each prediction to the next prediction.
    """
    )
    return


@app.cell
def _(alt, elapsed, mc_errors, pd, pred_total, states):
    # MC Chart - FIX: use _i instead of i
    chart_data = pd.DataFrame(
        {
            "elapsed": elapsed,
            "predicted_total": pred_total,
            "state_idx": range(len(states)),
            "state": states,
        }
    )

    mc_base = (
        alt.Chart(chart_data)
        .mark_line(
            point=alt.OverlayMarkDef(size=80, filled=True),
            color="steelblue",
            strokeWidth=3,
        )
        .encode(
            x=alt.X(
                "elapsed:Q",
                title="Elapsed Time (minutes)",
                scale=alt.Scale(domain=[0, 45]),
            ),
            y=alt.Y(
                "predicted_total:Q",
                title="Predicted Total Time (minutes)",
                scale=alt.Scale(domain=[28, 52]),
            ),
            tooltip=[
                alt.Tooltip("state:N", title="State"),
                alt.Tooltip("elapsed:Q", title="Elapsed Time"),
                alt.Tooltip("predicted_total:Q", title="Predicted Total"),
            ],
        )
    )

    actual_line_mc = (
        alt.Chart(pd.DataFrame({"y": [43]}))
        .mark_rule(strokeDash=[8, 4], color="darkred", strokeWidth=2)
        .encode(y="y:Q")
    )

    # FIX: use _i for loop variable
    mc_arrow_data = []
    for _i in range(len(states) - 1):
        mc_arrow_data.append(
            {
                "x": elapsed[_i],
                "y0": pred_total[_i],
                "y1": 43,
                "error": mc_errors[_i],
            }
        )

    mc_arrows = (
        alt.Chart(pd.DataFrame(mc_arrow_data))
        .mark_rule(color="red", strokeWidth=3, opacity=0.6)
        .encode(
            x="x:Q",
            y="y0:Q",
            y2="y1:Q",
            tooltip=[
                alt.Tooltip("x:Q", title="Elapsed Time"),
                alt.Tooltip("error:Q", title="MC Error", format=".1f"),
            ],
        )
    )

    mc_chart = (mc_base + actual_line_mc + mc_arrows).properties(
        title={
            "text": "Monte Carlo Method",
            "subtitle": "Updates based on final outcome (offline)",
        },
        width=380,
        height=320,
    )

    mc_chart
    return chart_data, mc_chart


@app.cell
def _(alt, chart_data, elapsed, pd, pred_total, states, td_errors):
    # TD Chart - FIX: use _i instead of i
    td_base = (
        alt.Chart(chart_data)
        .mark_line(
            point=alt.OverlayMarkDef(size=80, filled=True),
            color="steelblue",
            strokeWidth=3,
        )
        .encode(
            x=alt.X(
                "elapsed:Q",
                title="Elapsed Time (minutes)",
                scale=alt.Scale(domain=[0, 45]),
            ),
            y=alt.Y(
                "predicted_total:Q",
                title="Predicted Total Time (minutes)",
                scale=alt.Scale(domain=[28, 52]),
            ),
            tooltip=[
                alt.Tooltip("state:N", title="State"),
                alt.Tooltip("elapsed:Q", title="Elapsed Time"),
                alt.Tooltip("predicted_total:Q", title="Predicted Total"),
            ],
        )
    )

    # FIX: use _i for loop variable
    td_arrow_data = []
    for _i in range(len(states) - 1):
        td_arrow_data.append(
            {
                "x": elapsed[_i],
                "y0": pred_total[_i],
                "y1": pred_total[_i + 1],
                "td_error": td_errors[_i],
            }
        )

    td_arrows = (
        alt.Chart(pd.DataFrame(td_arrow_data))
        .mark_rule(color="darkorange", strokeWidth=3, opacity=0.7)
        .encode(
            x="x:Q",
            y="y0:Q",
            y2="y1:Q",
            tooltip=[
                alt.Tooltip("x:Q", title="Elapsed Time"),
                alt.Tooltip("td_error:Q", title="TD Error", format=".1f"),
            ],
        )
    )

    td_chart = (td_base + td_arrows).properties(
        title={
            "text": "TD(0) Method",
            "subtitle": "Updates based on next prediction (online)",
        },
        width=380,
        height=320,
    )

    td_chart
    return (td_chart,)


@app.cell
def _(mc_chart, mo, td_chart):
    # Side-by-side comparison
    mo.hstack([mc_chart, td_chart], gap=2)
    return


if __name__ == "__main__":
    app.run()
