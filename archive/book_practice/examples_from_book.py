import marimo

__generated_with = "0.17.6"
app = marimo.App(
    width="columns",
    app_title="notebook for JJ",
    auto_download=["ipynb"],
)


@app.cell
def _(batch_mc_example64, batch_td_example64, example64_episodes):
    print("Running Batch TD(0)...")
    V_td_ex64 = batch_td_example64(example64_episodes, alpha=0.01)

    print("\nRunning Batch MC...")
    V_mc_ex64 = batch_mc_example64(example64_episodes, alpha=0.01)

    print("\n" + "=" * 50)
    print("Results:")
    print("=" * 50)
    print(f"Batch TD:  V(A)={V_td_ex64[0]:.4f}, V(B)={V_td_ex64[1]:.4f}")
    print(f"Batch MC:  V(A)={V_mc_ex64[0]:.4f}, V(B)={V_mc_ex64[1]:.4f}")
    print(f"Expected:  V(B) ≈ 0.75 (6/8 episodes)")
    return V_mc_ex64, V_td_ex64


@app.cell
def _():


    return


@app.cell
def _(mo):
    mo.md(r"""
    # imports
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import altair as alt
    import pandas as pd
    import matplotlib.pyplot as plt

    # i decided to be fancy and try pydantic for types, no reason, just trying to learn it as well
    from typing import List, Tuple, Optional
    from pydantic import (
        BaseModel,
        Field,
        field_validator,
        ConfigDict,
        computed_field,
    )
    from numpy.typing import NDArray
    from matplotlib.figure import Figure
    return (
        BaseModel,
        ConfigDict,
        Field,
        List,
        NDArray,
        Tuple,
        alt,
        computed_field,
        field_validator,
        mo,
        np,
        pd,
        pl,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Example 6.1 (ride home MV vs TD)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 6.1: Driving Home
    """)
    return


@app.cell
def _(np):
    states = [
        "leaving office",
        "reach car, raining",
        "exiting highway",
        "secondary road, behind truck",
        "entering home street",
        "arrive home",
    ]

    elapsed_time = np.array([0, 5, 20, 30, 40, 43])

    predicted_time_to_go = np.array([30, 35, 15, 10, 3, 0])

    predicted_total_time = elapsed_time + predicted_time_to_go

    actual_total_time = 43
    gamma = 1.0
    return (
        actual_total_time,
        elapsed_time,
        predicted_time_to_go,
        predicted_total_time,
        states,
    )


@app.cell
def _(mo):
    alpha_slider = mo.ui.slider(0.0, 1.0, value=1.0, step=0.1, label="Step size α")
    alpha_slider
    return (alpha_slider,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Monte Carlo Update Rule

    Monte Carlo methods wait until the end of an episode to update value estimates based on the actual return:

    $$V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$$

    where:
    - $V(S_t)$ is the estimated value of state $S_t$
    - $\alpha$ is the step size (learning rate)
    - $G_t$ is the actual return (total reward) from time $t$ onwards
    - $G_t - V(S_t)$ is the MC error
    """)
    return


@app.cell
def _(actual_total_time, alpha_slider, elapsed_time, np, predicted_time_to_go):
    def compute_monte_carlo_updates(predictions, actual, elapsed, alpha):
        updates = []
        n_states = len(predictions)

        for i in range(n_states - 1):
            current_prediction = predictions[i] + elapsed[i]
            mc_error = actual - current_prediction
            update = alpha * mc_error
            updates.append(update)

        updates.append(0)

        return np.array(updates)


    mc_updates = compute_monte_carlo_updates(
        predicted_time_to_go, actual_total_time, elapsed_time, alpha_slider.value
    )
    return (mc_updates,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Temporal-Difference (TD) Update Rule

    TD methods update estimates based on other estimates, without waiting for the final outcome:

    $$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

    where:
    - $V(S_t)$ is the estimated value of state $S_t$
    - $\alpha$ is the step size (learning rate)
    - $R_{t+1}$ is the reward received after transitioning to the next state
    - $\gamma$ is the discount factor
    - $V(S_{t+1})$ is the estimated value of the next state
    - $R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the TD error (δ)
    """)
    return


@app.cell
def _(alpha_slider, elapsed_time, np, predicted_time_to_go):
    def compute_td_updates(predictions, elapsed, alpha, gamma=1.0):
        updates = []
        n_states = len(predictions)

        for i in range(n_states - 1):
            current_estimate = predictions[i]
            next_estimate = predictions[i + 1]

            reward = elapsed[i + 1] - elapsed[i]

            td_error = reward + gamma * next_estimate - current_estimate
            update = alpha * td_error
            updates.append(update)

        updates.append(0)

        return np.array(updates)


    td_updates = compute_td_updates(
        predicted_time_to_go, elapsed_time, alpha_slider.value
    )
    return (td_updates,)


@app.cell(hide_code=True)
def _(
    elapsed_time,
    mc_updates,
    pl,
    predicted_time_to_go,
    predicted_total_time,
    states,
    td_updates,
):
    df_data = pl.DataFrame(
        {
            "State": states,
            "Elapsed_Time": elapsed_time,
            "Predicted_Time_to_Go": predicted_time_to_go,
            "Predicted_Total_Time": predicted_total_time,
            "MC_Update": mc_updates,
            "TD_Update": td_updates,
            "State_Index": list(range(len(states))),
        }
    )

    df_data
    return (df_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 6.1: Visualization of Updates
    """)
    return


@app.cell
def _(alpha_slider, alt, df_data, pd, pl):
    # Define the true total time value (from Sutton & Barto Example 6.1)
    true_total_time = 43.0

    # Create a horizontal dashed line at that y position
    true_line = (
        alt.Chart(pd.DataFrame({"y": [true_total_time]}))
        .mark_rule(strokeDash=[6, 4], color="black", opacity=0.6)
        .encode(y="y:Q")
    )

    # === MONTE CARLO CHART ===
    base_mc = (
        alt.Chart(df_data.to_pandas())
        .mark_line(point=True, color="steelblue")
        .encode(
            x=alt.X(
                "Elapsed_Time:Q",
                title="Elapsed Time (minutes)",
                scale=alt.Scale(domain=[0, 45]),
            ),
            y=alt.Y(
                "Predicted_Total_Time:Q",
                title="Predicted Total Time (minutes)",
                scale=alt.Scale(domain=[25, 55]),
            ),
            tooltip=["State:N", "Elapsed_Time:Q", "Predicted_Total_Time:Q"],
        )
    )

    # --- build MC arrows as you had ---
    arrows_data_mc = []
    for i in range(len(df_data) - 2):
        row = df_data.row(i, named=True)
        arrows_data_mc.append(
            {
                "x": row["Elapsed_Time"],
                "y": row["Predicted_Total_Time"],
                "x2": row["Elapsed_Time"],
                "y2": row["Predicted_Total_Time"] + row["MC_Update"],
            }
        )

    if arrows_data_mc:
        arrows_mc = (
            alt.Chart(pl.DataFrame(arrows_data_mc).to_pandas())
            .mark_rule(strokeWidth=2, color="red", opacity=0.7)
            .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
        )

        arrow_points_mc = (
            alt.Chart(pl.DataFrame(arrows_data_mc).to_pandas())
            .mark_point(size=100, filled=True, color="red", shape="triangle-up")
            .encode(x="x2:Q", y="y2:Q")
        )

        mc_chart = (base_mc + arrows_mc + arrow_points_mc + true_line).properties(
            title=f"Monte Carlo (α={alpha_slider.value})", width=350, height=300
        )
    else:
        mc_chart = (base_mc + true_line).properties(
            title=f"Monte Carlo (α={alpha_slider.value})", width=350, height=300
        )

    # === TD(0) CHART ===
    base_td = (
        alt.Chart(df_data.to_pandas())
        .mark_line(point=True, color="steelblue")
        .encode(
            x=alt.X(
                "Elapsed_Time:Q",
                title="Elapsed Time (minutes)",
                scale=alt.Scale(domain=[0, 45]),
            ),
            y=alt.Y(
                "Predicted_Total_Time:Q",
                title="Predicted Total Time (minutes)",
                scale=alt.Scale(domain=[25, 55]),
            ),
            tooltip=["State:N", "Elapsed_Time:Q", "Predicted_Total_Time:Q"],
        )
    )
    # --- Build vertical TD arrows (fix) ---
    arrows_data_td = []
    for i in range(len(df_data) - 1):
        row = df_data.row(i, named=True)
        # next_row = df_data.row(i + 1, named=True)   # not needed for x2
        x = row["Elapsed_Time"]
        y = row["Predicted_Total_Time"]
        # TD_Update should be the change in predicted total time for state t (could be alpha * TD_error)
        # y2 is the new predicted total time for the same state (vertical arrow)
        y2 = row["Predicted_Total_Time"] + row["TD_Update"]
        arrows_data_td.append({"x": x, "y": y, "x2": x, "y2": y2})

    if arrows_data_td:
        arrows_td = (
            alt.Chart(pl.DataFrame(arrows_data_td).to_pandas())
            .mark_rule(strokeWidth=2, color="red", opacity=0.7)
            .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
        )

        arrow_points_td = (
            alt.Chart(pl.DataFrame(arrows_data_td).to_pandas())
            .mark_point(size=100, filled=True, color="red", shape="triangle-up")
            .encode(x="x2:Q", y="y2:Q")
        )

        td_chart = (base_td + arrows_td + arrow_points_td + true_line).properties(
            title=f"TD(0) (α={alpha_slider.value})", width=350, height=300
        )
    else:
        td_chart = (base_td + true_line).properties(
            title=f"TD(0) (α={alpha_slider.value})", width=350, height=300
        )


    # --- Combine ---
    combined_chart = alt.hconcat(mc_chart, td_chart)
    combined_chart.interactive()
    return


@app.cell
def _(alpha_slider):
    alpha_slider
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Example 6.2: Random Walk
    """)
    return


@app.cell
def _():
    ## set random seed for reproducibility
    # np.random.seed(42)  # why 42 I dunno, but it seems that all use it
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## configs for types (ignore it)
    """)
    return


@app.cell
def _(
    BaseModel,
    ConfigDict,
    Field,
    List,
    NDArray,
    computed_field,
    field_validator,
    np,
):
    class MRPConfig(BaseModel):
        """Configuration for MRP environment with computed fields."""

        # Primary configuration - only what user needs to specify
        n_states: int = Field(default=5, gt=0, description="Number of states")
        gamma: float = Field(
            default=1.0, ge=0.0, le=1.0, description="Discount factor"
        )

        # Allow arbitrary types like numpy arrays
        model_config = ConfigDict(arbitrary_types_allowed=True)

        @computed_field  # Automatically computed from n_states
        @property
        def states(self) -> List[str]:
            """Generate state names dynamically (A, B, C, ...)."""
            return [chr(65 + i) for i in range(self.n_states)]  # 65 = ord('A')

        @computed_field
        @property
        def start_state(self) -> int:
            """Starting state is always the middle."""
            return self.n_states // 2

        @computed_field
        @property
        def left_terminal(self) -> int:
            """Left terminal is always first state (index 0)."""
            return 0

        @computed_field
        @property
        def right_terminal(self) -> int:
            """Right terminal is always last state."""
            return self.n_states - 1

        @computed_field
        @property
        def true_values(self) -> NDArray[np.float64]:
            """Compute true values: probability of reaching right terminal."""
            return np.linspace(0.0, 1.0, self.n_states)


    class AlgorithmConfig(BaseModel):
        """Configuration for learning algorithms."""

        # Learning rate (step-size parameter)
        alpha: float = Field(
            default=0.1, gt=0.0, le=1.0, description="Learning rate"
        )
        # Number of episodes to run
        n_episodes: int = Field(
            default=100, gt=0, description="Number of episodes"
        )
        # Number of independent runs for averaging
        n_runs: int = Field(
            default=100, gt=0, description="Number of runs to average"
        )

        @field_validator("alpha")
        @classmethod
        def validate_alpha(cls, v: float) -> float:
            """Ensure alpha is a valid learning rate."""
            if v <= 0 or v > 1:
                raise ValueError(f"Alpha must be in (0, 1], got {v}")
            return v


    class ExperimentConfig(BaseModel):
        """Complete experiment configuration."""

        # MRP configuration
        mrp: MRPConfig = Field(default_factory=MRPConfig)
        # Algorithm configuration
        algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
        # Alpha values to test - SEPARATE for TD and MC
        alpha_values_td: List[float] = Field(
            default=[0.05, 0.1, 0.15], description="TD learning rates to compare"
        )
        alpha_values_mc: List[float] = Field(
            default=[0.01, 0.02, 0.03, 0.04],
            description="MC learning rates to compare",
        )
        # Episodes to plot for single run visualization
        episodes_to_plot: List[int] = Field(
            default=[0, 1, 10, 100], description="Episode numbers to visualize"
        )
        # Random seed for reproducibility
        random_seed: int = Field(default=42, description="Random seed")

        @field_validator("alpha_values_td", "alpha_values_mc")
        @classmethod
        def validate_alphas(cls, v: List[float]) -> List[float]:
            """Ensure all alpha values are valid learning rates."""
            for alpha in v:
                if alpha <= 0 or alpha > 1:
                    raise ValueError(f"All alphas must be in (0, 1], got {alpha}")
            return v

        @field_validator("episodes_to_plot")
        @classmethod
        def validate_episodes(cls, v: List[int]) -> List[int]:
            """Ensure episode numbers are non-negative."""
            for ep in v:
                if ep < 0:
                    raise ValueError(
                        f"Episode numbers must be non-negative, got {ep}"
                    )
            return sorted(v)  # Return sorted list
    return AlgorithmConfig, ExperimentConfig, MRPConfig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Result models
    """)
    return


@app.cell
def _(BaseModel, ConfigDict, Field, List, NDArray, np):
    class EpisodeTransition(BaseModel):
        """single transition, step ?"""

        state: int = Field(ge=0, description="state index")
        reward: float = Field(description="rw after each action")
        model_config = ConfigDict(arbitrary_types_allowed=True)


    class ExperimentResult(BaseModel):
        """Results from a single experiment run."""

        method: str = Field(description="Method name TD or MC")
        alpha: float = Field(
            le=1.0, gt=0.0, description="step size / learning rate"
        )
        final_values: NDArray[np.float64] = Field(
            description="Final estimated values after all episodes"
        )
        value_history: List[NDArray[np.float64]] = Field(
            description="Value estimates over time"
        )
        rmse_history: List[float] = Field(
            description="RMSE over episodes (standard metric used to measure the accuracy of a model's predictions)"
        )
        model_config = ConfigDict(arbitrary_types_allowed=True)
    return EpisodeTransition, ExperimentResult


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Env class
    """)
    return


@app.cell
def _(MRPConfig, Tuple, np):
    class RandomWalk_ENV_MRP:
        """Markov Reward Process for Random Walk."""

        def __init__(self, config: MRPConfig):
            self.config = config
            self.n_states = config.n_states
            self.start_state = config.start_state
            self.current_state = config.start_state

        def reset(self):
            """Reset to start state."""
            self.current_state = self.start_state
            return self.current_state

        def step(self) -> Tuple[int, float, bool]:
            """Take a step left or right

            Returns:
                Tupple of:
                next_state (int): Index of the next state
                reward (float): Reward received after the step (1 if right terminal, 0 otherwise)
                done (bool): Whether a terminal state is reached
            """

            # if (self.current_state == self.config.left_terminal) or (
            #     self.current_state == self.config.right_terminal
            # ):
            #     raise ValueError(
            #         "Episode has terminated. Please reset the environment."
            #     )

            self.current_state += np.random.choice([-1, 1])
            __next_state: int = self.current_state

            done: bool = (self.current_state == self.config.left_terminal) or (
                self.current_state == self.config.right_terminal
            )

            reward: float = (
                1.0 if self.current_state == self.config.right_terminal else 0.0
            )
            return __next_state, reward, done
    return (RandomWalk_ENV_MRP,)


@app.cell
def _(EpisodeTransition, List, RandomWalk_ENV_MRP):
    # == Generate episode function ===, pass enviroment from cell above
    def generate_episode(env: RandomWalk_ENV_MRP) -> List[EpisodeTransition]:
        """Generate an episode of random walk transition

        and return list of transitions
        return: List[EpisodeTransition]

        where EpisodeTransition is a dataclass with fields:
            state: int
            reward: float

        Args:
            env: MRP environment instance

        Returns:
            List of episode transitions (state, reward pairs)
        """

        #  Reset env to starting state for new episode
        state: int = env.reset()
        # store episode trajectory as list of transitions
        episode: List[EpisodeTransition] = [
            EpisodeTransition(state=state, reward=0.0)  # init state has no reward
        ]

        done: bool = False

        while not done:
            next_state: int
            reward: float
            next_state, reward, done = env.step()

            # note that step to step history
            episode.append(EpisodeTransition(state=next_state, reward=reward))
            state = next_state

        return episode  # full history of it
    return (generate_episode,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RMSE Calculation ( (Root Mean Squared Error))
    """)
    return


@app.cell
def _(NDArray, np):
    def calculate_rmse(
        V: NDArray[np.float64],  # predicted values
        true_values: NDArray[np.float64],  # ground truth values (from MRP config)
    ) -> float:
        """
        Calculate Root Mean Squared Error between predicted and true values.

        we need that for tracking performance over episodes

          Returns:
            Root mean squared error (lower is better)
        """

        # square differences from each state
        squared_errors: NDArray[np.float64] = (V - true_values) ** 2
        # take mean across all states
        mean_squared_error: float = np.mean(squared_errors)

        rmse: float = np.sqrt(mean_squared_error)

        return rmse
    return (calculate_rmse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TD(0)


    note:
    (temporal difference algorithm, 0 means how far to look ahead, 0 means only next step)
    """)
    return


@app.cell
def _(
    AlgorithmConfig,
    ExperimentResult,
    List,
    NDArray,
    RandomWalk_ENV_MRP,
    calculate_rmse,
    np,
):
    def td_zero(
        env: RandomWalk_ENV_MRP,
        config: AlgorithmConfig,
        true_values: NDArray[np.float64],
    ) -> ExperimentResult:
        """
           Temporal Difference TD(0) learning algorithm.
        Updates value function after each step using bootstrapping.

        Formula: V(S) ← V(S) + α[R + γV(S') - V(S)]

        Args:
            env: MRP environment instance
            config: Algorithm configuration (alpha, n_episodes)
            true_values: Ground truth values for RMSE calculation

        Returns:
            Experiment result with final values and history
        """

        # Initialize value function to 0.5 for all states
        V: NDArray[np.float64] = np.ones(env.n_states) * 0.5
        # Store value function snapshot after each episode
        value_history: List[NDArray[np.float64]] = [V.copy()]
        # Store RMSE after each episode for performance tracking
        rmse_history: List[float] = [calculate_rmse(V, true_values)]

        # Run logic (aka training)
        for episode_num in range(config.n_episodes):
            # start from center
            state: int = env.reset()
            done: bool = False

            # until we are on edge (L/R):
            # Run episode until termination
            while not done:
                # Take one step: observe next state and reward
                next_state: int
                reward: float
                next_state, reward, done = env.step()

                # TD(0) Update Formula: V(S) ← V(S) + α[TD_target - V(S)]
                # where TD_target = R + γV(S') for non-terminal states
                #       TD_target = R          for terminal states (no future value)
                # so:
                # V(S) ← V(S) + α[TD_target - V(S)]
                #        ^      ^^^-part1-^       ^^^
                #        |      ||------part2-----|||
                #        |      |--------part3-----||
                #        |---------part4------------|

                # part1
                if done:
                    # Terminal state: target is just the immediate reward
                    td_target: float = reward
                else:
                    # Non-terminal: target is reward + discounted next state value
                    td_target: float = reward + env.config.gamma * V[next_state]

                # part2
                # Update current state's value toward the target
                td_error: float = td_target - V[state]

                # part4<----------part3
                V[state] += config.alpha * td_error

                # CRITICAL: Also update terminal state when reached
                if done:
                    # Terminal state value = reward received (no future)
                    # V(E) learns toward 1.0, V(A) learns toward 0.0
                    V[next_state] += config.alpha * (reward - V[next_state])

                # Move to next state for next iteration
                state = next_state

            # After episode ends, record the learned values and error
            value_history.append(V.copy())
            rmse_history.append(calculate_rmse(V, true_values))
        return ExperimentResult(
            method="TD",
            alpha=config.alpha,
            final_values=V,
            value_history=value_history,
            rmse_history=rmse_history,
        )
    return (td_zero,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Monte Carlo algo
    """)
    return


@app.cell
def _(
    AlgorithmConfig,
    EpisodeTransition,
    ExperimentResult,
    List,
    NDArray,
    RandomWalk_ENV_MRP,
    calculate_rmse,
    generate_episode,
    np,
):
    def monte_carlo_constant_alpha(
        env: RandomWalk_ENV_MRP,
        config: AlgorithmConfig,
        true_values: NDArray[np.float64],
    ) -> ExperimentResult:
        """
            Constant-α Monte Carlo learning algorithm.
        Updates value function after complete episode using actual returns.

        Formula: V(S_t) ← V(S_t) + α[G_t - V(S_t)]

        Args:
            env: MRP environment instance
            config: Algorithm configuration (alpha, n_episodes)
            true_values: Ground truth values for RMSE calculation

        Returns:
            Experiment result with final values and history"""
        # init state value function to 0.5 for all states
        V: NDArray[np.float64] = np.ones(env.n_states) * 0.5
        # store value function snapshot after each episode
        value_history: List[NDArray[np.float64]] = [V.copy()]
        # store RMSE after each episode for performance tracking
        rmse_history: List[float] = [calculate_rmse(V, true_values)]

        # run logic (aka training)
        for episode_num in range(config.n_episodes):
            episode: List[EpisodeTransition] = generate_episode(env)

            # Monte Carlo: Learn from COMPLETE episode returns
            # Walk backwards to calculate G_t (cumulative return from each state)
            # Formula: V(S_t) ← V(S_t) + α[G_t - V(S_t)]
            # where G_t = R_{t+1} + γ*R_{t+2} + γ²*R_{t+3} + ... (actual observed return)

            G: float = 0.0  # Cumulative return (starts at 0 from terminal state)

            # Iterate backwards through episode, skipping the terminal state
            # Why backwards? To accumulate returns: G_t = R_{t+1} + γ*G_{t+1}
            for transition, next_transition in zip(
                reversed(episode[:-1]),  # Current states: [S_0, S_1, ..., S_{T-1}]
                reversed(
                    episode[1:]
                ),  # Next transitions: [S_1, S_2, ..., S_T] (for rewards)
            ):
                # Step 1: Accumulate return backwards
                # G_t = R_{t+1} + γ*G_{t+1}
                #  ^       ^      ^    ^
                #  |       |      |    └─ Return from next time step (already calculated)
                #  |       |      └────── Discount factor
                #  |       └───────────── Immediate reward after leaving current state
                #  └───────────────────── Return from current state (what we're calculating)
                G = next_transition.reward + env.config.gamma * G

                # Step 2: MC Update Formula: V(S) ← V(S) + α[G - V(S) ]
                #                                    ^     ^ ^-part1^ ^ ^
                #                                    |     |--part2---| |
                #                                    |------part3-------|

                # part1: Calculate MC error (difference between actual return and estimate)
                mc_error: float = G - V[transition.state]

                # part3 ← part2 (update value toward actual observed return)
                V[transition.state] += config.alpha * mc_error

            # After episode ends, record the learned values and error
            value_history.append(V.copy())
            rmse_history.append(calculate_rmse(V, true_values))

        return ExperimentResult(
            method="MC",
            alpha=config.alpha,
            final_values=V,
            value_history=value_history,
            rmse_history=rmse_history,
        )
    return (monte_carlo_constant_alpha,)


@app.cell
def _(mo):
    alpha_slider_interactive = mo.ui.slider(
        0.01,
        0.5,
        value=0.1,
        step=0.01,
        label="Learning Rate (α) for Interactive Plot",
    )
    alpha_slider_interactive
    return (alpha_slider_interactive,)


@app.cell(hide_code=True)
def _(
    AlgorithmConfig,
    ExperimentConfig,
    MRPConfig,
    RandomWalk_ENV_MRP,
    alpha_slider_interactive,
    mo,
    monte_carlo_constant_alpha,
    np,
    pd,
    plt,
    td_zero,
):
    # ==================== SINGLE INTERACTIVE PLOT ====================

    # Get alpha value from slider
    _alpha_val = alpha_slider_interactive.value

    # Create fresh config and environments
    _mrp_cfg = MRPConfig(n_states=5, gamma=1.0)

    # Run TD(0)
    _env_td = RandomWalk_ENV_MRP(_mrp_cfg)
    _algo_cfg_td = AlgorithmConfig(alpha=_alpha_val, n_episodes=100)
    _td_res = td_zero(_env_td, _algo_cfg_td, _mrp_cfg.true_values)

    # Run MC
    _env_mc = RandomWalk_ENV_MRP(_mrp_cfg)
    _algo_cfg_mc = AlgorithmConfig(alpha=_alpha_val, n_episodes=100)
    _mc_res = monte_carlo_constant_alpha(
        _env_mc, _algo_cfg_mc, _mrp_cfg.true_values
    )

    # Debug print
    print(f"Episode 0 TD values: {_td_res.value_history[0]}")
    print(f"Episode 1 TD values: {_td_res.value_history[1]}")
    print(f"Episode 100 TD values: {_td_res.value_history[100]}")

    # Create 2 subplots
    _fig1, _axes1 = plt.subplots(1, 2, figsize=(15, 5))

    # LEFT: Value estimates over episodes
    _episodes_show = [0, 1, 10, 100]
    _colors_ep = ["purple", "orange", "green", "cyan"]

    for _idx, _ep in enumerate(_episodes_show):
        if _ep < len(_td_res.value_history):
            _axes1[0].plot(
                range(_mrp_cfg.n_states),
                _td_res.value_history[_ep],
                marker="o",
                markersize=8,
                color=_colors_ep[_idx],
                label=f"{_ep} episodes",
                linewidth=2,
            )

    # True values line
    _axes1[0].plot(
        range(_mrp_cfg.n_states),
        _mrp_cfg.true_values,
        "k--",
        linewidth=3,
        label="True values",
    )

    _axes1[0].set_xlabel("State", fontsize=12)
    _axes1[0].set_ylabel("Estimated Value", fontsize=12)
    _axes1[0].set_title(f"TD(0) Value Estimates (α={_alpha_val:.2f})", fontsize=14)
    _axes1[0].set_xticks(range(_mrp_cfg.n_states))
    _axes1[0].set_xticklabels(_mrp_cfg.states)
    _axes1[0].legend(fontsize=9)
    _axes1[0].grid(True, alpha=0.3)
    _axes1[0].set_ylim(-0.1, 1.1)

    # RIGHT: RMSE comparison
    _axes1[1].plot(
        range(len(_td_res.rmse_history)),
        _td_res.rmse_history,
        color="blue",
        linewidth=2,
        label=f"TD(0) α={_alpha_val:.2f}",
    )
    _axes1[1].plot(
        range(len(_mc_res.rmse_history)),
        _mc_res.rmse_history,
        color="red",
        linewidth=2,
        label=f"MC α={_alpha_val:.2f}",
    )

    _axes1[1].set_xlabel("Episodes", fontsize=12)
    _axes1[1].set_ylabel("RMS Error", fontsize=12)
    _axes1[1].set_title(f"RMSE Comparison (α={_alpha_val:.2f})", fontsize=14)
    _axes1[1].legend(fontsize=10)
    _axes1[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # ==================== AVERAGED MULTI-RUN PLOT ====================

    # Config for multi-run experiments
    _exp_cfg = ExperimentConfig(
        mrp=MRPConfig(n_states=5, gamma=1.0),
        algorithm=AlgorithmConfig(n_runs=100, n_episodes=100),
        alpha_values_td=[0.05, 0.1, 0.15],
        alpha_values_mc=[0.01, 0.02, 0.03, 0.04],
        random_seed=45,
    )

    np.random.seed(_exp_cfg.random_seed)

    # Store results
    _results = []

    # Run TD experiments
    print("\nRunning TD experiments...")
    for _alpha in _exp_cfg.alpha_values_td:
        print(f"  TD α={_alpha}...")
        _td_rmse_acc = np.zeros(
            (_exp_cfg.algorithm.n_runs, _exp_cfg.algorithm.n_episodes + 1)
        )

        for _run in range(_exp_cfg.algorithm.n_runs):
            _env = RandomWalk_ENV_MRP(_exp_cfg.mrp)
            _algo = AlgorithmConfig(
                alpha=_alpha, n_episodes=_exp_cfg.algorithm.n_episodes
            )
            _res = td_zero(_env, _algo, _exp_cfg.mrp.true_values)
            _td_rmse_acc[_run, :] = np.array(_res.rmse_history)

        _td_mean = np.mean(_td_rmse_acc, axis=0)

        for _ep in range(_exp_cfg.algorithm.n_episodes + 1):
            _results.append(
                {
                    "Episode": _ep,
                    "RMSE": float(_td_mean[_ep]),
                    "Method": "TD",
                    "Alpha": _alpha,
                }
            )

    # Run MC experiments
    print("Running MC experiments...")
    for _alpha in _exp_cfg.alpha_values_mc:
        print(f"  MC α={_alpha}...")
        _mc_rmse_acc = np.zeros(
            (_exp_cfg.algorithm.n_runs, _exp_cfg.algorithm.n_episodes + 1)
        )

        for _run in range(_exp_cfg.algorithm.n_runs):
            _env = RandomWalk_ENV_MRP(_exp_cfg.mrp)
            _algo = AlgorithmConfig(
                alpha=_alpha, n_episodes=_exp_cfg.algorithm.n_episodes
            )
            _res = monte_carlo_constant_alpha(
                _env, _algo, _exp_cfg.mrp.true_values
            )
            _mc_rmse_acc[_run, :] = np.array(_res.rmse_history)

        _mc_mean = np.mean(_mc_rmse_acc, axis=0)

        for _ep in range(_exp_cfg.algorithm.n_episodes + 1):
            _results.append(
                {
                    "Episode": _ep,
                    "RMSE": float(_mc_mean[_ep]),
                    "Method": "MC",
                    "Alpha": _alpha,
                }
            )

    print("  Experiments completed!")

    # Create averaged plot
    _df_avg = pd.DataFrame(_results)
    _fig2, _ax2 = plt.subplots(figsize=(10, 6))

    # TD curves (blue shades)
    _td_cols = ["darkblue", "blue", "royalblue"]
    for _i, _alpha in enumerate(_exp_cfg.alpha_values_td):
        _td_dat = _df_avg[
            (_df_avg["Method"] == "TD") & (_df_avg["Alpha"] == _alpha)
        ]
        _ax2.plot(
            _td_dat["Episode"],
            _td_dat["RMSE"],
            color=_td_cols[_i % len(_td_cols)],
            label=f"TD α={_alpha}",
            linewidth=2,
        )

    # MC curves (red shades)
    _mc_cols = ["darkred", "red", "lightcoral", "salmon"]
    for _i, _alpha in enumerate(_exp_cfg.alpha_values_mc):
        _mc_dat = _df_avg[
            (_df_avg["Method"] == "MC") & (_df_avg["Alpha"] == _alpha)
        ]
        _ax2.plot(
            _mc_dat["Episode"],
            _mc_dat["RMSE"],
            color=_mc_cols[_i % len(_mc_cols)],
            label=f"MC α={_alpha}",
            linewidth=2,
        )

    _ax2.set_xlabel("Walks / Episodes", fontsize=12)
    _ax2.set_ylabel("RMS Error (averaged over states)", fontsize=12)
    _ax2.set_title("TD vs MC: Empirical Performance Comparison", fontsize=14)
    _ax2.legend(fontsize=9, loc="upper right")
    _ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Display both figures
    mo.vstack(
        [
            mo.md("## Interactive Single Run (adjust α with slider above)"),
            _fig1,
            mo.md("## Averaged Multi-Run Comparison (100 runs each)"),
            _fig2,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Example 6.3: Random Walk under Batch Updating
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## BAtches set up
    """)
    return


@app.cell
def _(EpisodeTransition, List, NDArray, RandomWalk_ENV_MRP, Tuple, np):
    def batch_td_zero(
        env: RandomWalk_ENV_MRP,
        episodes_batch: List[List[EpisodeTransition]],
        alpha: float,
        true_values: NDArray[np.float64],
        max_iterations: int = 200,
        convergence_threshold: float = 1e-3,
    ) -> Tuple[NDArray[np.float64], int]:
        """
        Batch TD(0): repeatedly train on all episodes until convergence.

        Args:
            env: MRP environment
            episodes_batch: List of all episodes to train on
            alpha: Learning rate (should be small for convergence)
            true_values: Ground truth for comparison
            max_iterations: Maximum sweeps through batch
            convergence_threshold: Stop when max value change < this

        Returns:
            Tuple of (converged values, iterations needed)
        """
        # Initialize all states to 0.5
        V: NDArray[np.float64] = np.ones(env.n_states) * 0.5

        # Repeatedly sweep through all episodes until convergence
        for iteration in range(max_iterations):
            _delta_max: float = 0.0  # Track largest change for convergence check

            # Process each episode in the batch
            for _episode in episodes_batch:
                # TD(0) updates for this episode
                for _t in range(len(_episode) - 1):
                    _current_trans = _episode[_t]
                    _next_trans = _episode[_t + 1]

                    _state = _current_trans.state
                    _next_state = _next_trans.state
                    _reward = _next_trans.reward

                    # Check if next state is terminal
                    _done = (
                        _next_state == env.config.left_terminal
                        or _next_state == env.config.right_terminal
                    )

                    # TD target
                    if _done:
                        _td_target = _reward
                    else:
                        _td_target = _reward + env.config.gamma * V[_next_state]

                    # TD update
                    _td_error = _td_target - V[_state]
                    _old_value = V[_state]
                    V[_state] += alpha * _td_error

                    # Track max change
                    _delta_max = max(_delta_max, abs(V[_state] - _old_value))

            # Check convergence
            if _delta_max < convergence_threshold:
                return V, iteration + 1

        return V, max_iterations
    return (batch_td_zero,)


@app.cell
def _(EpisodeTransition, List, NDArray, RandomWalk_ENV_MRP, Tuple, np):
    def batch_monte_carlo(
        env: RandomWalk_ENV_MRP,
        episodes_batch: List[List[EpisodeTransition]],
        alpha: float,
        true_values: NDArray[np.float64],
        max_iterations: int = 200,
        convergence_threshold: float = 1e-3,
    ) -> Tuple[NDArray[np.float64], int]:
        """
        Batch MC: repeatedly train on all episodes until convergence.

        Args:
            env: MRP environment
            episodes_batch: List of all episodes to train on
            alpha: Learning rate (should be small for convergence)
            true_values: Ground truth for comparison
            max_iterations: Maximum sweeps through batch
            convergence_threshold: Stop when max value change < this

        Returns:
            Tuple of (converged values, iterations needed)
        """
        # Initialize all states to 0.5
        V: NDArray[np.float64] = np.ones(env.n_states) * 0.5

        # Repeatedly sweep through all episodes until convergence
        for iteration in range(max_iterations):
            _delta_max: float = 0.0  # Track largest change

            # Process each episode in the batch
            for _episode in episodes_batch:
                # Calculate returns backwards for this episode
                G: float = 0.0

                for _transition, _next_trans in zip(
                    reversed(_episode[:-1]), reversed(_episode[1:])
                ):
                    # Accumulate return
                    G = _next_trans.reward + env.config.gamma * G

                    # MC update
                    _old_value = V[_transition.state]
                    V[_transition.state] += alpha * (G - V[_transition.state])

                    # Track max change
                    _delta_max = max(
                        _delta_max, abs(V[_transition.state] - _old_value)
                    )

            # Check convergence
            if _delta_max < convergence_threshold:
                return V, iteration + 1

        return V, max_iterations
    return (batch_monte_carlo,)


@app.cell
def _(mo):
    mo.md(r"""
    ## runer experiment
    """)
    return


@app.cell
def _(
    EpisodeTransition,
    List,
    MRPConfig,
    RandomWalk_ENV_MRP,
    batch_monte_carlo,
    batch_td_zero,
    calculate_rmse,
    generate_episode,
    pd,
):
    def run_batch_experiment(
        n_episodes: int = 50, n_runs: int = 20, alpha: float = 0.01
    ) -> pd.DataFrame:
        """
        Run batch training experiment comparing TD and MC.
        Exact book procedure: after EACH episode, batch train until convergence.

        Args:
            n_episodes: Number of episodes to accumulate
            n_runs: Number of independent runs
            alpha: Learning rate for batch convergence

        Returns:
            DataFrame with results
        """
        _mrp = MRPConfig(n_states=5, gamma=1.0)
        _results = []

        print(f"Running batch experiment: {n_runs} runs × {n_episodes} episodes")
        print("This implements the exact book procedure...")

        for _run in range(n_runs):
            _env = RandomWalk_ENV_MRP(_mrp)
            _episodes_accumulated: List[List[EpisodeTransition]] = []

            # After EACH new episode, do batch training
            for _ep_num in range(1, n_episodes + 1):
                # Generate ONE new episode
                _new_episode = generate_episode(_env)
                _episodes_accumulated.append(_new_episode)

                # Batch TD: train on ALL episodes accumulated so far
                _V_td, _iters_td = batch_td_zero(
                    _env, _episodes_accumulated, alpha, _mrp.true_values
                )
                _rmse_td = calculate_rmse(_V_td, _mrp.true_values)

                # Batch MC: train on ALL episodes accumulated so far
                _V_mc, _iters_mc = batch_monte_carlo(
                    _env, _episodes_accumulated, alpha, _mrp.true_values
                )
                _rmse_mc = calculate_rmse(_V_mc, _mrp.true_values)

                # Store results for this episode count
                _results.append(
                    {
                        "Run": _run,
                        "Episodes": _ep_num,
                        "RMSE": _rmse_td,
                        "Method": "Batch TD",
                    }
                )
                _results.append(
                    {
                        "Run": _run,
                        "Episodes": _ep_num,
                        "RMSE": _rmse_mc,
                        "Method": "Batch MC",
                    }
                )

            if (_run + 1) % 5 == 0:
                print(f"  Completed {_run + 1}/{n_runs} runs")

        print("  Batch experiment completed!")
        return pd.DataFrame(_results)
    return (run_batch_experiment,)


@app.cell
def _(plt, run_batch_experiment):
    # Run batch training experiment
    batch_results = run_batch_experiment(
        n_episodes=50,
        n_runs=20,
        # n_episodes=1,
        # n_runs=1,
        alpha=0.01,
    )

    # Calculate mean RMSE across runs
    batch_avg = (
        batch_results.groupby(["Episodes", "Method"])["RMSE"].mean().reset_index()
    )

    # DEBUG: Check what data we have
    print("Batch avg shape:", batch_avg.shape)
    print("Unique methods:", batch_avg["Method"].unique())
    print("First few rows:")
    print(batch_avg.head(10))

    # Plot batch training results (Figure 6.2 style)
    _fig_batch, _ax_batch = plt.subplots(figsize=(10, 6))

    # Plot TD curve
    _td_batch = batch_avg[batch_avg["Method"] == "Batch TD"]
    _ax_batch.plot(
        _td_batch["Episodes"],
        _td_batch["RMSE"],
        color="blue",
        linewidth=2.5,
        label="Batch TD(0)",
    )

    # Plot MC curve
    _mc_batch = batch_avg[batch_avg["Method"] == "Batch MC"]
    _ax_batch.plot(
        _mc_batch["Episodes"],
        _mc_batch["RMSE"],
        color="red",
        linewidth=2.5,
        label="Batch MC",
    )

    _ax_batch.set_xlabel("Episodes", fontsize=12)
    _ax_batch.set_ylabel("RMS Error (averaged over states and runs)", fontsize=12)
    _ax_batch.set_title("Batch Training: TD(0) vs MC on Random Walk", fontsize=14)
    _ax_batch.legend(fontsize=11, loc="upper right")
    _ax_batch.grid(True, alpha=0.3)
    _ax_batch.set_ylim(0, 0.3)  # Typical range for this task

    plt.tight_layout()

    # # 👇 THIS is the missing piece
    # plt.show()
    return (batch_avg,)


@app.cell
def _(mo):
    mo.md(r"""
    ## result
    """)
    return


@app.cell
def _(batch_avg, plt):
    # Check the data
    print(batch_avg.head())

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 4))

    for method, group in batch_avg.groupby("Method"):
        ax.plot(group["Episodes"], group["RMSE"], marker="o", label=method)

    ax.set_xlabel("Episodes")
    ax.set_ylabel("RMSE")
    ax.set_title("Batch TD(0) vs Monte Carlo")
    ax.legend()
    ax.grid(True)

    # 👇 This line ensures the plot actually renders
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Example 6.4: You are the Predictor
    """)
    return


@app.cell
def _(EpisodeTransition):
    # Define the 8 episodes from Example 6.4
    # States: A=0, B=1, Terminal=2
    # Format: list of (state, reward) transitions

    example64_episodes = [
        # Episode 1: A → B → Terminal (rewards: 0 after B, 0 at terminal)
        [
            EpisodeTransition(state=0, reward=0.0),  # Start at A
            EpisodeTransition(state=1, reward=0.0),  # Go to B with reward 0
            EpisodeTransition(state=2, reward=0.0),  # Terminate with reward 0
        ],
        # Episodes 2-7: B → Terminal with reward 1 (6 times)
        *[
            [
                EpisodeTransition(state=1, reward=0.0),  # Start at B
                EpisodeTransition(state=2, reward=1.0),  # Terminate with reward 1
            ]
            for _ in range(6)
        ],
        # Episode 8: B → Terminal with reward 0 (1 time)
        [
            EpisodeTransition(state=1, reward=0.0),  # Start at B
            EpisodeTransition(state=2, reward=0.0),  # Terminate with reward 0
        ],
    ]

    print(f"Total episodes: {len(example64_episodes)}")
    print(
        f"Episode 1 (A→B→Terminal): {[(t.state, t.reward) for t in example64_episodes[0]]}"
    )
    print(f"Episodes 2-7: B→Terminal with reward=1 (6 episodes)")
    print(f"Episode 8: B→Terminal with reward=0 (1 episode)")
    return (example64_episodes,)


@app.cell
def _(EpisodeTransition, List, NDArray, np):
    def batch_td_example64(
        episodes: List[List[EpisodeTransition]],
        alpha: float = 0.01,
        max_iters: int = 10000,
    ) -> NDArray[np.float64]:
        """
        Batch TD(0) on Example 6.4 data.
        States: A=0, B=1, Terminal=2
        """
        # Initialize: A, B start at 0.5, Terminal at 0.0
        V = np.array([0.5, 0.5, 0.0])

        # Batch training: sweep through all episodes until convergence
        for _iter in range(max_iters):
            _delta = 0.0

            for _ep in episodes:
                # TD updates for this episode
                for _t in range(len(_ep) - 1):
                    _s = _ep[_t].state
                    _s_next = _ep[_t + 1].state
                    _r = _ep[_t + 1].reward

                    # Skip if already at terminal
                    if _s == 2:
                        continue

                    # TD target: R + γV(S')
                    _target = _r + 1.0 * V[_s_next]  # gamma=1

                    # TD update
                    _old = V[_s]
                    V[_s] += alpha * (_target - V[_s])
                    _delta = max(_delta, abs(V[_s] - _old))

            # Check convergence
            if _delta < 1e-6:
                print(f"  Batch TD converged in {_iter + 1} iterations")
                break

        return V
    return (batch_td_example64,)


@app.cell
def _(EpisodeTransition, List, NDArray, np):
    def batch_mc_example64(
        episodes: List[List[EpisodeTransition]],
        alpha: float = 0.01,
        max_iters: int = 10000,
    ) -> NDArray[np.float64]:
        """
        Batch MC on Example 6.4 data.
        States: A=0, B=1, Terminal=2
        """
        # Initialize: A, B start at 0.5, Terminal at 0.0
        V = np.array([0.5, 0.5, 0.0])

        # Batch training: sweep through all episodes until convergence
        for _iter in range(max_iters):
            _delta = 0.0

            for _ep in episodes:
                # Calculate returns backwards
                G = 0.0
                for _t in range(len(_ep) - 2, -1, -1):
                    _s = _ep[_t].state
                    _r_next = _ep[_t + 1].reward

                    # Accumulate return
                    G = _r_next + 1.0 * G  # gamma=1

                    # MC update (skip terminal)
                    if _s < 2:
                        _old = V[_s]
                        V[_s] += alpha * (G - V[_s])
                        _delta = max(_delta, abs(V[_s] - _old))

            # Check convergence
            if _delta < 1e-6:
                print(f"  Batch MC converged in {_iter + 1} iterations")
                break

        return V
    return (batch_mc_example64,)


@app.cell(hide_code=True)
def _(V_mc_ex64, V_td_ex64, np, plt):
    # # Create comparison visualization
    _fig_ex64, _ax_ex64 = plt.subplots(figsize=(10, 6))

    # State names (only A and B)
    _states_ex64 = ["State A", "State B"]
    _x_pos = np.arange(len(_states_ex64))

    # Bar width
    _width = 0.35

    # Plot bars side by side
    _bars_td = _ax_ex64.bar(
        _x_pos - _width / 2,  # Shift left by half width
        V_td_ex64[:2],
        _width,
        label="Batch TD (certainty-equiv.)",
        color="blue",
        alpha=0.8,
    )
    _bars_mc = _ax_ex64.bar(
        _x_pos + _width / 2,  # Shift right by half width
        V_mc_ex64[:2],
        _width,
        label="Batch MC (sample avg.)",
        color="red",
        alpha=0.8,
    )

    # Add value labels on bars
    for _bar in _bars_td:
        _height = _bar.get_height()
        _ax_ex64.text(
            _bar.get_x() + _bar.get_width() / 2.0,
            _height + 0.02,
            f"{_height:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    for _bar in _bars_mc:
        _height = _bar.get_height()
        _ax_ex64.text(
            _bar.get_x() + _bar.get_width() / 2.0,
            _height + 0.02,
            f"{_height:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Customize plot
    _ax_ex64.set_xlabel("State", fontsize=13)
    _ax_ex64.set_ylabel("Value Estimate", fontsize=13)
    _ax_ex64.set_title(
        "Example 6.4: TD learns MDP structure, MC learns sample averages",
        fontsize=14,
    )
    _ax_ex64.set_xticks(_x_pos)
    _ax_ex64.set_xticklabels(_states_ex64, fontsize=12)
    _ax_ex64.legend(fontsize=11, loc="upper left")
    _ax_ex64.grid(True, alpha=0.3, axis="y")
    _ax_ex64.set_ylim(0, 0.9)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Example 6.5: Windy Gridworld
    """)
    return


@app.cell
def _(Tuple):
    class WindyGridworld:
        """
        Windy Gridworld environment from Example 6.5.
        7 rows × 10 columns grid with upward wind in middle columns.
        """

        def __init__(self):
            self.rows = 7
            self.cols = 10
            self.start = (3, 0)  # Start position (row, col)
            self.goal = (3, 7)  # Goal position

            # Wind strength for each column (shifts position upward)
            self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

            # Action space: 0=up, 1=down, 2=left, 3=right
            self.actions = 4
            self.action_effects = {
                0: (-1, 0),  # up
                1: (1, 0),  # down
                2: (0, -1),  # left
                3: (0, 1),  # right
            }

            self.current_state = self.start

        def reset(self) -> Tuple[int, int]:
            """Reset to start position."""
            self.current_state = self.start
            return self.current_state

        def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
            """
            Take action and return (next_state, reward, done).

            Args:
                action: 0=up, 1=down, 2=left, 3=right

            Returns:
                (next_state, reward, done)
            """
            _row, _col = self.current_state
            _d_row, _d_col = self.action_effects[action]

            # Apply action
            _new_row = _row + _d_row
            _new_col = _col + _d_col

            # Apply wind (moves up by wind strength)
            _wind_effect = self.wind[_col]  # Wind based on current column
            _new_row -= _wind_effect

            # Clip to grid boundaries
            _new_row = max(0, min(self.rows - 1, _new_row))
            _new_col = max(0, min(self.cols - 1, _new_col))

            self.current_state = (_new_row, _new_col)

            # Check if reached goal
            _done = self.current_state == self.goal
            _reward = 0.0 if _done else -1.0

            return self.current_state, _reward, _done

        def state_to_index(self, state: Tuple[int, int]) -> int:
            """Convert (row, col) to flat index."""
            return state[0] * self.cols + state[1]

        def index_to_state(self, index: int) -> Tuple[int, int]:
            """Convert flat index to (row, col)."""
            return (index // self.cols, index % self.cols)
    return (WindyGridworld,)


@app.cell
def _(List, NDArray, Tuple, WindyGridworld, np):
    def sarsa_windy_gridworld(
        env: WindyGridworld,
        n_episodes: int = 500,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 1.0,
    ) -> Tuple[NDArray[np.float64], List[int]]:
        """
        SARSA algorithm for Windy Gridworld.

        Args:
            env: Windy gridworld environment
            n_episodes: Number of episodes to train
            alpha: Learning rate
            epsilon: Exploration rate for ε-greedy
            gamma: Discount factor

        Returns:
            (Q_values, episode_lengths): Q-table and length of each episode
        """
        # Initialize Q(s,a) arbitrarily (here: zeros)
        _n_states = env.rows * env.cols
        Q = np.zeros((_n_states, env.actions))

        # Track episode lengths for plotting
        _episode_lengths = []
        _total_steps = 0

        # Loop for each episode
        for _ep in range(n_episodes):
            # Initialize S
            _state = env.reset()
            _s_idx = env.state_to_index(_state)

            # Choose A from S using ε-greedy policy
            if np.random.random() < epsilon:
                _action = np.random.randint(env.actions)
            else:
                _action = np.argmax(Q[_s_idx])

            _steps = 0
            _done = False

            # Loop for each step of episode
            while not _done:
                # Take action A, observe R, S'
                _next_state, _reward, _done = env.step(_action)
                _next_s_idx = env.state_to_index(_next_state)

                # Choose A' from S' using ε-greedy policy
                if np.random.random() < epsilon:
                    _next_action = np.random.randint(env.actions)
                else:
                    _next_action = np.argmax(Q[_next_s_idx])

                # SARSA update: Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
                if _done:
                    _target = _reward  # No future value at terminal
                else:
                    _target = _reward + gamma * Q[_next_s_idx, _next_action]

                Q[_s_idx, _action] += alpha * (_target - Q[_s_idx, _action])

                # S ← S'; A ← A'
                _s_idx = _next_s_idx
                _action = _next_action
                _steps += 1
                _total_steps += 1

            _episode_lengths.append(_total_steps)

        return Q, _episode_lengths
    return (sarsa_windy_gridworld,)


@app.cell
def _(WindyGridworld, sarsa_windy_gridworld):
    print("Training SARSA on Windy Gridworld...")

    # Create environment
    windy_env = WindyGridworld()

    # Run SARSA
    Q_windy, episode_lengths_windy = sarsa_windy_gridworld(
        windy_env, n_episodes=170, alpha=0.5, epsilon=0.1, gamma=1.0
    )

    print(f"  Training completed! Ran {len(episode_lengths_windy)} episodes")
    print(f"  Total steps: {episode_lengths_windy[-1]}")
    print(
        f"  Final episode length: {episode_lengths_windy[-1] - episode_lengths_windy[-2] if len(episode_lengths_windy) > 1 else 0}"
    )
    return Q_windy, episode_lengths_windy, windy_env


@app.cell
def _(List, NDArray, Q_windy, Tuple, WindyGridworld, np, windy_env):
    def get_policy_path(
        env: WindyGridworld, Q: NDArray[np.float64]
    ) -> List[Tuple[int, int]]:
        """
        Extract greedy policy path from Q-values.

        Returns:
            List of (row, col) positions from start to goal
        """
        _path = []
        _state = env.reset()
        _path.append(_state)
        _done = False
        _max_steps = 100  # Prevent infinite loops

        for _ in range(_max_steps):
            if _done:
                break

            _s_idx = env.state_to_index(_state)
            _action = np.argmax(Q[_s_idx])  # Greedy action

            _state, _reward, _done = env.step(_action)
            _path.append(_state)

        return _path


    # Get optimal path
    optimal_path = get_policy_path(windy_env, Q_windy)
    print(f"Optimal path length: {len(optimal_path) - 1} steps")
    print(f"Path: Start {optimal_path[0]} → Goal {optimal_path[-1]}")
    return (optimal_path,)


@app.cell
def _(episode_lengths_windy, np, optimal_path, plt, windy_env):
    # Create figure with 2 subplots
    _fig_windy, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # LEFT PLOT: Learning curve (episodes vs time steps)
    _ax1.plot(
        range(1, len(episode_lengths_windy) + 1),
        episode_lengths_windy,
        linewidth=2,
    )
    _ax1.set_xlabel("Episodes", fontsize=12)
    _ax1.set_ylabel("Time steps", fontsize=12)
    _ax1.set_title("Windy Gridworld: SARSA Learning Curve", fontsize=14)
    _ax1.grid(True, alpha=0.3)

    # RIGHT PLOT: Gridworld with optimal path
    _grid = np.zeros((windy_env.rows, windy_env.cols))

    # Mark start and goal
    _grid[windy_env.start] = 0.3
    _grid[windy_env.goal] = 0.8

    # Mark optimal path
    for _pos in optimal_path[1:-1]:  # Exclude start and goal
        _grid[_pos] = 0.5

    _im = _ax2.imshow(_grid, cmap="RdYlGn", alpha=0.7, vmin=0, vmax=1)

    # Add wind arrows
    for _col in range(windy_env.cols):
        _wind_str = windy_env.wind[_col]
        if _wind_str > 0:
            _ax2.text(
                _col,
                windy_env.rows - 0.5,
                f"↑{_wind_str}",
                ha="center",
                va="center",
                fontsize=10,
                color="blue",
                fontweight="bold",
            )

    # Add start and goal markers
    _ax2.text(
        windy_env.start[1],
        windy_env.start[0],
        "S",
        ha="center",
        va="center",
        fontsize=16,
        color="black",
        fontweight="bold",
    )
    _ax2.text(
        windy_env.goal[1],
        windy_env.goal[0],
        "G",
        ha="center",
        va="center",
        fontsize=16,
        color="black",
        fontweight="bold",
    )

    # Draw path arrows
    for _i in range(len(optimal_path) - 1):
        _r1, _c1 = optimal_path[_i]
        _r2, _c2 = optimal_path[_i + 1]
        _ax2.arrow(
            _c1,
            _r1,
            _c2 - _c1,
            _r2 - _r1,
            head_width=0.15,
            head_length=0.15,
            fc="red",
            ec="red",
            linewidth=1.5,
        )

    _ax2.set_title(f"Optimal Path ({len(optimal_path) - 1} steps)", fontsize=14)
    _ax2.set_xlabel("Column", fontsize=12)
    _ax2.set_ylabel("Row", fontsize=12)
    _ax2.set_xticks(range(windy_env.cols))
    _ax2.set_yticks(range(windy_env.rows))
    _ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Example 6.6: Cliff Walking (SARSA vs Q-learning)
    """)
    return


@app.cell
def _(Tuple):
    class CliffWalking:
        """Cliff Walking environment from Example 6.6."""

        def __init__(self):
            self.rows = 4
            self.cols = 12
            self.start = (3, 0)  # Bottom-left
            self.goal = (3, 11)  # Bottom-right

            # The cliff: bottom row from (3,1) to (3,10)
            self.cliff = [(3, c) for c in range(1, 11)]

            # Actions: 0=up, 1=down, 2=left, 3=right
            self.actions = 4
            self.action_effects = {
                0: (-1, 0),  # up
                1: (1, 0),  # down
                2: (0, -1),  # left
                3: (0, 1),  # right
            }

            self.current_state = self.start

        def reset(self) -> Tuple[int, int]:
            """Reset to start."""
            self.current_state = self.start
            return self.current_state

        def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
            """Take action, return (next_state, reward, done)."""
            _row, _col = self.current_state
            _d_row, _d_col = self.action_effects[action]

            # Apply action
            _new_row = max(0, min(self.rows - 1, _row + _d_row))
            _new_col = max(0, min(self.cols - 1, _col + _d_col))

            # Check if stepped into cliff
            if (_new_row, _new_col) in self.cliff:
                self.current_state = self.start  # Reset to start
                return self.start, -100.0, False  # Not done, just penalized

            # Check if reached goal
            self.current_state = (_new_row, _new_col)
            _done = self.current_state == self.goal
            _reward = 0.0 if _done else -1.0

            return self.current_state, _reward, _done

        def state_to_index(self, state: Tuple[int, int]) -> int:
            """Convert (row, col) to flat index."""
            return state[0] * self.cols + state[1]
    return (CliffWalking,)


@app.cell
def _(CliffWalking, List, NDArray, Tuple, np):
    def sarsa_cliff(
        env: CliffWalking,
        n_episodes: int = 5000,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 1.0,
    ) -> Tuple[NDArray[np.float64], List[float]]:
        """SARSA on Cliff Walking."""

        _n_states = env.rows * env.cols
        Q = np.zeros((_n_states, env.actions))
        _episode_rewards = []

        for _ep in range(n_episodes):
            _state = env.reset()
            _s_idx = env.state_to_index(_state)

            # Choose A from S using ε-greedy
            if np.random.random() < epsilon:
                _action = np.random.randint(env.actions)
            else:
                _action = np.argmax(Q[_s_idx])

            _total_reward = 0.0
            _done = False

            while not _done:
                # Take action A, observe R, S'
                _next_state, _reward, _done = env.step(_action)
                _next_s_idx = env.state_to_index(_next_state)
                _total_reward += _reward

                # Choose A' from S' using ε-greedy
                if np.random.random() < epsilon:
                    _next_action = np.random.randint(env.actions)
                else:
                    _next_action = np.argmax(Q[_next_s_idx])

                # SARSA update
                if _done:
                    _target = _reward
                else:
                    _target = _reward + gamma * Q[_next_s_idx, _next_action]

                Q[_s_idx, _action] += alpha * (_target - Q[_s_idx, _action])

                _s_idx = _next_s_idx
                _action = _next_action

            _episode_rewards.append(_total_reward)

        return Q, _episode_rewards
    return (sarsa_cliff,)


@app.cell
def _(CliffWalking, List, NDArray, Tuple, np):
    def qlearning_cliff(
        env: CliffWalking,
        n_episodes: int = 5000,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 1.0,
    ) -> Tuple[NDArray[np.float64], List[float]]:
        """Q-learning on Cliff Walking."""

        _n_states = env.rows * env.cols
        Q = np.zeros((_n_states, env.actions))
        _episode_rewards = []

        for _ep in range(n_episodes):
            _state = env.reset()
            _total_reward = 0.0
            _done = False

            while not _done:
                _s_idx = env.state_to_index(_state)

                # Choose A from S using ε-greedy
                if np.random.random() < epsilon:
                    _action = np.random.randint(env.actions)
                else:
                    _action = np.argmax(Q[_s_idx])

                # Take action A, observe R, S'
                _next_state, _reward, _done = env.step(_action)
                _next_s_idx = env.state_to_index(_next_state)
                _total_reward += _reward

                # Q-learning update: use max over next actions
                if _done:
                    _target = _reward
                else:
                    _target = _reward + gamma * np.max(Q[_next_s_idx])

                Q[_s_idx, _action] += alpha * (_target - Q[_s_idx, _action])

                _state = _next_state

            _episode_rewards.append(_total_reward)

        return Q, _episode_rewards
    return (qlearning_cliff,)


@app.cell
def _(CliffWalking, np, qlearning_cliff, sarsa_cliff):
    # print("Running Cliff Walking comparison...")

    # cliff_env = CliffWalking()

    # print("  Training SARSA...")
    # Q_sarsa, rewards_sarsa = sarsa_cliff(cliff_env, n_episodes=500)

    # print("  Training Q-learning...")
    # Q_qlearn, rewards_qlearn = qlearning_cliff(cliff_env, n_episodes=500)

    # print("Training completed!")

    print("Running Cliff Walking comparison...")

    cliff_env = CliffWalking()

    # Run multiple times and average for stability
    n_runs = 10
    n_episodes = 500

    print(f"  Running {n_runs} independent runs for each algorithm...")

    # Store results from all runs
    all_sarsa_rewards = []
    all_qlearn_rewards = []

    for run in range(n_runs):
        print(f"    Run {run + 1}/{n_runs}")

        # SARSA with safer parameters
        Q_sarsa, rewards_sarsa = sarsa_cliff(
            cliff_env,
            n_episodes=n_episodes,
            alpha=0.1,  # Smaller alpha for stability
            epsilon=0.1,  # Standard exploration
            gamma=1.0,
        )
        all_sarsa_rewards.append(rewards_sarsa)

        # Q-learning with same parameters
        Q_qlearn, rewards_qlearn = qlearning_cliff(
            cliff_env,
            n_episodes=n_episodes,
            alpha=0.1,  # Smaller alpha for stability
            epsilon=0.1,  # Standard exploration
            gamma=1.0,
        )
        all_qlearn_rewards.append(rewards_qlearn)

    # Average across runs
    avg_sarsa_rewards = np.mean(all_sarsa_rewards, axis=0)
    avg_qlearn_rewards = np.mean(all_qlearn_rewards, axis=0)

    print("✓ Training completed!")
    return avg_qlearn_rewards, avg_sarsa_rewards


@app.cell
def _(avg_qlearn_rewards, avg_sarsa_rewards, np, plt):
    # # Smooth rewards for better visualization
    # def smooth(data, window=10):
    #     """Moving average smoothing."""
    #     return np.convolve(data, np.ones(window)/window, mode='valid')

    # _fig_cliff, _ax_cliff = plt.subplots(figsize=(10, 6))

    # # Plot smoothed rewards
    # _ax_cliff.plot(smooth(rewards_sarsa, 10), label='SARSA', color='blue', linewidth=2)
    # _ax_cliff.plot(smooth(rewards_qlearn, 10), label='Q-learning', color='red', linewidth=2)

    # _ax_cliff.set_xlabel('Episodes', fontsize=12)
    # _ax_cliff.set_ylabel('Sum of rewards during episode', fontsize=12)
    # _ax_cliff.set_title('Cliff Walking: SARSA vs Q-learning', fontsize=14)
    # _ax_cliff.legend(fontsize=11)
    # _ax_cliff.grid(True, alpha=0.3)
    # _ax_cliff.axhline(y=-13, color='green', linestyle='--', alpha=0.5, label='Optimal (safe path)')
    # _ax_cliff.axhline(y=-100, color='black', linestyle='--', alpha=0.3, label='Fell off cliff')

    # plt.tight_layout()
    # plt.show()

    # Smooth rewards for better visualization
    def smooth(data, window=10):
        """Moving average smoothing."""
        return np.convolve(data, np.ones(window) / window, mode="valid")


    _fig_cliff, _ax_cliff = plt.subplots(figsize=(12, 7))

    # Plot smoothed averaged rewards
    _sarsa_smooth = smooth(avg_sarsa_rewards, 10)
    _qlearn_smooth = smooth(avg_qlearn_rewards, 10)

    _ax_cliff.plot(
        _sarsa_smooth, label="SARSA (safe path)", color="blue", linewidth=2.5
    )
    _ax_cliff.plot(
        _qlearn_smooth,
        label="Q-learning (risky optimal)",
        color="red",
        linewidth=2.5,
    )

    # Add asymptotic performance lines
    _ax_cliff.axhline(
        y=-13,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label="Optimal path (~-13)",
    )
    _ax_cliff.axhline(
        y=-25,
        color="blue",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label="Safe path (~-25)",
    )

    _ax_cliff.set_xlabel("Episodes", fontsize=13)
    _ax_cliff.set_ylabel("Sum of rewards during episode", fontsize=13)
    _ax_cliff.set_title(
        "Cliff Walking: SARSA vs Q-learning (averaged over 10 runs)", fontsize=14
    )
    _ax_cliff.legend(fontsize=11, loc="lower right")
    _ax_cliff.grid(True, alpha=0.3)
    _ax_cliff.set_ylim(-100, 0)


    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Example 6.7: Maximization Bias
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Problem: What is Maximization Bias?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Maximization bias** occurs when we use max of estimated values as an estimate of the max value.

    **Example:** If true values are all 0, but estimates are uncertain (some above, some below 0):
    - max(true values) = 0
    - max(estimates) > 0 (positive bias!)

    **In RL:** Q-learning uses `max Q(s',a)` which causes overestimation.

    **Solution:** Double Q-learning - use two independent estimates to eliminate bias.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## bias mdp max
    """)
    return


@app.cell
def _(Tuple, np):
    class MaxBiasMDP:
        """
        Simple MDP from Example 6.7 to demonstrate maximization bias.

        States:
        - A: Starting state with 2 actions (left/right)
        - B: State reached by 'left' action, has many actions (all bad)
        - Terminal: Absorbing state

        Dynamics:
        - Right from A → Terminal (reward=0)
        - Left from A → B (reward=0)
        - Any action from B → Terminal (reward ~ N(-0.1, 1.0))

        Expected return starting left: -0.1 (suboptimal!)
        Expected return starting right: 0 (optimal!)
        """

        def __init__(self, n_actions_B: int = 10):
            # State space: 0=A, 1=B, 2=Terminal
            self.state_A = 0
            self.state_B = 1
            self.terminal = 2

            # Action space
            # In state A: 0=left (go to B), 1=right (go to terminal)
            self.actions_A = 2
            # In state B: many actions (all go to terminal with random reward)
            self.actions_B = n_actions_B

            # Current state
            self.current_state = self.state_A

        def reset(self) -> int:
            """Reset to starting state A."""
            self.current_state = self.state_A
            return self.current_state

        def step(self, action: int) -> Tuple[int, float, bool]:
            """
            Take action and return (next_state, reward, done).

            Args:
                action: Action index

            Returns:
                (next_state, reward, done)
            """
            if self.current_state == self.state_A:
                # In state A
                if action == 1:  # Right action
                    # Go directly to terminal with reward 0
                    self.current_state = self.terminal
                    return self.terminal, 0.0, True
                else:  # Left action (action 0)
                    # Go to state B with reward 0
                    self.current_state = self.state_B
                    return self.state_B, 0.0, False

            elif self.current_state == self.state_B:
                # In state B: all actions lead to terminal
                # Reward drawn from N(-0.1, 1.0) - mean -0.1, variance 1.0
                _reward = np.random.normal(-0.1, 1.0)
                self.current_state = self.terminal
                return self.terminal, _reward, True

            else:  # Terminal state
                return self.terminal, 0.0, True
    return (MaxBiasMDP,)


@app.cell
def _(mo):
    mo.md(r"""
    ## algo Configuration
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Q-learning algorithm
    """)
    return


@app.cell
def _(List, MaxBiasMDP, NDArray, Tuple, np):
    def qlearning_maxbias(
        env: MaxBiasMDP,
        n_episodes: int = 300,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        gamma: float = 1.0,
    ) -> Tuple[NDArray[np.float64], List[float]]:
        """
        Standard Q-learning on MaxBiasMDP.

        This will exhibit maximization bias because it uses max Q(s',a)
        in the update, causing overestimation of Q(B, a) values.

        Args:
            env: MaxBiasMDP environment
            n_episodes: Number of episodes to run
            alpha: Learning rate (step size)
            epsilon: Exploration probability for ε-greedy
            gamma: Discount factor

        Returns:
            (Q_values, left_action_pct): Q-table and % of times left was chosen
        """
        # Initialize Q(s,a) = 0 for all s,a
        # Shape: [3 states, max actions possible]
        # State A has 2 actions, State B has many, Terminal has 0
        _max_actions = max(env.actions_A, env.actions_B)
        Q = np.zeros((3, _max_actions))

        # Track percentage of times 'left' action chosen in state A
        _left_action_count = []
        _total_A_visits = 0
        _left_chosen = 0

        for _ep in range(n_episodes):
            # Initialize S (always start at A)
            _state = env.reset()
            _done = False

            while not _done:
                # Choose action using ε-greedy policy
                if _state == env.state_A:
                    # State A: choose from 2 actions
                    _n_actions = env.actions_A
                elif _state == env.state_B:
                    # State B: choose from many actions
                    _n_actions = env.actions_B
                else:
                    break  # Terminal state

                # ε-greedy action selection
                if np.random.random() < epsilon:
                    # Explore: random action
                    _action = np.random.randint(_n_actions)
                else:
                    # Exploit: greedy action (argmax Q)
                    _action = np.argmax(Q[_state, :_n_actions])

                # Track left action choice in state A
                if _state == env.state_A:
                    _total_A_visits += 1
                    if _action == 0:  # Left action
                        _left_chosen += 1

                # Take action A, observe R, S'
                _next_state, _reward, _done = env.step(_action)

                # Q-learning update: Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
                if _done:
                    # Terminal state: no future value
                    _target = _reward
                else:
                    # Use max over next state actions (THIS CAUSES MAXIMIZATION BIAS)
                    if _next_state == env.state_B:
                        _next_n_actions = env.actions_B
                    else:
                        _next_n_actions = env.actions_A
                    _target = _reward + gamma * np.max(
                        Q[_next_state, :_next_n_actions]
                    )

                # Update Q-value
                Q[_state, _action] += alpha * (_target - Q[_state, _action])

                # Move to next state
                _state = _next_state

            # Record percentage of left actions
            if _total_A_visits > 0:
                _left_action_count.append(100.0 * _left_chosen / _total_A_visits)
            else:
                _left_action_count.append(0.0)

        return Q, _left_action_count
    return (qlearning_maxbias,)


@app.cell
def _(mo):
    mo.md(r"""
    #### Double Q-learning algo
    """)
    return


@app.cell
def _(List, MaxBiasMDP, NDArray, Tuple, np):
    def double_qlearning_maxbias(
        env: MaxBiasMDP,
        n_episodes: int = 300,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        gamma: float = 1.0,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[float]]:
        """
        Double Q-learning on MaxBiasMDP.

        Uses two independent Q-value estimates Q1 and Q2.
        On each step, randomly choose which to update:
        - If updating Q1: use Q1 to select action, Q2 to evaluate it
        - If updating Q2: use Q2 to select action, Q1 to evaluate it

        This eliminates maximization bias.

        Args:
            env: MaxBiasMDP environment
            n_episodes: Number of episodes to run
            alpha: Learning rate
            epsilon: Exploration probability
            gamma: Discount factor

        Returns:
            (Q1, Q2, left_action_pct): Two Q-tables and % left action chosen
        """
        # Initialize Q1(s,a) and Q2(s,a) = 0 for all s,a
        _max_actions = max(env.actions_A, env.actions_B)
        Q1 = np.zeros((3, _max_actions))
        Q2 = np.zeros((3, _max_actions))

        # Track percentage of times 'left' action chosen in state A
        _left_action_count = []
        _total_A_visits = 0
        _left_chosen = 0

        for _ep in range(n_episodes):
            # Initialize S (always start at A)
            _state = env.reset()
            _done = False

            while not _done:
                # Determine number of actions available
                if _state == env.state_A:
                    _n_actions = env.actions_A
                elif _state == env.state_B:
                    _n_actions = env.actions_B
                else:
                    break  # Terminal

                # Choose action using ε-greedy based on Q1 + Q2
                # (Average or sum of both Q-functions)
                _Q_sum = Q1[_state, :_n_actions] + Q2[_state, :_n_actions]

                if np.random.random() < epsilon:
                    # Explore: random action
                    _action = np.random.randint(_n_actions)
                else:
                    # Exploit: greedy action based on sum
                    _action = np.argmax(_Q_sum)

                # Track left action in state A
                if _state == env.state_A:
                    _total_A_visits += 1
                    if _action == 0:  # Left action
                        _left_chosen += 1

                # Take action A, observe R, S'
                _next_state, _reward, _done = env.step(_action)

                # Double Q-learning: randomly choose which Q to update
                if np.random.random() < 0.5:
                    # Update Q1: use Q1 to select action, Q2 to evaluate
                    if _done:
                        _target = _reward
                    else:
                        # Select best action according to Q1
                        if _next_state == env.state_B:
                            _next_n_actions = env.actions_B
                        else:
                            _next_n_actions = env.actions_A
                        _best_action = np.argmax(Q1[_next_state, :_next_n_actions])
                        # Evaluate that action using Q2 (eliminates bias!)
                        _target = _reward + gamma * Q2[_next_state, _best_action]

                    # Update Q1
                    Q1[_state, _action] += alpha * (_target - Q1[_state, _action])
                else:
                    # Update Q2: use Q2 to select action, Q1 to evaluate
                    if _done:
                        _target = _reward
                    else:
                        # Select best action according to Q2
                        if _next_state == env.state_B:
                            _next_n_actions = env.actions_B
                        else:
                            _next_n_actions = env.actions_A
                        _best_action = np.argmax(Q2[_next_state, :_next_n_actions])
                        # Evaluate that action using Q1 (eliminates bias!)
                        _target = _reward + gamma * Q1[_next_state, _best_action]

                    # Update Q2
                    Q2[_state, _action] += alpha * (_target - Q2[_state, _action])

                # Move to next state
                _state = _next_state

            # Record percentage of left actions
            if _total_A_visits > 0:
                _left_action_count.append(100.0 * _left_chosen / _total_A_visits)
            else:
                _left_action_count.append(0.0)

        return Q1, Q2, _left_action_count
    return (double_qlearning_maxbias,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Running Experiments
    """)
    return


@app.cell
def _(MaxBiasMDP, double_qlearning_maxbias, np, qlearning_maxbias):
    # Run multiple independent trials and average results
    n_runs_maxbias = 1000  # Many runs to get smooth curves
    n_episodes_maxbias = 300

    print(
        f"Running {n_runs_maxbias} independent trials (this may take ~30 seconds)..."
    )

    # Storage for results
    _qlearn_left_percentages = []
    _double_qlearn_left_percentages = []

    # Create environment once (same for all trials)
    _maxbias_env = MaxBiasMDP(n_actions_B=10)

    for _run in range(n_runs_maxbias):
        # Run Q-learning
        _, _qlearn_left = qlearning_maxbias(
            _maxbias_env,
            n_episodes=n_episodes_maxbias,
            alpha=0.1,
            epsilon=0.1,
            gamma=1.0,
        )
        _qlearn_left_percentages.append(_qlearn_left)

        # Run Double Q-learning
        _, _, _double_left = double_qlearning_maxbias(
            _maxbias_env,
            n_episodes=n_episodes_maxbias,
            alpha=0.1,
            epsilon=0.1,
            gamma=1.0,
        )
        _double_qlearn_left_percentages.append(_double_left)

        if (_run + 1) % 100 == 0:
            print(f"  Completed {_run + 1}/{n_runs_maxbias} runs")

    # Average across all runs
    avg_qlearn_left = np.mean(_qlearn_left_percentages, axis=0)
    avg_double_left = np.mean(_double_qlearn_left_percentages, axis=0)

    print("✓ Experiments completed!")
    return avg_double_left, avg_qlearn_left


@app.cell
def _(mo):
    mo.md(r"""
    ## Results: Left Action Selection Over Time
    """)
    return


@app.cell
def _(avg_double_left, avg_qlearn_left, plt):
    # Create plot showing % of left actions over episodes
    _fig_maxbias, _ax_maxbias = plt.subplots(figsize=(10, 6))

    # Plot Q-learning (shows maximization bias)
    _ax_maxbias.plot(
        avg_qlearn_left, label="Q-learning (biased)", color="red", linewidth=2.5
    )

    # Plot Double Q-learning (eliminates bias)
    _ax_maxbias.plot(
        avg_double_left,
        label="Double Q-learning (unbiased)",
        color="blue",
        linewidth=2.5,
    )

    # Add reference line at 5% (optimal with ε=0.1)
    # With ε-greedy, optimal is: ε * 50% + (1-ε) * 0% = 5%
    # (explore 10% of time, choose left randomly 50%, never choose left when exploiting)
    _ax_maxbias.axhline(
        y=5.0, color="green", linestyle="--", linewidth=2, label="Optimal (5%)"
    )

    # Customize plot
    _ax_maxbias.set_xlabel("Episodes", fontsize=13)
    _ax_maxbias.set_ylabel("% Left actions from A", fontsize=13)
    _ax_maxbias.set_title(
        "Maximization Bias: Q-learning vs Double Q-learning", fontsize=14
    )
    _ax_maxbias.legend(fontsize=11, loc="upper right")
    _ax_maxbias.grid(True, alpha=0.3)
    _ax_maxbias.set_ylim(0, 80)


    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():


    return


if __name__ == "__main__":
    app.run()
