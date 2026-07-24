import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import model_core
importlib.reload(model_core)

from model_core import PARAMS, RecoveryModel, FirmAgent

# -------------------------
# Run logic
# -------------------------

def run_comparison(
        shock_type,
        shock_severity,
        network_type,
        network_size,
        simulation_length,
        seed_count,
        avg_degree,
        evaluation_recovery_threshold,
        hybrid_switch_perf,
        bridge_prob,
        rewire_prob,
        community_count,
        progress_bar=None,
        status_placeholder=None,
):

    regimes = ["DS", "DA", "ES", "EA", "HY"]

    total_runs = seed_count * len(regimes)
    completed_runs = 0

    rows = []
    trajectory_rows = []

    for seed in range(1, seed_count + 1):
        for reg in regimes:
            if status_placeholder is not None:
                status_placeholder.info(
                    f"Running scenario: seed {seed}/{seed_count}, regime {reg}"
                )

            test_params = PARAMS.copy()
            test_params["seed"] = seed
            test_params["regime"] = reg
            test_params["N"] = network_size
            test_params["T"] = simulation_length
            test_params["shock_type"] = shock_type
            test_params["shock_severity"] = shock_severity
            test_params["network_type"] = network_type
            test_params["avg_degree"] = avg_degree
            test_params["hybrid_switch_perf"] = hybrid_switch_perf
            test_params["bridge_prob"] = bridge_prob
            test_params["rewire_prob"] = rewire_prob
            test_params["C"] = community_count

            model = RecoveryModel(test_params)

            perf_t1 = None
            auc_perf = model.perf_net

            while model.running:
                model.step()

                if model.t == 1:
                    perf_t1 = model.perf_net

                auc_perf += model.perf_net

            df = model.datacollector.get_model_vars_dataframe().copy().reset_index(drop=True)
            df["seed"] = seed
            df["regime"] = reg
            trajectory_rows.append(df)

            recovered = df[
                (df["time"] > 0) &
                (df["perf_net"] >= evaluation_recovery_threshold)
            ]
            recovery_time = recovered["time"].iloc[0] if len(recovered) > 0 else np.nan

            total_innovation_stock = sum(a.innovation_stock for a in model.agents_by_id.values())

            perf_t5 = np.nan
            perf_t10 = np.nan

            if "time" in df.columns and "perf_net" in df.columns:
                perf_t5_rows = df[df["time"] == 5]
                perf_t10_rows = df[df["time"] == 10]

                if len(perf_t5_rows) > 0:
                    perf_t5 = perf_t5_rows["perf_net"].iloc[0]

                if len(perf_t10_rows) > 0:
                    perf_t10 = perf_t10_rows["perf_net"].iloc[0]

            cross_comm_ties = 0
            for u, v in model.G.edges():
                if model.agents_by_id[u].community != model.agents_by_id[v].community:
                    cross_comm_ties += 1

            rows.append({
                "N": network_size,
                "network_type": network_type,
                "shock_type": shock_type,
                "shock_severity": shock_severity,
                "seed": seed,
                "regime": reg,
                "loss_t1": test_params["baseline_perf"] - perf_t1,
                "perf_t1": perf_t1,
                "perf_t5": perf_t5,
                "perf_t10": perf_t10,
                "recovery_time": recovery_time,
                "final_perf_net": model.perf_net,
                "auc_perf": auc_perf,
                "final_active_pre_shock_ties": model.active_pre_shock_ties,
                "final_new_ties_count": model.new_ties_count,
                "final_cross_community_ties": cross_comm_ties,
                "final_innovation_stock": total_innovation_stock,
            })

            completed_runs += 1
            if progress_bar is not None:
                progress_bar.progress(completed_runs / total_runs)

    baseline_df = pd.DataFrame(rows)
    baseline_traj_df = pd.concat(trajectory_rows, ignore_index=True)

    baseline_traj_variance = (
        baseline_traj_df
        .groupby(["regime", "time"], as_index=False)
        .agg(
            perf_net_sd=("perf_net", "std"),
        )
    )

    p3_summary = (
        baseline_traj_variance[baseline_traj_variance["time"] > 0]
        .groupby("regime", as_index=False)
        .agg(
            mean_traj_sd=("perf_net_sd", "mean"),
        )
    )

    plot_order = ["DS", "DA", "ES", "EA", "HY"]

    baseline_summary = (
        baseline_df
        .groupby("regime", as_index=False)
        .agg(
            loss_t1_mean=("loss_t1", "mean"),
            perf_t1_mean=("perf_t1", "mean"),
            recovery_time_mean=("recovery_time", "mean"),
            final_perf_mean=("final_perf_net", "mean"),
            auc_perf_mean=("auc_perf", "mean"),
            final_new_ties_mean=("final_new_ties_count", "mean"),
            final_innovation_mean=("final_innovation_stock", "mean"),
        )
    )

    baseline_summary = baseline_summary.merge(
        p3_summary[["regime", "mean_traj_sd"]],
        on="regime",
        how="left"
    )

    baseline_summary = (
        baseline_summary
        .set_index("regime")
        .loc[plot_order]
        .reset_index()
    )

    compact_pretty = baseline_summary[[
        "regime",
        "loss_t1_mean",
        "recovery_time_mean",
        "final_perf_mean",
        "auc_perf_mean",
        "final_new_ties_mean",
        "final_innovation_mean",
        "mean_traj_sd",
    ]].copy()

    for col in [
        "loss_t1_mean",
        "final_perf_mean",
        "auc_perf_mean",
        "mean_traj_sd",
    ]:
        compact_pretty[col] = compact_pretty[col].round(3)

    for col in [
        "final_new_ties_mean",
        "final_innovation_mean",
    ]:
        compact_pretty[col] = compact_pretty[col].round(1)

    compact_pretty["recovery_time_mean"] = compact_pretty["recovery_time_mean"].apply(
        lambda x: f"Not reached by t={simulation_length}" if pd.isna(x) else round(x, 1)
    )

    compact_pretty = compact_pretty.rename(columns={
        "regime": "Recovery regime",
        "loss_t1_mean": "Recovery score loss at t = 1",
        "recovery_time_mean": f"Time to reach {evaluation_recovery_threshold:.2f} recovery score",
        "final_perf_mean": "Final recovery score",
        "auc_perf_mean": "Cumulative recovery score",
        "final_new_ties_mean": "New ties formed",
        "final_innovation_mean": "Innovation stock",
        "mean_traj_sd": "Trajectory dispersion",
    })

    full_pretty = baseline_df.copy()

    for col in [
        "shock_severity",
        "loss_t1",
        "perf_t1",
        "perf_t5",
        "perf_t10",
        "recovery_time",
        "final_perf_net",
        "auc_perf",
    ]:
        full_pretty[col] = full_pretty[col].round(3)

    for col in [
        "final_active_pre_shock_ties",
        "final_new_ties_count",
        "final_cross_community_ties",
        "final_innovation_stock",
    ]:
        full_pretty[col] = full_pretty[col].round(1)

    full_pretty["recovery_time"] = full_pretty["recovery_time"].apply(
        lambda x: f"Not reached by t={simulation_length}" if pd.isna(x) else round(x, 1)
    )

    full_pretty = full_pretty.rename(columns={
        "N": "Network size",
        "network_type": "Network type",
        "shock_type": "Shock type",
        "shock_severity": "Shock severity",
        "seed": "Seed",
        "regime": "Recovery regime",
        "loss_t1": "Recovery score loss at t = 1",
        "perf_t1": "Recovery score at t = 1",
        "perf_t5": "Recovery score at t = 5",
        "perf_t10": "Recovery score at t = 10",
        "recovery_time": f"Time to reach {evaluation_recovery_threshold:.2f} recovery score",
        "final_perf_net": "Final recovery score",
        "auc_perf": "Cumulative recovery score",
        "final_active_pre_shock_ties": "Final active pre-shock ties",
        "final_new_ties_count": "Final new ties formed",
        "final_cross_community_ties": "Final cross-community ties",
        "final_innovation_stock": "Final innovation stock",
    })

    return baseline_df, baseline_traj_df, compact_pretty, full_pretty


# -------------------------
# Plot helpers
# -------------------------

def make_trajectory_figures(baseline_traj_df):
    plot_order = ["DS", "DA", "ES", "EA", "HY"]

    baseline_mean_traj = (
        baseline_traj_df
        .groupby(["regime", "time"], as_index=False)
        .agg(
            perf_net_mean=("perf_net", "mean"),
            active_pre_shock_ties_mean=("active_pre_shock_ties", "mean"),
            new_ties_count_mean=("new_ties_count", "mean"),
        )
    )

    baseline_mean_traj["perf_loss_mean"] = 1.0 - baseline_mean_traj["perf_net_mean"]

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    regime_colors = {reg: default_colors[i] for i, reg in enumerate(plot_order)}

    figures = []

    for var, ylab, title in [
        ("perf_loss_mean", "Mean recovery score loss", "Recovery-score loss trajectories"),
        ("active_pre_shock_ties_mean", "Mean active pre-shock ties", "Restoration trajectories"),
        ("new_ties_count_mean", "Mean new ties", "Novelty trajectories"),
    ]:
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for reg in plot_order:
            sub = baseline_mean_traj[baseline_mean_traj["regime"] == reg]
            ax.plot(sub["time"], sub[var], label=reg, color=regime_colors[reg])
        ax.set_xlabel("Time")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures.append(fig)

    return figures


def make_bar_figures(baseline_df):
    plot_order = ["DS", "DA", "ES", "EA", "HY"]

    summary = (
        baseline_df
        .groupby("regime", as_index=False)
        .agg(
            loss_t1_mean=("loss_t1", "mean"),
            final_perf_mean=("final_perf_net", "mean"),
            auc_perf_mean=("auc_perf", "mean"),
            final_innovation_mean=("final_innovation_stock", "mean"),
        )
        .set_index("regime")
        .loc[plot_order]
        .reset_index()
    )

    ref_row = summary[summary["regime"] == "DS"].iloc[0]

    for col in [
        "final_perf_mean",
        "auc_perf_mean",
        "final_innovation_mean",
    ]:
        summary[col + "_pct_vs_DS"] = (
            (summary[col] - ref_row[col]) / ref_row[col] * 100
        )

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    regime_colors = {reg: default_colors[i] for i, reg in enumerate(plot_order)}
    bar_colors = [regime_colors[r] for r in summary["regime"]]

    figures = []

    for ycol, title, ylabel in [
        ("final_perf_mean_pct_vs_DS", "Final recovery score vs DS", "Percent difference from DS"),
        ("auc_perf_mean_pct_vs_DS", "Cumulative recovery score vs DS", "Percent difference from DS"),
        ("final_innovation_mean_pct_vs_DS", "Innovation stock vs DS", "Percent difference from DS"),
    ]:
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ax.bar(summary["regime"], summary[ycol], color=bar_colors)
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Regime")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        figures.append(fig)

    return figures


def make_joint_tradeoff_figure(baseline_df, simulation_length):
    plot_order = ["DS", "DA", "ES", "EA", "HY"]

    regime_labels = {
        "DS": "CR",        # Centralized restorative
        "DA": "CA",        # Centralized adaptive
        "ES": "ER",        # Emergent restorative
        "EA": "EA",        # Emergent adaptive
        "HY": "CR → CA",   # Current hybrid sequence in the model
    }

    summary = (
        baseline_df
        .groupby("regime", as_index=False)
        .agg(
            perf_t1_mean=("perf_t1", "mean"),
            final_innovation_mean=("final_innovation_stock", "mean"),
        )
        .set_index("regime")
        .loc[plot_order]
        .reset_index()
    )

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    regime_colors = {reg: default_colors[i] for i, reg in enumerate(plot_order)}

    fig, ax = plt.subplots(figsize=(5.8, 4.2))

    fig.patch.set_facecolor("#f7f7f7")
    ax.set_facecolor("#f7f7f7")

    x_vals = summary["perf_t1_mean"].values
    y_vals = summary["final_innovation_mean"].values

    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    x_range = max(x_max - x_min, 0.05)
    y_range = max(y_max - y_min, 10)

    ax.set_xlim(max(0, x_min - 0.35 * x_range), min(1.05, x_max + 0.35 * x_range))
    ax.set_ylim(max(0, y_min - 0.35 * y_range), y_max + 0.45 * y_range)

    label_offsets = {
        "DS": (0, 12),
        "DA": (0, 12),
        "ES": (0, -16),
        "EA": (0, 12),
        "HY": (0, 12),
    }

    for _, row in summary.iterrows():
        reg = row["regime"]
        x = row["perf_t1_mean"]
        y = row["final_innovation_mean"]

        ax.scatter(
            x,
            y,
            s=190,
            color=regime_colors[reg],
            alpha=0.9,
            edgecolors="white",
            linewidths=1.2,
            zorder=3
        )

        dx, dy = label_offsets.get(reg, (0, 10))
        ax.annotate(
            regime_labels.get(reg, reg),
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#222222"
        )

    ax.set_title("Short-term Restoration vs. Long-term Innovation", fontsize=11, pad=10)
    ax.set_xlabel("Short-term restoration: t=1")
    ax.set_ylabel(f"Long-term innovation: t={simulation_length}")

    ax.grid(True, color="#d6d6d6", alpha=0.6, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b0b0b0")
    ax.spines["bottom"].set_color("#b0b0b0")

    fig.tight_layout()
    return fig


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Network Recovery Regimes ABM", layout="wide")

st.markdown("""
<style>
/* Move title higher */
.block-container {
    padding-top: 2.8rem;
}

/* Sidebar labels */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p {
    color: #0b3d91 !important;
    font-weight: 600;
}

/* Make sidebar buttons full width */
[data-testid="stSidebar"] .stButton,
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
}

/* Primary run button */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    color: #f0f2f6 !important;
}

/* Force all inner text/icons inside the primary button */
[data-testid="stSidebar"] .stButton > button[kind="primary"] * {
    color: #f0f2f6 !important;
    fill: #f0f2f6 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Network Recovery Regimes ABM")
st.caption("Version 1 Streamlit interface for the ABM model")

with st.sidebar:
    st.header("Scenario inputs")

    shock_type = st.selectbox(
        "Shock type",
        options=["targeted_hub", "random", "clustered"],
        index=["targeted_hub", "random", "clustered"].index(PARAMS["shock_type"])
    )

    shock_severity = st.slider(
        "Shock severity",
        min_value=0.05,
        max_value=0.60,
        value=0.50,
        step=0.05
    )

    network_type_options = [
        "modular_small_world",
        "random_erdos_renyi",
        "small_world",
        "scale_free",
        "core_periphery",
    ]
    network_type = st.selectbox(
        "Network type",
        options=network_type_options,
        index=network_type_options.index(PARAMS["network_type"])
    )

    network_size = st.slider(
        "Network size",
        min_value=50,
        max_value=500,
        value=200,
        step=10
    )

    avg_degree = st.slider(
        "Average degree",
        min_value=3,
        max_value=10,
        value=5,
        step=1
    )

    if network_type == "modular_small_world":
        community_count = st.slider(
            "Number of communities",
            min_value=2,
            max_value=12,
            value=5,
            step=1
        )
    else:
        community_count = int(PARAMS["C"])

    if network_type in ["modular_small_world", "core_periphery"]:
        bridge_prob = st.slider(
            "Bridge probability",
            min_value=0.00,
            max_value=0.10,
            value=float(PARAMS["bridge_prob"]),
            step=0.01
        )
    else:
        bridge_prob = float(PARAMS["bridge_prob"])

    if network_type in ["modular_small_world", "small_world"]:
        rewire_prob = st.slider(
            "Rewire probability",
            min_value=0.00,
            max_value=0.30,
            value=float(PARAMS["rewire_prob"]),
            step=0.01
        )
    else:
        rewire_prob = float(PARAMS["rewire_prob"])

    evaluation_recovery_threshold = st.slider(
        "Evaluation recovery threshold",
        min_value=0.50,
        max_value=1.00,
        value=0.85,
        step=0.05
    )

    hybrid_switch_perf = st.slider(
        "HY switch threshold",
        min_value=0.40,
        max_value=1.00,
        value=0.70,
        step=0.02
    )

    simulation_length = st.slider(
        "Steps",
        min_value=10,
        max_value=500,
        value=30,
        step=10
    )

    seed_count = st.slider(
        "Seeds",
        min_value=1,
        max_value=100,
        value=10,
        step=1
    )

    results_table_choice = st.selectbox(
        "Results table",
        options=["compact", "full", "both"],
        index=0
    )

    plot_display_choice = st.selectbox(
        "Plots",
        options=["all", "trajectory", "bars", "none"],
        index=0
    )

    run_clicked = st.button(
        "Run model comparison",
        type="primary",
        use_container_width=True
    )

if run_clicked:
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    baseline_df, baseline_traj_df, compact_pretty, full_pretty = run_comparison(
        shock_type=shock_type,
        shock_severity=shock_severity,
        network_type=network_type,
        network_size=network_size,
        simulation_length=simulation_length,
        seed_count=seed_count,
        avg_degree=avg_degree,
        evaluation_recovery_threshold=evaluation_recovery_threshold,
        hybrid_switch_perf=hybrid_switch_perf,
        community_count=community_count,
        bridge_prob=bridge_prob,
        rewire_prob=rewire_prob,
        progress_bar=progress_bar,
        status_placeholder=status_placeholder,
    )

    progress_bar.progress(1.0)
    status_placeholder.success("Model comparison finished.")

    st.subheader("Scenario summary")
    summary_rows = {
        "Shock type": shock_type,
        "Shock severity": f"{shock_severity:.2f}",
        "Network type": network_type,
        "Network size": network_size,
        "Steps": simulation_length,
        "Seeds": seed_count,
        "Average degree": avg_degree,
        "Evaluation threshold": f"{evaluation_recovery_threshold:.2f}",
        "HY switch threshold": f"{hybrid_switch_perf:.2f}",
    }

    if network_type == "modular_small_world":
        summary_rows["Number of communities"] = community_count

    if network_type in ["modular_small_world", "core_periphery"]:
        summary_rows["Bridge probability"] = f"{bridge_prob:.2f}"

    if network_type in ["modular_small_world", "small_world"]:
        summary_rows["Rewire probability"] = f"{rewire_prob:.2f}"

    st.table(pd.DataFrame(summary_rows.items(), columns=["Setting", "Value"]))

    if plot_display_choice != "none":
        st.subheader("Plots")

    if plot_display_choice in ["all", "trajectory"]:
        traj_figs = make_trajectory_figures(baseline_traj_df)
        col1, col2, col3 = st.columns(3)
        for col, fig in zip([col1, col2, col3], traj_figs):
            with col:
                st.pyplot(fig, use_container_width=False)

    if plot_display_choice in ["all", "bars"]:
        bar_figs = make_bar_figures(baseline_df)
        col1, col2, col3 = st.columns(3)
        for col, fig in zip([col1, col2, col3], bar_figs):
            with col:
                st.pyplot(fig, use_container_width=False)

    if plot_display_choice == "none":
        st.info("No trajectory or bar plots selected.")

    st.subheader("Results")

    if results_table_choice == "compact":
        st.dataframe(compact_pretty, use_container_width=True)
    elif results_table_choice == "full":
        st.dataframe(full_pretty, use_container_width=True)
    else:
        st.markdown("**Compact results table**")
        st.dataframe(compact_pretty, use_container_width=True)
        st.markdown("**Full results table**")
        st.dataframe(full_pretty, use_container_width=True)

    st.subheader("Short-term Restoration vs. Long-term Innovation")

    left_col, center_col, right_col = st.columns([1, 2, 1])
    with center_col:
        joint_fig = make_joint_tradeoff_figure(baseline_df, simulation_length)
        st.pyplot(joint_fig, use_container_width=False)

else:
    st.info("Set parameters in the sidebar and click 'Run model comparison'.")