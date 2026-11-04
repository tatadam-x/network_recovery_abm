"""Streamlit interface for the Network Recovery ABM (coauthor release).
Run: python -m streamlit run streamlit_app.py

The structural 'perf_net' index is deliberately omitted from reported outcomes.
Innovation is evaluated only after structural recovery.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from model_core import PARAMS
from comparison_engine import compare_scenarios
from benchmark_reporting import (prepare_benchmark_trajectories,
                                 check_summary_alignment)
from chain_experiments import (all_chain_scenarios, one_chain_scenarios,
                               priority_chain_scenarios, display_label)

MODE_NAMES = {
    "DS": "Directed Stabilization",
    "DA": "Directed Adaptation",
    "ES": "Emergent Stabilization",
    "EA": "Emergent Adaptation",
    "CHAIN": "Custom recovery chain",
    "HY": "Hybrid alias (DS → EA)",
}
BASE_MODES = ("DS", "DA", "ES", "EA")


@st.cache_data(show_spinner=False)
def cached_comparison(params_tuple, scenarios, n_seeds):
    """Memoize exact model scenarios for repeat views in Streamlit."""
    return compare_scenarios(params_tuple, scenarios, n_seeds)


def plot_recovery_trajectory(recovery, metric, label,
                             baseline=None, baseline_label=None):
    """Show shock-to-recovery structure and its pre-shock reference."""
    means = recovery.groupby(["mode", "time"], as_index=False)[metric].mean()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for key, group in means.groupby("mode", sort=False):
        ax.plot(group["time"], group[metric], label=display_label(key), lw=1.9)
    if baseline is not None:
        ax.axhline(baseline, color="black", linestyle="--",
                   linewidth=1.4, label=baseline_label or "Pre-shock reference")
    ax.set(xlabel="Recovery period (0 = immediately after shock)", ylabel=label)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    return fig


KNOWLEDGE_PLOTS = {
    "knowledge": ("Cumulative knowledge produced", "Knowledge produced since window start"),
    "frontier": ("Frontier advancement", "Change in mean knowledge frontier"),
    "gini": ("Knowledge inequality", "Change in Gini coefficient"),
    "diversity": ("Knowledge diversity", "Change in mean effective knowledge dimensions"),
}


def plot_benchmark_trajectory(trajectories, outcome):
    """Directly compare within-window pre, recovered and intact trajectories."""
    title, ylabel = KNOWLEDGE_PLOTS[outcome]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    # Deduplicate references: every mode shares the same seed's pre and intact
    # windows; otherwise the visual average would overweight duplicated rows.
    ref = trajectories.drop_duplicates(["seed", "period"])
    for col, label, style, color in [
        (f"pre_{outcome}_change", "Pre-shock benchmark", "--", "#262626"),
        (f"intact_{outcome}_change", "Paused intact network", ":", "#858585"),
    ]:
        avg = ref.groupby("period", sort=True)[col].mean()
        ax.plot(avg.index, avg.values, label=label,
                ls=style, color=color, linewidth=2.3)
    avg_post = (trajectories
                .groupby(["mode", "period"], as_index=False)
                [f"post_{outcome}_change"].mean())
    for key, group in avg_post.groupby("mode", sort=False):
        ax.plot(group["period"], group[f"post_{outcome}_change"],
                label=display_label(key), lw=1.8, alpha=0.9)
    ax.axhline(0, color="#BBBBBB", lw=0.6)
    ax.set(title=title, xlabel="Period within each equal-length innovation window",
           ylabel=ylabel)
    ax.grid(alpha=0.18)
    ax.legend(ncol=2, fontsize=7.6)
    fig.tight_layout()
    return fig


def plot_benchmark_gap(trajectories, outcome, reference):
    """Paired seed differences in cumulative changes, at each window period."""
    title, ylabel = KNOWLEDGE_PLOTS[outcome]
    column = f"{outcome}_gap_vs_{reference}"
    means = trajectories.groupby(["mode", "period"], as_index=False)[column].mean()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for key, group in means.groupby("mode", sort=False):
        ax.plot(group["period"], group[column], lw=1.9,
                label=display_label(key))
    ax.axhline(0, color="#333333", linewidth=1.2, linestyle="--")
    ref_name = ("pre-shock" if reference == "pre"
                else "paused intact-network")
    ax.set(title=f"{title}: difference from {ref_name}",
           xlabel="Period within the innovation window",
           ylabel=f"Post-recovery minus {ref_name} change")
    ax.grid(alpha=0.18)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def render_figure(fig):
    try:
        st.pyplot(fig, use_container_width=True)
    finally:
        plt.close(fig)


def download_frame(name, frame):
    st.download_button(
        label=f"Download {name} (CSV)",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=f"{name}.csv",
        mime="text/csv",
    )


st.set_page_config(page_title="Network Recovery | Research ABM", layout="wide",
                   page_icon="🔗")
# Keep the main heading and sidebar's first heading vertically aligned on desktop.
# The sidebar has its own header area, so matching the two content paddings
# would push the sidebar's controls lower than the page title.
st.markdown(
    """
    <style>
    @media (min-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            padding-top: 2.25rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding-top: 0.25rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Network recovery & collective innovation")
st.caption("An agent-based research model of interorganizational network recovery, governance and collective learning.")
st.markdown("Compare four recovery mechanisms and sequential combinations under matched shocks and starting networks.")

with st.sidebar:
    st.subheader("Experimental design")
    shock_type = st.selectbox("Shock type",
        ["random", "targeted_hub", "clustered"], index=0)
    shock_severity = st.slider("Share of firms shocked", 0.0, 1.0, 0.45, 0.05)
    network_type = st.selectbox("Initial network",
        ["modular_small_world", "random_erdos_renyi", "small_world",
         "scale_free", "core_periphery"])
    N = st.slider("Number of firms", 20, 200, 100, 10)
    T = st.slider("Structural recovery periods", 2, 60, 10)
    innovation_horizon = st.slider("Pre/post innovation window (each)", 4, 40, 15)
    st.subheader("Recovery comparison")
    comparison_mode = st.radio(
        "Comparison set",
        (
            "Recommended chains + four recovery modes",
            "One custom chain + four recovery modes",
            "All 16 ordered chains + four recovery modes",
        ),
        index=0,
        help=(
            "Recommended compares DS→EA, ES→EA, DA→EA, and EA→DA together. "
            "All chains includes every ordered pair, including four same-mode checks."
        ),
    )
    is_custom = comparison_mode.startswith("One custom")
    is_all = comparison_mode.startswith("All")
    chain_first, chain_second = "DS", "EA"
    chain_rule, chain_time, chain_threshold = "time", max(1, T // 2), 0.80

    if is_custom:
        chain_first = st.selectbox(
            "First stage", BASE_MODES, index=0,
            format_func=lambda x: f"{x} — {MODE_NAMES[x]}"
        )
        chain_second = st.selectbox(
            "Second stage", BASE_MODES, index=3,
            format_func=lambda x: f"{x} — {MODE_NAMES[x]}"
        )
    else:
        st.caption(
            "Compare all four foundational modes with "
            + ("16 ordered chains (including four same-mode checks)."
               if is_all else "DS → EA, ES → EA, DA → EA, and EA → DA.")
        )
    chain_rule = st.selectbox(
        "Chain transition trigger", ("time", "connectivity", "restoration"),
        help="Time: switch after a set number of completed periods. "
             "Connectivity: switch when the largest component reaches a threshold. "
             "Restoration: switch when the share of original ties restored reaches a threshold."
    )
    if chain_rule == "time":
        chain_time = st.slider("Switch after recovery period", 1, T-1, max(1, T // 2))
    else:
        chain_threshold = st.slider("Switch threshold", 0.05, 1.0, 0.80, 0.05)
    with st.expander("Advanced settings", expanded=False):
        st.caption("These controls are held constant across recovery modes.")
        n_seeds = st.slider(
            "Paired random seeds", 1, 50, 5,
            help="Number of matched simulation replications. Each seed generates the same "
                 "starting network, shock, and knowledge across all modes/chains. "
                 "Use 5 for an exploratory run and 20–30 for a more stable comparison."
        )
        avg_degree = st.slider("Initial average degree", 2, 12, 4, 2)
        knowledge_init_rule = st.selectbox(
            "Knowledge initialization",
            ("dirichlet", "gaussian"),
            help="Dirichlet: structured community specialization. "
                 "Gaussian: clipped-Gaussian initialization for sensitivity checks."
        )
        knowledge_production_rule = st.selectbox(
            "Knowledge production increment",
            ("bounded_surplus", "bounded_ces", "ces_stock"),
            help="Bounded surplus: Scott's cap plus complementarity premium in collaboration. "
                 "Bounded CES: capped increment without complementarity premium. CES stock: original unbounded increment."
        )
        C = st.slider("Number of communities", 2,
                      PARAMS["K"] if knowledge_init_rule == "dirichlet" else 8,
                      4)
        alpha_between = st.slider(
            "Dirichlet community concentration αB", 2., 200., 40., 2.,
            help="Higher means community references are closer to their expected focal profiles."
        )
        alpha_within = st.slider(
            "Dirichlet within-community concentration αW", 2., 200., 40., 2.,
            help="Higher means firms in a community have more similar profiles."
        )
        bounded_max_increment = st.slider(
            "Bounded innovation increment cap", .02, .80, .24, .02,
        )
        bounded_initial_fraction = st.slider(
            "Initial fraction of bounded cap", .05, 1., .25, .05,
        )
        collab_surplus_strength = st.slider(
            "Collaboration complementarity strength", 0.0, 6.0, 2.0, 0.25,
            help="Only affects successful collaborations under bounded_surplus. "
                 "Higher values increase the premium for complementary knowledge profiles; "
                 "the increment remains capped."
        )
        ea_partner_rule = st.selectbox(
            "EA new-partner selection",
            ("uniform", "community_gap"),
            help="Community gap favors partners from underrepresented communities; "
                 "uniform uses random selection among eligible partners."
        )
        sigma_within = st.slider(
            "Gaussian within-community SD (legacy)", .0, .4, .05, .01
        )
        knowledge_scale = st.slider(
            "Legacy CES production scale", .001, .05, .01, .001
        )
        bridge_prob = st.slider("Initial cross-community edge probability",
                                0.0, 0.10, 0.01, 0.005)
        rewire_prob = st.slider("Initial small-world rewiring probability",
                                0.0, 0.50, 0.05, 0.01)
        slack_init = st.slider("Initial recovery resources per firm", 1.0, 8.0, 3.0, 0.5)
        collaboration_prob = st.slider("Probability of collaboration",
                                       0.0, 1.0, 0.5, 0.05)
        delta_between = st.slider("Between-community knowledge differentiation",
                                   0.0, 1.0, 0.70, 0.05)
        decomposability = st.slider("Innovation task decomposability",
                                     0.0, 1.0, 0.75, 0.05)
        rewiring_budget = st.slider("Directed new-tie cap per period", 1, 10, 3)
        ea_exploration_prob = st.slider("EA distant exploration probability",
                                        0.0, 1.0, 0.35, 0.05)
        p_inactive = st.slider("Firm inactivation probability",
                                0.0, 1.0, 0.50, 0.05)
        reactivation_prob = st.slider("Firm reactivation probability",
                                      0.05, 1.0, 0.40, 0.05)
        include_uninterrupted = st.checkbox(
            "Include uninterrupted no-shock diagnostic", False
        )
    run = st.button("Run paired comparison", type="primary", use_container_width=True)

params = {
    **PARAMS,
    "N": N, "T": T, "innovation_horizon": innovation_horizon,
    "shock_type": shock_type, "shock_severity": shock_severity,
    "network_type": network_type,
    "avg_degree": avg_degree, "C": C,
    "bridge_prob": bridge_prob, "rewire_prob": rewire_prob,
    "slack_init": slack_init, "collaboration_prob": collaboration_prob,
    "delta_between": delta_between, "decomposability": decomposability,
    "knowledge_init_rule": knowledge_init_rule,
    "knowledge_production_rule": knowledge_production_rule,
    "alpha_between": alpha_between, "alpha_within": alpha_within,
    "bounded_max_increment": bounded_max_increment,
    "bounded_initial_fraction": bounded_initial_fraction,
    "collab_surplus_strength": collab_surplus_strength,
    "ea_partner_rule": ea_partner_rule,
    "sigma_within": sigma_within, "knowledge_scale": knowledge_scale,
    "rewiring_budget": rewiring_budget,
    "ea_exploration_prob": ea_exploration_prob,
    "p_inactive": p_inactive, "reactivation_prob": reactivation_prob,
    "include_uninterrupted_reference": include_uninterrupted,
    "chain_first": chain_first, "chain_second": chain_second,
    "chain_switch_rule": chain_rule, "chain_switch_time": chain_time,
    "chain_switch_threshold": chain_threshold,
}

if is_custom:
    scenarios = one_chain_scenarios(chain_first, chain_second)
elif is_all:
    scenarios = all_chain_scenarios()
else:
    scenarios = priority_chain_scenarios()

if len(scenarios) * n_seeds >= 100:
    st.sidebar.caption(
        f"{len(scenarios)} specifications × {n_seeds} seeds = "
        f"{len(scenarios) * n_seeds} simulations. "
        "The exhaustive comparison can take longer on hosted infrastructure."
    )
else:
    st.sidebar.caption(
        f"Run size: {len(scenarios)} specifications × {n_seeds} paired seeds "
        f"= {len(scenarios) * n_seeds} simulations."
    )
if run:
    with st.spinner("Running paired recovery and knowledge experiments..."):
        st.session_state["network_recovery_data"] = cached_comparison(
            tuple(sorted(params.items())), scenarios, n_seeds
        )
        st.session_state["scenario_description"] = (
            f"{len(scenarios)} specifications × {n_seeds} paired seeds; "
            f"N={N}; T={T}; innovation window={innovation_horizon}; "
            f"shock={shock_type}, severity={shock_severity:.2f}; "
            f"transition rule={chain_rule}; "
            f"knowledge={knowledge_init_rule}+{knowledge_production_rule}"
        )
        st.session_state["scenario_is_custom"] = is_custom
        st.session_state["scenario_mode"] = comparison_mode
        st.session_state["scenario_params"] = params
        st.session_state["scenario_scenarios"] = scenarios
        st.session_state["scenario_seeds"] = n_seeds
        st.session_state["scenario_chain_label"] = f"{chain_first} → {chain_second}"

if "network_recovery_data" in st.session_state:
    summary, recovery, innovation = st.session_state["network_recovery_data"]
    benchmark_curves = prepare_benchmark_trajectories(innovation)
    # Fail visibly if the published outcome table contradicts the plotted data.
    check_summary_alignment(summary, benchmark_curves)
    st.caption(st.session_state["scenario_description"])
    st.subheader("Benchmark-adjusted collective innovation")
    st.markdown(
        "Each outcome is measured as the **change during its innovation window** "
        "(using the window length selected in the sidebar). "
        "The pre-shock benchmark, post-recovery evaluation, and paused intact-network "
        "reference have equal durations. Knowledge is held constant during structural "
        "recovery in both the recovered and the intact reference networks."
    )
    st.caption(
        "Positive differences in knowledge production, frontier advancement, and knowledge "
        "diversity indicate higher post-recovery changes. A positive Gini difference means "
        "knowledge inequality increased more (or decreased less) than in the reference window."
    )

    aggregate_columns = {
        "knowledge_gain": "post_knowledge_gain",
        "pre_knowledge_gain": "pre_knowledge_gain",
        "intact_knowledge_gain": "intact_knowledge_gain",
        "knowledge_gap_vs_pre": "knowledge_gap_vs_pre",
        "knowledge_gap_pct_vs_pre": "knowledge_gap_pct_vs_pre",
        "knowledge_gap_vs_intact": "knowledge_gap_vs_intact",
        "knowledge_gap_pct_vs_intact": "knowledge_gap_pct_vs_intact",
        "knowledge_gain_pct_vs_first_stage": "knowledge_gain_pct_vs_first_stage",
        "frontier_gain": "post_frontier_gain",
        "pre_frontier_gain": "pre_frontier_gain",
        "intact_frontier_gain": "intact_frontier_gain",
        "frontier_gap_vs_pre": "frontier_gap_vs_pre",
        "frontier_gap_vs_intact": "frontier_gap_vs_intact",
        "gini_change": "post_gini_change",
        "pre_gini_change": "pre_gini_change",
        "intact_gini_change": "intact_gini_change",
        "gini_gap_vs_pre": "gini_gap_vs_pre",
        "gini_gap_vs_intact": "gini_gap_vs_intact",
        "effective_domains_change": "post_diversity_change",
        "pre_effective_domains_change": "pre_diversity_change",
        "intact_effective_domains_change": "intact_diversity_change",
        "effective_domains_gap_vs_pre": "diversity_gap_vs_pre",
        "effective_domains_gap_vs_intact": "diversity_gap_vs_intact",
        "original_ties_retained": "original_ties",
        "original_ties_total": "original_ties_total",
        "initial_cross_community_ties": "initial_cross_community_ties",
        "new_ties": "new_ties",
        "cross_community_ties": "cross_community_ties",
        "cross_community_collab_share": "cross_community_collaboration",
        "recovery_cost": "recovery_cost",
    }
    aggregate = summary.groupby("mode", as_index=False).agg(
        **{out: (column, "mean") for column, out in aggregate_columns.items()}
    )
    aggregate["display_mode"] = aggregate["mode"].map(display_label)

    knowledge_table = aggregate[[
        "display_mode", "pre_knowledge_gain", "post_knowledge_gain",
        "intact_knowledge_gain", "knowledge_gap_vs_pre",
        "knowledge_gap_pct_vs_pre", "knowledge_gap_pct_vs_intact",
    ]].rename(columns={
        "display_mode": "Recovery mode or chain",
        "pre_knowledge_gain": "Pre-shock production",
        "post_knowledge_gain": "Post-recovery production",
        "intact_knowledge_gain": "Paused-intact production",
        "knowledge_gap_vs_pre": "Post − pre (units)",
        "knowledge_gap_pct_vs_pre": "Post vs pre (%)",
        "knowledge_gap_pct_vs_intact": "Post vs intact (%)",
    })
    st.markdown("**Aggregate knowledge production**")
    st.dataframe(knowledge_table.round(3), use_container_width=True,
                 hide_index=True)
    st.caption(
        "Production is the total knowledge accumulated during the stated window. "
        "Both percentage columns use each seed's own matched reference as denominator, "
        "then average the paired percentages across seeds. "
        "The pre-shock window occurs earlier in knowledge accumulation; its comparison "
        "can reflect endogenous knowledge growth. The paused intact-network reference "
        "controls for the starting knowledge stock and isolates the consequences "
        "of the recovered network and remaining firm inactivity."
    )

    other_table = aggregate[[
        "display_mode", "pre_frontier_gain", "post_frontier_gain",
        "intact_frontier_gain", "frontier_gap_vs_pre", "frontier_gap_vs_intact",
        "gini_gap_vs_pre", "gini_gap_vs_intact",
        "diversity_gap_vs_pre", "diversity_gap_vs_intact",
    ]].rename(columns={
        "display_mode": "Recovery mode or chain",
        "pre_frontier_gain": "Frontier pre",
        "post_frontier_gain": "Frontier post",
        "intact_frontier_gain": "Frontier intact",
        "frontier_gap_vs_pre": "Frontier post − pre",
        "frontier_gap_vs_intact": "Frontier post − intact",
        "gini_gap_vs_pre": "Gini post − pre",
        "gini_gap_vs_intact": "Gini post − intact",
        "diversity_gap_vs_pre": "Diversity post − pre",
        "diversity_gap_vs_intact": "Diversity post − intact",
    })
    st.markdown("**Frontier advancement, inequality, and knowledge diversity**")
    st.dataframe(other_table.round(4), use_container_width=True,
                 hide_index=True)
    st.caption(
        "Frontier measures the mean of the highest knowledge stock in each domain. "
        "Gini and diversity entries compare *within-window changes* (end minus start)."
    )

    if not st.session_state.get("scenario_is_custom"):
        options = list(summary["mode"].unique())
        preferred = ["DS", "DA", "EA", "DS->EA", "DA->EA", "EA->DA"]
        selected = st.multiselect(
            "Show trajectories for",
            options=options,
            default=[v for v in preferred if v in options],
            format_func=display_label,
            help="Charts show selected modes. Reference curves use each seed exactly once. "
                 "Tables and exported results always include all modes.",
        )
    else:
        selected = list(summary["mode"].unique())

    if not selected:
        st.info("Select one or more modes or chains to display trajectories.")
    else:
        recovery_plot = recovery[recovery["mode"].isin(selected)]
        benchmark_plot = benchmark_curves[
            benchmark_curves["mode"].isin(selected)
        ]
        st.subheader("Innovation trajectories relative to both benchmarks")
        st.caption(
            "All curves start at zero: each shows the cumulative change since the "
            "beginning of its own equal-length window. Black dashed = intact pre-shock "
            "benchmark; gray dotted = paused intact-network reference; solid = "
            "post-recovery modes. Curves are averages over paired seeds."
        )
        for row_outcomes in (("knowledge", "frontier"), ("gini", "diversity")):
            left, right = st.columns(2)
            for col, outcome in zip((left, right), row_outcomes):
                with col:
                    render_figure(plot_benchmark_trajectory(benchmark_plot, outcome))
        st.subheader("Benchmark-adjusted innovation differences")
        st.caption(
            "At each period, the plotted gap subtracts the comparison-window change "
            "from the post-recovery change **within the same seed**. Zero indicates "
            "equal performance. These differences align exactly with the reported "
            "endpoint measures."
        )
        gap_reference_label = st.radio(
            "Compare post-recovery change with",
            ["Pre-shock benchmark", "Paused intact network"],
            horizontal=True, index=0,
            help="Pre-shock evaluates whether functioning recovers relative to the "
                 "prior benchmark; paused intact isolates network-reconstruction effects.",
        )
        gap_reference = ("pre" if gap_reference_label.startswith("Pre")
                         else "intact")
        for row_outcomes in (("knowledge", "frontier"), ("gini", "diversity")):
            left, right = st.columns(2)
            for col, outcome in zip((left, right), row_outcomes):
                with col:
                    render_figure(plot_benchmark_gap(
                        benchmark_plot, outcome, gap_reference
                    ))

        st.subheader("Structural recovery relative to the pre-shock network")
        st.caption(
            "Period 0 shows the network immediately after the shock. "
            "Dashed horizontal references show the intact pre-shock state. "
            "Innovation is suspended during this interval."
        )
        left, right = st.columns(2)
        with left:
            render_figure(plot_recovery_trajectory(
                recovery_plot, "restoration_share",
                "Share of original ties active", baseline=1.0,
                baseline_label="Pre-shock: 100% of original ties"
            ))
            render_figure(plot_recovery_trajectory(
                recovery_plot, "new_ties_count",
                "New and activated backup ties", baseline=0.0,
                baseline_label="Pre-shock: 0 new ties"
            ))
        with right:
            render_figure(plot_recovery_trajectory(
                recovery_plot, "lcc_share",
                "Largest component share", baseline=1.0,
                baseline_label="Pre-shock: full network"
            ))
            render_figure(plot_recovery_trajectory(
                recovery_plot, "cross_community_ties",
                "Active cross-community ties",
                baseline=float(summary.drop_duplicates("seed")[
                    "initial_cross_community_ties"].mean()),
                baseline_label="Mean pre-shock cross-community ties"
            ))

    st.subheader("Network reconstruction and collaboration")
    structural_table = aggregate[[
        "display_mode", "original_ties", "original_ties_total",
        "new_ties", "cross_community_ties",
        "cross_community_collaboration", "recovery_cost",
    ]].rename(columns={
        "display_mode": "Recovery mode or chain",
        "original_ties": "Original ties retained",
        "original_ties_total": "Original ties before shock",
        "new_ties": "New / backup ties",
        "cross_community_ties": "Cross-community ties",
        "cross_community_collaboration": "Cross-community collaborations (%)",
        "recovery_cost": "Recovery resources spent",
    })
    structural_table["Cross-community collaborations (%)"] *= 100
    st.dataframe(structural_table.round(2), use_container_width=True,
                 hide_index=True)

    if st.session_state.get("scenario_is_custom"):
        label = st.session_state.get("scenario_chain_label", "")
        coded = label.replace(" → ", "->")
        times = summary[summary["mode"] == coded]["switch_period"].dropna()
        st.caption(
            f"Custom chain {label}: observed mean transition period {times.mean():.1f}"
            if len(times) else f"Custom chain {label}: switching condition was not reached."
        )

    st.subheader("Export research data")
    st.caption(
        "The benchmark-aligned trajectories CSV contains every paired-seed "
        "pre-, post-, and intact-window change and its two gap measures. "
        "The innovation trajectories CSV also retains the raw levels for audit."
    )
    x, y, z = st.columns(3)
    with x:
        download_frame("scenario_summary", summary)
    with y:
        download_frame("comparison_means", aggregate)
    with z:
        download_frame("benchmark_trajectories", benchmark_curves)
    x, y = st.columns(2)
    with x:
        download_frame("recovery_trajectories", recovery)
    with y:
        download_frame("innovation_trajectories", innovation)

    with st.expander("Experiment settings and reproducibility", expanded=False):
        st.caption("Matched seeds: each seed initializes comparable networks, shocks and knowledge.")
        settings_payload = {
            "comparison_set": st.session_state["scenario_mode"],
            "paired_seeds": st.session_state["scenario_seeds"],
            "scenarios": [scenario[0] for scenario in
                          st.session_state["scenario_scenarios"]],
            "parameters": st.session_state["scenario_params"],
            "benchmark_convention": (
                "within-window cumulative changes; pre/post/intact aligned "
                "by innovation period; innovation paused during structural recovery"
            ),
        }
        st.json(settings_payload)
        st.download_button(
            "Download experiment settings (JSON)",
            data=json.dumps(settings_payload, indent=2),
            file_name="experiment_settings.json", mime="application/json",
        )
else:
    st.info("Set the experiment in the sidebar, then click **Run paired comparison**.")
