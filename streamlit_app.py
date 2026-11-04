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
from chain_experiments import (all_chain_scenarios, one_chain_scenarios,
                               priority_chain_scenarios, display_label)

REGIME_NAMES = {
    "DS": "Directed Stabilization",
    "DA": "Directed Adaptation",
    "ES": "Emergent Stabilization",
    "EA": "Emergent Adaptation",
    "CHAIN": "Custom recovery chain",
    "HY": "Hybrid alias (DS → EA)",
}
BASE_REGIMES = ("DS", "DA", "ES", "EA")


@st.cache_data(show_spinner=False)
def cached_comparison(params_tuple, scenarios, n_seeds):
    """Memoize exact model scenarios for repeat views in Streamlit."""
    return compare_scenarios(params_tuple, scenarios, n_seeds)


def plot_trajectories(recovery, innovation, metric, label, phase="recovery"):
    source = recovery if phase == "recovery" else innovation
    time_key = "time" if phase == "recovery" else "period"
    if phase == "innovation":
        source = source[source["phase"].eq("post")]
    means = (source.groupby(["regime", time_key], as_index=False)[metric].mean())
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for reg, group in means.groupby("regime", sort=False):
        ax.plot(group[time_key], group[metric], label=display_label(reg))
    ax.set(xlabel=("Recovery period" if phase == "recovery"
                   else "Post-recovery innovation period"), ylabel=label)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


def show_plot(recovery, innovation, metric, label, phase="recovery"):
    fig = plot_trajectories(recovery, innovation, metric, label, phase)
    try:
        st.pyplot(fig)
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
            "Recommended chains + four regimes",
            "One custom chain + four regimes",
            "All 16 ordered chains + four regimes",
        ),
        index=0,
        help=(
            "Recommended compares DS→EA, ES→EA, DA→EA, and EA→DA together. "
            "All chains includes every ordered pair, including four same-regime checks."
        ),
    )
    is_custom = comparison_mode.startswith("One custom")
    is_all = comparison_mode.startswith("All")
    chain_first, chain_second = "DS", "EA"
    chain_rule, chain_time, chain_threshold = "time", max(1, T // 2), 0.80

    if is_custom:
        chain_first = st.selectbox(
            "First stage", BASE_REGIMES, index=0,
            format_func=lambda x: f"{x} — {REGIME_NAMES[x]}"
        )
        chain_second = st.selectbox(
            "Second stage", BASE_REGIMES, index=3,
            format_func=lambda x: f"{x} — {REGIME_NAMES[x]}"
        )
    else:
        st.caption(
            "Compare all four foundational regimes with "
            + ("16 ordered chains (including four same-regime checks)."
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
        st.caption("These controls are held constant across recovery regimes.")
        n_seeds = st.slider(
            "Paired random seeds", 1, 50, 5,
            help="Number of matched simulation replications. Each seed generates the same "
                 "starting network, shock, and knowledge across all regimes/chains. "
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
    st.caption(st.session_state["scenario_description"])
    aggregate = summary.groupby("regime", as_index=False).agg(
        knowledge_gap_pct=("knowledge_gap_pct_vs_intact","mean"),
        knowledge_gain_pct_vs_first_stage=("knowledge_gain_pct_vs_first_stage","mean"),
        frontier_gap=("frontier_gap_vs_intact","mean"),
        gini_gap=("gini_gap_vs_intact","mean"),
        effective_domains_gap=("effective_domains_gap_vs_intact","mean"),
        final_effective_domains=("final_effective_domains","mean"),
        original_ties=("original_ties_retained","mean"),
        new_ties=("new_ties","mean"),
        cross_community_ties=("cross_community_ties","mean"),
        cross_community_collaboration=("cross_community_collab_share","mean"),
        recovery_cost=("recovery_cost","mean"),
    )
    st.subheader("Comparative outcomes")
    st.caption(
        "All regimes within a paired seed share the pre-shock starting conditions. "
        "Innovation pauses during recovery, including in the intact-network reference. "
        "Tables show means across paired seeds."
    )
    aggregate["regime"] = aggregate["regime"].map(display_label)
    innovation_table = aggregate[[
        "regime", "knowledge_gap_pct", "knowledge_gain_pct_vs_first_stage",
        "frontier_gap", "gini_gap", "effective_domains_gap",
    ]].rename(columns={
        "regime": "Regime or chain",
        "knowledge_gap_pct": "Knowledge gain vs intact (%)",
        "knowledge_gain_pct_vs_first_stage": "Knowledge gain vs first stage (%)",
        "frontier_gap": "Frontier change vs intact",
        "gini_gap": "Gini change vs intact",
        "effective_domains_gap": "Diversity change vs intact",
    })
    structural_table = aggregate[[
        "regime", "original_ties", "new_ties", "cross_community_ties",
        "cross_community_collaboration", "recovery_cost",
    ]].rename(columns={
        "regime": "Regime or chain",
        "original_ties": "Original ties retained",
        "new_ties": "New / backup ties",
        "cross_community_ties": "Cross-community ties",
        "cross_community_collaboration": "Cross-community collaborations (%)",
        "recovery_cost": "Recovery resources spent",
    })
    structural_table["Cross-community collaborations (%)"] *= 100
    st.markdown("**Knowledge outcomes**")
    st.dataframe(innovation_table.round(3), use_container_width=True, hide_index=True)
    st.caption(
        "Knowledge gain vs first stage is paired against the corresponding fixed regime. "
        "Positive diversity gap indicates more diversified firm knowledge than the intact reference."
    )
    st.markdown("**Network reconstruction and collaboration**")
    st.dataframe(structural_table.round(2), use_container_width=True, hide_index=True)
    if not st.session_state.get("scenario_is_custom"):
        options = list(summary["regime"].unique())
        preferred = ["DS", "DA", "EA", "DS->EA", "DA->EA", "EA->DA"]
        selected = st.multiselect(
            "Show trajectories for",
            options=options,
            default=[v for v in preferred if v in options],
            format_func=display_label,
            help="Charts show the selected specifications only; table and downloads include all.",
        )
        recovery_plot = recovery[recovery["regime"].isin(selected)]
        innovation_plot = innovation[innovation["regime"].isin(selected)]
    else:
        selected = list(summary["regime"].unique())
        recovery_plot, innovation_plot = recovery, innovation
    if not selected:
        st.info("Select one or more regimes or chains to display trajectories.")
    else:
        c1,c2 = st.columns(2)
        with c1:
            show_plot(recovery_plot, innovation_plot,
                    "active_pre_shock_ties", "Active original ties")
            show_plot(recovery_plot, innovation_plot,
                    "new_ties_count", "Novel / backup ties")
        with c2:
            show_plot(recovery_plot, innovation_plot,
                    "lcc_share", "Largest component share")
            show_plot(recovery_plot, innovation_plot,
                    "cross_community_ties", "Cross-community ties")
        st.subheader("Post-recovery innovation")
        c3,c4,c5=st.columns(3)
        with c3:
            show_plot(recovery_plot, innovation_plot,
                    "total_knowledge", "Total knowledge", "innovation")
        with c4:
            show_plot(recovery_plot, innovation_plot,
                    "mean_frontier", "Mean knowledge frontier", "innovation")
        with c5:
            show_plot(recovery_plot, innovation_plot,
                    "gini", "Knowledge inequality (Gini)", "innovation")
        show_plot(recovery_plot, innovation_plot,
                    "mean_effective_domains", "Mean effective knowledge dimensions",
                    "innovation")
    if st.session_state.get("scenario_is_custom"):
        label = st.session_state.get("scenario_chain_label", "")
        coded = label.replace(" → ", "->")
        times = summary[summary["regime"] == coded]["switch_period"].dropna()
        st.caption(
            f"Custom chain {label}: observed mean transition period {times.mean():.1f}"
            if len(times) else f"Custom chain {label}: switching condition was not reached."
        )
    else:
        st.caption("The full comparison includes four same-regime chains as invariance checks." 
                   if st.session_state.get("scenario_mode", "").startswith("All") 
                   else "Each chain is compared with the four fixed-regime benchmarks.")
    st.subheader("Export detailed results")
    x, y, z, w = st.columns(4)
    with x: download_frame("scenario_summary", summary)
    with y: download_frame("comparison_means", aggregate)
    with z: download_frame("recovery_trajectories", recovery)
    with w: download_frame("innovation_trajectories", innovation)

    with st.expander("Experiment settings and reproducibility", expanded=False):
        st.caption("Matched seeds: the same seed initializes comparable networks, shocks and knowledge.")
        st.json({
            "comparison_set": st.session_state["scenario_mode"],
            "paired_seeds": st.session_state["scenario_seeds"],
            "scenarios": [scenario[0] for scenario in st.session_state["scenario_scenarios"]],
            "parameters": st.session_state["scenario_params"],
        })
        st.download_button(
            "Download experiment settings (JSON)",
            data=json.dumps({
                "comparison_set": st.session_state["scenario_mode"],
                "paired_seeds": st.session_state["scenario_seeds"],
                "scenarios": [scenario[0] for scenario in st.session_state["scenario_scenarios"]],
                "parameters": st.session_state["scenario_params"],
            }, indent=2),
            file_name="experiment_settings.json", mime="application/json",
        )
else:
    st.info("Set the experiment in the sidebar, then click **Run paired comparison**.")
