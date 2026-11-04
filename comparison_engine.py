"""Headless, reproducible comparison engine for Streamlit and local tests.

All scenarios sharing a seed receive common network, shock and initial knowledge.
Runs return three ordinary pandas dataframes: run-level outcomes, structural
recovery trajectories and knowledge trajectories.
"""
import numpy as np
import pandas as pd
from model_core import RecoveryModel
from chain_experiments import BASE_MODES

def compare_scenarios(params_tuple, scenarios, n_seeds):
    params = dict(params_tuple)
    summary_rows = []
    recovery_rows = []
    innovation_rows = []
    # Each mode/seed pair uses the same initial network, shock, and knowledge.
    for seed in range(1, n_seeds + 1):
        for label, reg, first, second in scenarios:
            p = {**params, "seed": seed, "mode": reg,
                 "chain_first": first, "chain_second": second}
            model = RecoveryModel(p)
            while model.running:
                model.step()
            r = model.innovation_results
            events = r["post_event_counts"]
            collabs = events["collab_attempts"]
            summary_rows.append({
                "seed": seed, "mode": label,
                "first_stage": first, "second_stage": second,
                "is_chain": reg == "CHAIN",
                # Equal-length pre-shock, post-recovery and paused-intact
                # innovation windows; changes are within-window differences.
                "knowledge_gain": r["post_knowledge_gain"],
                "pre_knowledge_gain": r["pre_knowledge_gain"],
                "intact_knowledge_gain": r["intact_paused_knowledge_gain"],
                "knowledge_gap_vs_pre": r["R_K"],
                "knowledge_gap_pct_vs_pre": (
                    100 * r["R_K"] / r["pre_knowledge_gain"]
                    if r["pre_knowledge_gain"] else np.nan
                ),
                "knowledge_gap_vs_intact": r["R_K_vs_intact_paused"],
                "knowledge_gap_pct_vs_intact": (
                    100 * r["R_K_vs_intact_paused"] / r["intact_paused_knowledge_gain"]
                    if r["intact_paused_knowledge_gain"] else np.nan
                ),
                "frontier_gain": r["post_frontier_gain"],
                "pre_frontier_gain": r["pre_frontier_gain"],
                "intact_frontier_gain": r["intact_paused_frontier_gain"],
                "frontier_gap_vs_pre": r["R_F"],
                "frontier_gap_vs_intact": r["R_F_vs_intact_paused"],
                "gini_change": r["post_gini_change"],
                "pre_gini_change": r["pre_gini_change"],
                "intact_gini_change": r["intact_paused_gini_change"],
                "gini_gap_vs_pre": r["R_G"],
                "gini_gap_vs_intact": r["R_G_vs_intact_paused"],
                "effective_domains_change": r["post_effective_domains_change"],
                "pre_effective_domains_change": r["pre_effective_domains_change"],
                "intact_effective_domains_change":
                    r["intact_paused_effective_domains_change"],
                "effective_domains_gap_vs_pre": r["R_D"],
                "effective_domains_gap_vs_intact": r["R_D_vs_intact_paused"],
                "final_effective_domains":
                    r["snapshots"]["post_end"]["mean_effective_domains"],
                "original_ties_retained": model.active_pre_shock_ties,
                "original_ties_total": model.G_pre.number_of_edges(),
                "initial_cross_community_ties": sum(
                    model.community_ids[u] != model.community_ids[v]
                    for u, v in model.G_pre.edges()
                ),
                "new_ties": model.new_ties_count,
                "cross_community_ties": r["recovered_cross_community_edges"],
                "active_firms": r["active_firms_at_evaluation"],
                "switch_period": model.switch_at,
                "cross_community_collab_share": (
                    events["collab_attempts_cross_community"] / collabs
                    if collabs else np.nan
                ),
                "new_tie_collab_share": (
                    events["collab_attempts_new_ties"] / collabs
                    if collabs else np.nan
                ),
                "recovery_cost": model.recovery_spend,
            })
            rec = model.datacollector.get_model_vars_dataframe().copy()
            rec["seed"] = seed
            rec["mode"] = label
            recovery_rows.append(rec)
            for record in model.innovation_trajectories:
                innovation_rows.append({"seed": seed, "mode": label, **record})
    summary = pd.DataFrame(summary_rows)
    benchmarks = summary[summary["mode"].isin(BASE_MODES)][
        ["seed", "mode", "knowledge_gain"]
    ].rename(columns={"mode": "first_stage",
                      "knowledge_gain": "first_stage_knowledge_gain"})
    summary = summary.merge(benchmarks, on=["seed", "first_stage"],
                            how="left", validate="many_to_one")
    summary["knowledge_gain_pct_vs_first_stage"] = np.where(
        summary["first_stage_knowledge_gain"] != 0,
        100 * (summary["knowledge_gain"] - summary["first_stage_knowledge_gain"])
        / summary["first_stage_knowledge_gain"],
        np.nan,
    )
    return (summary, pd.concat(recovery_rows, ignore_index=True),
            pd.DataFrame(innovation_rows))


