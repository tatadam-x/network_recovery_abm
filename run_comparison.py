"""Headless counterpart to the Streamlit recovery comparison.

Examples
--------
python run_comparison.py --seeds 10
python run_comparison.py --mode all --seeds 10
"""
import argparse
import json
from pathlib import Path
import pandas as pd

from chain_experiments import priority_chain_scenarios, all_chain_scenarios
from comparison_engine import compare_scenarios
from benchmark_reporting import (prepare_benchmark_trajectories,
                                 check_summary_alignment)
from model_core import PARAMS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("recommended","all"),default="recommended")
    parser.add_argument("--seeds",type=int,default=10)
    parser.add_argument("--n-firms",type=int,default=100)
    parser.add_argument("--recovery-periods",type=int,default=10)
    parser.add_argument("--innovation-periods",type=int,default=15)
    parser.add_argument("--shock-type",choices=("random","targeted_hub","clustered"),
                        default="random")
    parser.add_argument("--shock-severity",type=float,default=.45)
    parser.add_argument("--switch-after",type=int,default=5)
    parser.add_argument("--output-dir",default="comparison_results")
    args=parser.parse_args()
    if args.seeds<1:
        parser.error("--seeds must be at least 1")
    if not (1<=args.switch_after<args.recovery_periods):
        parser.error("--switch-after must be between 1 and recovery-periods - 1")
    if not (0<=args.shock_severity<=1):
        parser.error("--shock-severity must be between 0 and 1")
    scenarios=(all_chain_scenarios() if args.mode=="all"
               else priority_chain_scenarios())
    params={
        **PARAMS,
        "N":args.n_firms,
        "T":args.recovery_periods,
        "innovation_horizon":args.innovation_periods,
        "shock_type":args.shock_type,
        "shock_severity":args.shock_severity,
        "chain_switch_rule":"time",
        "chain_switch_time":args.switch_after,
        # Selected default learning assumptions:
        "knowledge_init_rule":"dirichlet",
        "knowledge_production_rule":"bounded_surplus",
        "ea_partner_rule":"uniform",
    }
    print(f"Running {len(scenarios)} recovery specifications × {args.seeds} seeds...")
    runs,recovery,innovation=compare_scenarios(
        tuple(sorted(params.items())), scenarios, args.seeds
    )
    benchmarks = prepare_benchmark_trajectories(innovation)
    check_summary_alignment(runs, benchmarks)
    means = runs.groupby("mode", sort=False, as_index=False).agg(
        pre_knowledge_gain=("pre_knowledge_gain", "mean"),
        post_knowledge_gain=("knowledge_gain", "mean"),
        intact_knowledge_gain=("intact_knowledge_gain", "mean"),
        knowledge_gap_pct_vs_pre=("knowledge_gap_pct_vs_pre", "mean"),
        knowledge_gap_pct_vs_intact=("knowledge_gap_pct_vs_intact", "mean"),
        knowledge_gain_pct_vs_first_stage=("knowledge_gain_pct_vs_first_stage", "mean"),
        pre_frontier_gain=("pre_frontier_gain", "mean"),
        post_frontier_gain=("frontier_gain", "mean"),
        intact_frontier_gain=("intact_frontier_gain", "mean"),
        frontier_gap_vs_pre=("frontier_gap_vs_pre", "mean"),
        frontier_gap_vs_intact=("frontier_gap_vs_intact", "mean"),
        gini_gap_vs_pre=("gini_gap_vs_pre", "mean"),
        gini_gap_vs_intact=("gini_gap_vs_intact", "mean"),
        effective_domains_gap_vs_pre=("effective_domains_gap_vs_pre", "mean"),
        effective_domains_gap_vs_intact=("effective_domains_gap_vs_intact", "mean"),
        original_ties_retained=("original_ties_retained", "mean"),
        new_ties=("new_ties", "mean"),
        recovery_cost=("recovery_cost", "mean"),
    )
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    for filename,df in (
        ("comparison_runs.csv",runs),
        ("comparison_means.csv",means),
        ("comparison_recovery_trajectories.csv",recovery),
        ("comparison_innovation_trajectories.csv",innovation),
        ("comparison_benchmark_trajectories.csv",benchmarks),
    ):
        df.to_csv(out/filename,index=False)
    (out/"experiment_settings.json").write_text(json.dumps({
        "mode":args.mode,"paired_seeds":args.seeds,
        "scenarios":[x[0] for x in scenarios],
        "params":params,
    },indent=2))
    print(means[["mode","knowledge_gap_pct_vs_pre",
                 "knowledge_gap_pct_vs_intact","new_ties"]].round(3)
          .to_string(index=False))
    print(f"Results saved to {out.resolve()}")


if __name__=="__main__":
    main()
