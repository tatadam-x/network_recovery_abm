"""Regression checks for pre/post benchmark reporting and trajectory alignment."""
import unittest
import numpy as np

from benchmark_reporting import (prepare_benchmark_trajectories,
                                 check_summary_alignment)
from comparison_engine import compare_scenarios
from chain_experiments import priority_chain_scenarios
from model_core import PARAMS


class BenchmarkReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        params = {
            **PARAMS, "N": 24, "C": 4, "K": 6, "T": 4,
            "innovation_horizon": 4, "shock_type": "random",
            "shock_severity": 0.45,
            "chain_switch_rule": "time", "chain_switch_time": 2,
        }
        cls.summary, cls.recovery, cls.innovation = compare_scenarios(
            tuple(sorted(params.items())), priority_chain_scenarios(), 2
        )
        cls.benchmark = prepare_benchmark_trajectories(cls.innovation)

    def test_window_alignment_and_endpoints(self):
        self.assertEqual(len(self.benchmark), len(self.summary) * 5)
        check_summary_alignment(self.summary, self.benchmark)

    def test_reference_trajectories_common_within_seed(self):
        for (_, period), group in self.benchmark.groupby(["seed", "period"]):
            for out in ("knowledge", "frontier", "gini", "diversity"):
                for prefix in ("pre", "intact"):
                    self.assertAlmostEqual(
                        group[f"{prefix}_{out}_change"].max(),
                        group[f"{prefix}_{out}_change"].min()
                    )

    def test_all_trajectory_differences_start_at_zero(self):
        start = self.benchmark[self.benchmark["period"] == 0]
        derived = [x for x in start if x.endswith("_change") or "_gap_vs_" in x]
        self.assertTrue(np.allclose(start[derived].to_numpy(dtype=float), 0))

    def test_summary_validation_fails_when_tampered(self):
        bad = self.summary.copy()
        bad.loc[0, "knowledge_gap_vs_pre"] += 0.1
        with self.assertRaises(AssertionError):
            check_summary_alignment(bad, self.benchmark)

    def test_reference_mismatch_is_detected(self):
        bad = self.innovation.copy()
        candidates = bad.index[(bad.phase == "pre")
                               & (bad["period"] == 1)]
        bad.loc[candidates[0], "total_knowledge"] += 1.0
        with self.assertRaises(ValueError):
            prepare_benchmark_trajectories(bad)

    def test_zero_shock_matches_paused_intact(self):
        p = {
            **PARAMS, "N": 20, "C": 4, "K": 6, "T": 3,
            "innovation_horizon": 3, "shock_type": "random",
            "shock_severity": 0., "chain_switch_rule": "time",
            "chain_switch_time": 1,
        }
        scenarios = [("DS", "DS", "DS", "DS"),
                     ("EA", "EA", "EA", "EA")]
        summary, _, innovation = compare_scenarios(
            tuple(sorted(p.items())), scenarios, 1)
        curves = prepare_benchmark_trajectories(innovation)
        check_summary_alignment(summary, curves)
        self.assertTrue(np.allclose(
            curves["knowledge_gap_vs_intact"].values, 0, atol=1e-10
        ))
        self.assertTrue(np.allclose(
            curves["frontier_gap_vs_intact"].values, 0, atol=1e-10
        ))


if __name__ == "__main__":
    unittest.main()
