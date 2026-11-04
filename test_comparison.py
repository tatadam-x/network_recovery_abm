"""Release-level tests of curated and exhaustive comparison calculations."""
import unittest
import numpy as np
from comparison_engine import compare_scenarios
from chain_experiments import (
    priority_chain_scenarios, all_chain_scenarios, BASE_MODES, PRIORITY_CHAINS
)
from model_core import PARAMS


class ReleaseComparisonTests(unittest.TestCase):
    def _run(self, scenarios):
        p = {
            **PARAMS,
            "N":24, "C":4, "K":6, "T":4, "innovation_horizon":3,
            "shock_type":"random", "shock_severity":0.45,
            "chain_switch_rule":"time", "chain_switch_time":2,
        }
        return compare_scenarios(tuple(sorted(p.items())), scenarios, 1)

    def test_curated_specifications_are_complete(self):
        scenarios = priority_chain_scenarios()
        self.assertEqual(len(scenarios),8)
        self.assertEqual([r[0] for r in scenarios[:4]],list(BASE_MODES))
        self.assertEqual([r[0] for r in scenarios[4:]],
                         [f"{a}->{b}" for a,b in PRIORITY_CHAINS])

    def test_recommended_screen_exports_and_pairing(self):
        summary, recovery, innovation = self._run(priority_chain_scenarios())
        self.assertEqual(len(summary),8)
        self.assertTrue(np.isfinite(summary.knowledge_gain).all())
        self.assertEqual(summary.original_ties_total.nunique(),1)
        self.assertEqual(summary.seed.nunique(),1)
        self.assertTrue({"time","mode","active_pre_shock_ties"}.issubset(recovery))
        self.assertTrue({"phase","period","mode","total_knowledge"}.issubset(innovation))
        self.assertTrue(np.isfinite(
            summary["knowledge_gain_pct_vs_first_stage"].to_numpy()).all())

    def test_all_chains_export_and_same_mode_invariance(self):
        summary,_,_ = self._run(all_chain_scenarios())
        self.assertEqual(len(summary),20)
        for mode in BASE_MODES:
            a=summary.loc[summary["mode"]==mode].iloc[0]
            b=summary.loc[summary["mode"]==f"{mode}->{mode}"].iloc[0]
            for key in ("knowledge_gain","original_ties_retained",
                        "new_ties","recovery_cost"):
                self.assertAlmostEqual(float(a[key]),float(b[key]))


if __name__=="__main__":
    unittest.main()
