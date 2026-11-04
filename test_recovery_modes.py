"""Integration tests for recovery mode rules (requires Mesa and dependencies)."""
import unittest
import numpy as np

from model_core import PARAMS, RecoveryModel

MODES = ("DS", "DA", "ES", "EA", "HY", "CHAIN")


def scenario(mode="DS", seed=4, **overrides):
    p = {**PARAMS, "N": 32, "C": 4, "T": 9,
         "innovation_horizon": 3, "seed": seed,
         "shock_type": "random", "shock_severity": 0.50,
         "mode": mode}
    p.update(overrides)
    return p


def finish(model):
    while model.running:
        model.step()
    return model


class RecoveryModeTests(unittest.TestCase):
    def test_budget_conservation_and_endpoint_capacity(self):
        for reg in MODES:
            with self.subTest(reg=reg):
                m = RecoveryModel(scenario(reg))
                before = sum(a.slack for a in m.agents_by_id.values())
                finish(m)
                after = sum(a.slack for a in m.agents_by_id.values())
                self.assertAlmostEqual(before, after + m.recovery_spend, places=7)
                self.assertTrue(all(a.slack >= -1e-8 for a in m.agents_by_id.values()))
                self.assertTrue(all(a.repair_capacity >= 0 for a in m.agents_by_id.values()))
                self.assertTrue(all(m.G.has_edge(u,v) is False
                                    for u,v in m.G.edges()
                                    if not m.agents_by_id[u].active or
                                       not m.agents_by_id[v].active))

    def test_pre_shock_state_is_paired(self):
        initial = []
        for reg in MODES:
            m = RecoveryModel(scenario(reg))
            initial.append((
                sorted(m.G.edges()),
                tuple(a.active for a in m.agents_by_id.values()),
                m.pre_knowledge_end.copy(),
            ))
        for data in initial[1:]:
            self.assertEqual(data[0], initial[0][0])
            self.assertEqual(data[1], initial[0][1])
            np.testing.assert_array_equal(data[2],initial[0][2])

    def test_zero_shock_creates_no_unmotivated_ties(self):
        for reg in MODES:
            with self.subTest(reg=reg):
                m = finish(RecoveryModel(scenario(reg, shock_severity=0)))
                self.assertEqual(set(m.G.edges()), set(m.G_pre.edges()))
                self.assertEqual(m.recovery_spend, 0)
                self.assertAlmostEqual(m.innovation_results["R_K_vs_intact_paused"],0)

    def test_timed_chain_switches_after_completed_periods(self):
        m = finish(RecoveryModel(scenario(
            "CHAIN", chain_first="DS", chain_second="EA",
            chain_switch_rule="time", chain_switch_time=3)))
        self.assertEqual([v for _,v in m.mode_history[:3]], ["DS"]*3)
        self.assertEqual([v for _,v in m.mode_history[3:]], ["EA"]*6)
        self.assertEqual(m.switch_at,4)
        self.assertEqual(m.innovation_results["final_recovery_mode"], "EA")

    def test_restoration_switch_is_threshold_based(self):
        m = finish(RecoveryModel(scenario(
            "CHAIN", chain_first="DS", chain_second="EA",
            chain_switch_rule="restoration", chain_switch_threshold=0)))
        self.assertEqual(m.mode_history[0][1], "DS")
        self.assertEqual(m.mode_history[1][1], "EA")

    def test_shock_severity_honored_across_types(self):
        # Full inactivation makes shocked firms directly observable.
        for kind in ("targeted_hub","random","clustered"):
            with self.subTest(shock=kind):
                m=RecoveryModel(scenario(shock_type=kind,shock_severity=.50,
                                         p_inactive=1.0))
                n_inactive=sum(not a.active for a in m.agents_by_id.values())
                self.assertEqual(n_inactive,16)

    def test_reactivation_is_paired_across_modes(self):
        histories=[]
        for reg in MODES:
            m=RecoveryModel(scenario(reg))
            state=[]
            while m.running:
                m.step()
                state.append(tuple(a.active for a in m.agents_by_id.values()))
            histories.append(state)
        self.assertTrue(all(history==histories[0] for history in histories))


if __name__ == "__main__":
    unittest.main()
