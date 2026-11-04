"""Tests for the revised innovation layer. Run: python -m unittest -v test_innovation_model.py"""
import unittest

import networkx as nx
import numpy as np

from innovation_model import (
    initialize_community_knowledge, innovation_period,
    knowledge_metrics, compare_knowledge_windows, ces_production,
)


PARAMS = {
    "collaboration_prob": 0.5,
    "pi_self_success": 1.0,
    "pi_collab_success": 1.0,
    "decomposability": 0.75,
    "knowledge_scale": 0.01,
    "knowledge_rho": 0.5,
}


class InnovationLayerTests(unittest.TestCase):
    def test_equal_initial_totals_and_reproducibility(self):
        kwargs = dict(
            n=20, k=5, communities=[i % 4 for i in range(20)],
            initial_total=1.5, within_sd=0.07, between_diff=0.7,
        )
        a, ma = initialize_community_knowledge(
            **kwargs, rng=np.random.default_rng(77)
        )
        b, mb = initialize_community_knowledge(
            **kwargs, rng=np.random.default_rng(77)
        )
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(ma, mb)
        np.testing.assert_allclose(a.sum(axis=1), 1.5)
        self.assertTrue(np.all(a >= 0))

    def test_community_parameter_extremes(self):
        groups = [0, 0, 1, 1]
        x, references = initialize_community_knowledge(
            4, 3, groups, 1, 0, 0, np.random.default_rng(3),
            method="gaussian"
        )
        np.testing.assert_allclose(x, np.full((4, 3), 1 / 3))
        np.testing.assert_allclose(references[0], references[1])
        differentiated, ref2 = initialize_community_knowledge(
            4, 3, groups, 1, 0, 1, np.random.default_rng(3),
            method="gaussian"
        )
        np.testing.assert_allclose(differentiated[0], differentiated[1])
        self.assertFalse(np.allclose(ref2[0], ref2[1]))

    def test_lapsed_collaboration(self):
        network = nx.empty_graph(3)
        p = {**PARAMS, "collaboration_prob": 1}
        initial = np.full((3, 4), 0.25)
        result, counts, successes = innovation_period(
            initial, network, np.ones(3, dtype=bool),
            p, np.random.default_rng(5)
        )
        np.testing.assert_array_equal(result, initial)
        self.assertEqual(counts["lapsed_collab"], 3)
        self.assertEqual(counts["collab_successes"], 0)
        np.testing.assert_array_equal(successes, np.zeros(3))

    def test_self_search_stochastic_limits(self):
        network = nx.empty_graph(3)
        initial = np.full((3, 4), 0.25)
        never = {**PARAMS, "collaboration_prob": 0, "pi_self_success": 0}
        always = {**never, "pi_self_success": 1}
        zero, counts, _ = innovation_period(
            initial, network, np.ones(3, dtype=bool),
            never, np.random.default_rng(4)
        )
        grown, yes_counts, _ = innovation_period(
            initial, network, np.ones(3, dtype=bool),
            always, np.random.default_rng(4)
        )
        np.testing.assert_array_equal(zero, initial)
        self.assertEqual(counts["self_successes"], 0)
        self.assertEqual(yes_counts["self_successes"], 3)
        self.assertGreater(grown.sum(), initial.sum())

    def test_partner_can_participate_multiple_times(self):
        graph = nx.path_graph(3)
        params = {**PARAMS, "collaboration_prob": 1}
        initial = np.full((3, 3), 1. / 3)
        new, counts, participation = innovation_period(
            initial, graph, np.ones(3, dtype=bool),
            params, np.random.default_rng(8)
        )
        self.assertEqual(counts["collab_successes"], 3)
        self.assertEqual(int(participation.sum()), 6)
        # Node 1 is the only possible partner for nodes 0 and 2 and
        # also initiates its own opportunity: it participates >= 3 times.
        self.assertGreaterEqual(participation[1], 3)
        self.assertGreater(new.sum(), initial.sum())

    def test_beginning_of_period_knowledge_is_frozen(self):
        params = {**PARAMS, "collaboration_prob": 0}
        initial = np.full((4, 2), 0.5)
        new, counts, _ = innovation_period(
            initial, nx.empty_graph(4), np.ones(4, dtype=bool),
            params, np.random.default_rng(9)
        )
        expected_gain = 4 * ces_production(initial[0], 0.01, 0.5)
        self.assertAlmostEqual(new.sum() - initial.sum(), expected_gain, places=10)
        self.assertEqual(counts["self_successes"], 4)

    def test_system_metrics_and_matched_control(self):
        baseline = np.ones((2, 2))
        pre_end = baseline + 0.5
        post_end = pre_end + 1
        intact_start = pre_end.copy()
        intact_end = intact_start + 0.8
        compared = compare_knowledge_windows(
            baseline, pre_end, pre_end, post_end,
            intact_start, intact_end,
        )
        self.assertAlmostEqual(compared["pre_knowledge_gain"], 2)
        self.assertAlmostEqual(compared["post_knowledge_gain"], 4)
        self.assertAlmostEqual(compared["intact_paused_knowledge_gain"], 3.2)
        self.assertAlmostEqual(compared["R_K"], 2)
        self.assertAlmostEqual(compared["R_K_vs_intact_paused"], 0.8)
        self.assertNotIn("uninterrupted_knowledge_gain", compared)
        self.assertAlmostEqual(compared["R_K_relative"], 1)
        self.assertEqual(compared["R_G"], 0)
        self.assertEqual(knowledge_metrics(np.zeros((2, 2)))["gini"], 0)

    def test_uninterrupted_is_optional_and_uses_own_start(self):
        pre = np.ones((2, 2))
        pre_end = pre * 2
        post_end = pre_end * 1.5
        intact_end = pre_end * 1.25
        uninterrupted_start = pre_end * 3
        uninterrupted_end = uninterrupted_start * 1.5
        result = compare_knowledge_windows(
            pre, pre_end, pre_end, post_end, pre_end, intact_end,
            uninterrupted_start, uninterrupted_end,
        )
        self.assertEqual(result["uninterrupted_knowledge_gain"], 12)
        self.assertEqual(result["R_K_vs_uninterrupted"], -8)
        with self.assertRaises(AssertionError):
            compare_knowledge_windows(
                pre, pre_end, pre_end, post_end,
                pre_end + 0.1, intact_end,
            )

    def test_mechanism_diagnostics_on_new_cross_community_tie(self):
        network = nx.Graph()
        network.add_nodes_from(range(2))
        network.add_edge(0, 1)
        original_network = nx.empty_graph(2)
        data = np.array([[0.9, 0.1], [0.1, 0.9]])
        p = {**PARAMS, "collaboration_prob": 1.0}
        next_data, counts, _ = innovation_period(
            data, network, np.ones(2, dtype=bool),
            p, np.random.default_rng(1),
            communities=[0, 1], pre_graph=original_network,
        )
        self.assertEqual(counts["collab_attempts"], 2)
        self.assertEqual(counts["collab_attempts_cross_community"], 2)
        self.assertEqual(counts["collab_attempts_new_ties"], 2)
        self.assertEqual(counts["collab_successes_new_ties"], 2)
        self.assertAlmostEqual(counts["collab_profile_distance_sum"], 1.6)
        self.assertGreater(counts["collab_knowledge_produced_new_ties"], 0)
        self.assertAlmostEqual(
            next_data.sum() - data.sum(), counts["total_knowledge_produced"],
        )
        self.assertAlmostEqual(
            counts["self_knowledge_produced"] +
            counts["collab_knowledge_produced"],
            counts["total_knowledge_produced"],
        )

    def test_inactive_firms_do_not_initiate(self):
        g = nx.path_graph(3)
        initial = np.full((3, 2), 0.5)
        p = {**PARAMS, "collaboration_prob": 0}
        next_x, counts, _ = innovation_period(
            initial, g, np.array([1, 0, 1], dtype=bool),
            p, np.random.default_rng(11)
        )
        np.testing.assert_array_equal(next_x[1], initial[1])
        self.assertEqual(counts["opportunities"], 2)



try:
    from model_core import PARAMS as CORE_PARAMS, RecoveryModel
    MESA_AVAILABLE = True
except ImportError:
    MESA_AVAILABLE = False


@unittest.skipUnless(MESA_AVAILABLE, "Mesa is not installed")
class RecoveryIntegrationTests(unittest.TestCase):
    def make_params(self, regime="DS"):
        params = CORE_PARAMS.copy()
        params.update({
            "N": 24, "C": 3, "T": 3, "innovation_horizon": 4,
            "regime": regime, "seed": 23, "shock_type": "random",
            "shock_severity": .5,
        })
        return params

    def test_benchmark_recovery_and_post_sequence(self):
        model = RecoveryModel(self.make_params())
        np.testing.assert_array_equal(
            model._knowledge_snapshot(), model.pre_knowledge_end
        )
        initial_post_shock_knowledge = model._knowledge_snapshot().copy()
        model.step()
        model.step()
        self.assertIsNone(model.innovation_results)
        np.testing.assert_array_equal(
            model._knowledge_snapshot(), initial_post_shock_knowledge
        )
        model.step()
        self.assertFalse(model.running)
        result = model.innovation_results
        self.assertIsNotNone(result)
        self.assertEqual(result["evaluation_horizon"], 4)
        self.assertAlmostEqual(
            result["snapshots"]["post_start"]["total_knowledge"],
            result["snapshots"]["pre_end"]["total_knowledge"],
        )
        self.assertGreater(result["post_knowledge_gain"], 0)

    def test_paired_counterfactual_same_across_regimes(self):
        ds = RecoveryModel(self.make_params("DS"))
        ea = RecoveryModel(self.make_params("EA"))
        np.testing.assert_array_equal(ds.pre_knowledge_end, ea.pre_knowledge_end)
        self.assertEqual(sorted(ds.G.edges()), sorted(ea.G.edges()))
        while ds.running:
            ds.step()
        while ea.running:
            ea.step()
        self.assertEqual(
            ds.innovation_results["pre_knowledge_gain"],
            ea.innovation_results["pre_knowledge_gain"],
        )
        self.assertEqual(
            ds.innovation_results["intact_paused_knowledge_gain"],
            ea.innovation_results["intact_paused_knowledge_gain"],
        )
        self.assertNotIn("uninterrupted_knowledge_gain", ds.innovation_results)
        np.testing.assert_array_equal(
            ds.innovation_results["snapshots"]["post_start"]["frontier_by_dimension"],
            ds.innovation_results["snapshots"]["intact_paused_start"]["frontier_by_dimension"],
        )

    def test_optional_uninterrupted_reference(self):
        params = self.make_params("DS")
        params["include_uninterrupted_reference"] = True
        model = RecoveryModel(params)
        while model.running:
            model.step()
        result = model.innovation_results
        self.assertIn("uninterrupted_knowledge_gain", result)
        self.assertIn("R_K_vs_uninterrupted", result)
        self.assertGreater(
            result["snapshots"]["uninterrupted_start"]["total_knowledge"],
            result["snapshots"]["post_start"]["total_knowledge"]
        )

    def test_zero_shock_matches_paused_intact_counterfactual(self):
        params = self.make_params("EA")
        params["shock_severity"] = 0.0
        model = RecoveryModel(params)
        while model.running:
            model.step()
        result = model.innovation_results
        self.assertEqual(sorted(model.G.edges()), sorted(model.G_pre.edges()))
        self.assertAlmostEqual(result["R_K_vs_intact_paused"], 0.0)
        self.assertAlmostEqual(result["R_F_vs_intact_paused"], 0.0)
        self.assertAlmostEqual(result["R_G_vs_intact_paused"], 0.0)

    def test_all_regimes_common_intact_reference(self):
        measurements = []
        for regime in ("DS", "DA", "ES", "EA", "HY"):
            model = RecoveryModel(self.make_params(regime))
            while model.running:
                model.step()
            measurements.append(model.innovation_results["intact_paused_knowledge_gain"])
        self.assertTrue(all(x == measurements[0] for x in measurements))


if __name__ == "__main__":
    unittest.main()
