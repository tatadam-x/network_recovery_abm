"""Regression tests for learning mechanisms and paired counterfactuals."""
import unittest
import numpy as np
import networkx as nx
from innovation_model import (
    knowledge_increment, collaborative_increment, innovation_period
)
from model_core import RecoveryModel, PARAMS


def setup(rule="bounded_surplus", theta=.75, strength=2.):
    return {
        **PARAMS,
        "knowledge_production_rule":rule,
        "decomposability":theta,
        "collab_surplus_strength":strength,
        "pi_collab_success":1.0,
        "pi_self_success":1.0,
    }


class CollaborativeSurplusTests(unittest.TestCase):
    def test_original_bounded_unchanged(self):
        a=np.array([.9,.1])
        b=np.array([.1,.9])
        p=setup("bounded_ces")
        pool=.25*np.minimum(a,b)+.75*np.maximum(a,b)
        score,chi,premium=collaborative_increment(pool,a,b,p,pool)
        self.assertAlmostEqual(score, knowledge_increment(pool,p,pool))
        self.assertEqual(chi,0.)
        self.assertEqual(premium,0.)

    def test_cap_and_identical_composition(self):
        p=setup()
        a=np.array([.8,.2,0.])
        b=2*a
        pool=.25*np.minimum(a,b)+.75*np.maximum(a,b)
        val,chi,bonus=collaborative_increment(pool,a,b,p,pool)
        self.assertAlmostEqual(chi,0.,places=12)
        self.assertAlmostEqual(bonus,0.,places=12)
        self.assertAlmostEqual(val,.24*.25,places=12)
        for multiplier in (1,10,1e5):
            v,_,_=collaborative_increment(
                pool*multiplier,a*multiplier,b*multiplier,p,pool
            )
            self.assertGreaterEqual(v,0)
            self.assertLessEqual(v,.24+1e-12)

    def test_complementary_produces_bonus_only_when_task_allows(self):
        a=np.array([.96,.04])
        b=np.array([.04,.96])
        pool=.25*np.minimum(a,b)+.75*np.maximum(a,b)
        p=setup()
        v,chi,bonus=collaborative_increment(pool,a,b,p,pool)
        self.assertGreater(chi,0)
        self.assertGreater(bonus,0)
        self.assertGreater(v,.06)
        self.assertLess(v,.24)
        p0=setup(theta=0)
        pool0=np.minimum(a,b)
        v0,chi0,bonus0=collaborative_increment(pool0,a,b,p0,pool0)
        self.assertEqual(chi0,0.)
        self.assertEqual(bonus0,0.)
        self.assertAlmostEqual(v0,.06)
        zero=setup(strength=0)
        self.assertAlmostEqual(collaborative_increment(pool,a,b,zero,pool)[0],.06)

    def test_self_search_unchanged(self):
        x=np.array([.2,.3,.5])
        self.assertAlmostEqual(
            knowledge_increment(x,setup("bounded_surplus"),x),
            knowledge_increment(x,setup("bounded_ces"),x)
        )

    def test_random_period_counts_and_cap(self):
        x=np.array([[.99,.01],[.01,.99]])
        g=nx.Graph([(0,1)])
        p=setup()
        p["collaboration_prob"]=1.
        post,counts,_=innovation_period(
            x,g,np.ones(2,dtype=bool),p,np.random.default_rng(3),
            reference_knowledge=x
        )
        self.assertGreater(counts["collab_premium_sum"],0)
        self.assertGreater(counts["collab_complementarity_sum"],0)
        self.assertLessEqual(counts["collab_knowledge_produced"],4*.24+1e-12)
        self.assertAlmostEqual(post.sum()-x.sum(),counts["total_knowledge_produced"])

    def test_invalid_strength_rejected(self):
        with self.assertRaises(ValueError):
            collaborative_increment(np.ones(2),np.ones(2),np.ones(2),
                                     setup(strength=-1.),np.ones(2))


class RecoveryHeuristicTests(unittest.TestCase):
    def model(self, rule, mode="EA",seed=5, severity=.5):
        p={**PARAMS,"N":36,"C":3,"T":4,"innovation_horizon":5,
           "seed":seed,"mode":mode,"shock_type":"random",
           "shock_severity":severity,
           "ea_partner_rule":rule,"knowledge_production_rule":"bounded_surplus"}
        m=RecoveryModel(p)
        pre_knowledge=m.initial_firm_knowledge.copy()
        shocked=tuple(sorted(tuple(sorted(e)) for e in m.G.edges()))
        while m.running: m.step()
        return m,pre_knowledge,shocked

    def test_non_ea_modes_unchanged_by_ea_selection(self):
        for mode in ("DS","DA","ES"):
            a,ka,ga=self.model("uniform",mode=mode)
            b,kb,gb=self.model("community_gap",mode=mode)
            np.testing.assert_array_equal(ka,kb)
            self.assertEqual(ga,gb)
            self.assertEqual(set(a.G.edges()),set(b.G.edges()))
            self.assertAlmostEqual(
                a.innovation_results["post_knowledge_gain"],
                b.innovation_results["post_knowledge_gain"]
            )

    def test_matched_pre_and_paused_reference(self):
        a,ka,ga=self.model("uniform")
        b,kb,gb=self.model("community_gap")
        np.testing.assert_array_equal(ka,kb)
        self.assertEqual(ga,gb)
        for key in ("pre_knowledge_gain","intact_paused_knowledge_gain"):
            self.assertEqual(a.innovation_results[key],b.innovation_results[key])
        self.assertGreaterEqual(a.innovation_results["snapshots"]["post_end"]["mean_effective_domains"],1)

    def test_no_shock_all_modes_equal_to_intact(self):
        for mode in ("DS","DA","ES","EA"):
            m,_,_=self.model("community_gap",mode=mode,severity=0.)
            self.assertAlmostEqual(
                m.innovation_results["R_K_vs_intact_paused"],0,places=10
            )
            self.assertEqual(m.new_ties_count,0)

    def test_learning_params_validate(self):
        bad={**PARAMS,"ea_partner_rule":"invalid"}
        with self.assertRaises(ValueError):
            RecoveryModel(bad)
        bad={**PARAMS,"collab_surplus_strength":-2}
        with self.assertRaises(ValueError):
            RecoveryModel(bad)


if __name__=="__main__":
    unittest.main()
