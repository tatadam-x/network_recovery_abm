"""Tests for community knowledge initialization, bounded increments and matching."""
import unittest
import numpy as np
import networkx as nx
from innovation_model import (
    initialize_community_knowledge, knowledge_increment,
    ces_aggregate, innovation_period, knowledge_metrics,
)

BOUNDED = {
    "knowledge_production_rule":"bounded_ces",
    "bounded_max_increment":0.24,
    "bounded_initial_fraction":0.25,
    "knowledge_scale":0.01,
    "knowledge_rho":0.5,
    "collaboration_prob":0.5,
    "pi_self_success":1.,
    "pi_collab_success":1.,
    "decomposability":0.75,
}

class PageInnovationTests(unittest.TestCase):
    def test_dirichlet_reproducible_and_equal_totals(self):
        kw=dict(n=40,k=6,communities=[i%4 for i in range(40)],
                initial_total=1.,within_sd=.05,between_diff=.70,
                method="dirichlet",alpha_between=40.,alpha_within=40.)
        a,profiles=initialize_community_knowledge(
            **kw,rng=np.random.default_rng(17))
        b,p2=initialize_community_knowledge(
            **kw,rng=np.random.default_rng(17))
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(profiles,p2)
        np.testing.assert_allclose(a.sum(axis=1),np.ones(40),atol=1e-14)
        self.assertTrue(np.all(a>=0))
        self.assertTrue(np.all(profiles>=0))
        self.assertTrue(np.all(a<=1))

    def test_focal_centers_have_distinct_domains(self):
        n=400
        communities=np.repeat(np.arange(4),n//4)
        data,_=initialize_community_knowledge(
            n,6,communities,1,.05,.80,np.random.default_rng(11),
            method="dirichlet",alpha_between=1000,alpha_within=1000)
        centers=np.array([data[communities==i].mean(axis=0)
                          for i in range(4)])
        self.assertEqual(len(set(np.argmax(centers,axis=1))),4)
        self.assertTrue(all(centers[i,i]>0.5 for i in range(4)))

    def test_dirichlet_support_and_focal_dimension_constraint(self):
        with self.assertRaises(ValueError):
            initialize_community_knowledge(
                8,3,list(range(4))*2,1.,.05,.8,np.random.default_rng(1),
                method="dirichlet")
        x,_=initialize_community_knowledge(
            6,6,list(range(6)),1.,.05,1.,np.random.default_rng(12),
            method="dirichlet")
        np.testing.assert_allclose(x,np.eye(6))

    def test_bounded_increment_calibration_and_cap(self):
        first=np.array([0.2,0.3,0.5])
        start=knowledge_increment(first,BOUNDED,first)
        self.assertAlmostEqual(start,0.06)
        self.assertGreater(knowledge_increment(first*20,BOUNDED,first),start)
        self.assertLess(knowledge_increment(first*20,BOUNDED,first),0.24)
        self.assertAlmostEqual(knowledge_increment(first*1e8,BOUNDED,first),0.24,places=7)
        self.assertEqual(knowledge_increment(np.zeros(3),BOUNDED,np.zeros(3)),0)

    def test_stock_and_bounded_modes_differ_as_intended(self):
        first=np.array([.3,.4,.3])
        legacy={**BOUNDED,"knowledge_production_rule":"ces_stock"}
        self.assertAlmostEqual(
            knowledge_increment(first,legacy,first),
            .01*ces_aggregate(first,.5)
        )
        self.assertGreater(
            knowledge_increment(first*100,legacy,first),.24
        )

    def test_bounded_collaboration_uses_initial_partner_pool(self):
        p={**BOUNDED,"collaboration_prob":1.0}
        graph=nx.Graph([(0,1)])
        original=np.array([[.9,.1],[.1,.9]])
        new, counts,_=innovation_period(
            original,graph,np.ones(2,dtype=bool),p,np.random.default_rng(3),
            reference_knowledge=original)
        self.assertEqual(counts["collab_successes"],2)
        self.assertAlmostEqual(counts["collab_knowledge_produced"],4*.06)
        self.assertAlmostEqual(new.sum()-original.sum(),.24)

    def test_diversity_effective_domains(self):
        portfolio=np.array([[1,0,0],[1/3,1/3,1/3]])
        metrics=knowledge_metrics(portfolio)
        self.assertAlmostEqual(metrics["mean_effective_domains"],2.)
        self.assertAlmostEqual(metrics["min_effective_domains"],1.)
        self.assertAlmostEqual(metrics["fraction_near_specialist"],.5)


try:
    from model_core import RecoveryModel, PARAMS
    MESA_AVAILABLE=True
except ImportError:
    MESA_AVAILABLE=False

@unittest.skipUnless(MESA_AVAILABLE,"Mesa unavailable")
class V5IntegrationTests(unittest.TestCase):
    def run_model(self,init,production,mode="EA",seed=21):
        p={**PARAMS,"N":30,"C":3,"T":3,"innovation_horizon":4,
           "shock_type":"random","shock_severity":.5,
           "mode":mode,"seed":seed,
           "knowledge_init_rule":init,
           "knowledge_production_rule":production}
        m=RecoveryModel(p)
        while m.running:m.step()
        return m

    def test_four_specs_and_controls(self):
        for init in ("dirichlet","gaussian"):
            for production in ("bounded_ces","ces_stock"):
                ds=self.run_model(init,production,"DS")
                ea=self.run_model(init,production,"EA")
                np.testing.assert_array_equal(ds.initial_firm_knowledge,
                                              ea.initial_firm_knowledge)
                self.assertAlmostEqual(
                    ds.innovation_results["intact_paused_knowledge_gain"],
                    ea.innovation_results["intact_paused_knowledge_gain"])
                self.assertIn("R_D_vs_intact_paused",ea.innovation_results)
                self.assertGreater(ds.innovation_results["post_knowledge_gain"],0)

    def test_zero_shock_no_difference_all_specs(self):
        for init in ("dirichlet","gaussian"):
            for prod in ("bounded_ces","ces_stock"):
                p={**PARAMS,"N":24,"C":3,"T":3,"innovation_horizon":3,
                   "mode":"EA","shock_severity":0,
                   "knowledge_init_rule":init,
                   "knowledge_production_rule":prod}
                m=RecoveryModel(p)
                while m.running:m.step()
                for key in ("R_K","R_F","R_G","R_D"):
                    self.assertAlmostEqual(
                        m.innovation_results[key+"_vs_intact_paused"],0,places=10)
