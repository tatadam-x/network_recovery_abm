
# =========================
# Shared ABM core module
# =========================

import copy
import random
import numpy as np
import networkx as nx
from innovation_model import (
    initialize_community_knowledge,
    innovation_period,
    knowledge_metrics,
    compare_knowledge_windows,
)
from mesa import Agent, Model
from mesa.datacollection import DataCollector


# =========================
# Parameter block
# =========================

PARAMS = {
    # -------------------------
    # Global settings
    # -------------------------
    "N": 100,                       # number of firms
    "T": 10,                        # structural recovery horizon, post-shock periods
    "seed": 1,                      # random seed
    "mode": "HY",                 # DS, DA, ES, EA, HY
    "network_type": "modular_small_world",  # modular_small_world, random_erdos_renyi,
                                            # small_world, scale_free, core_periphery
    "baseline_perf": 1.0,

    # -------------------------
    # Network structure
    # -------------------------
    "C": 4,                         # number of communities
    "avg_degree": 4,                # target average degree
    "rewire_prob": 0.05,            # within-community small-world rewiring
    "bridge_prob": 0.01,            # probability of cross-community ties
    "backup_density": 0.20,         # dormant backup ties as share of active ties

    # -------------------------
    # Shock parameters
    # -------------------------
    "shock_type": "targeted_hub",   # targeted_hub, random, clustered
    "hub_target_frac": 0.30,        # top-centrality firms eligible for targeting
    "p_tie_fail": 1.00,             # probability incident ties fail if firm is shocked
    "shock_severity": 0.50,         # share of targeted firms
    "p_inactive": 0.50,             # probability shocked firm becomes inactive

    # -------------------------
    # Recovery parameters
    # -------------------------
    "slack_init": 3.0,              # initial slack per firm
    "repair_capacity": 1,           # max repair actions per firm per period
    "repair_cost": 1.0,             # cost of restoring broken pre-shock tie
    "backup_cost": 0.5,             # cost of activating dormant backup tie
    "new_tie_cost": 1.5,            # cost of forming new tie
    "local_search_radius": 2,       # max graph distance for local substitute search
    "central_pool_frac": 0.40,      # share of slack pooled in centralized modes
    "rewiring_budget": 3,           # max new ties per period in DA
    "bridge_bonus": 0.20,           # extra value for cross-community ties
    "restore_threshold": 0.95,      # collective performance threshold for recovery
    "hybrid_switch_perf": 0.68,     # performance threshold for HY switch

    # -------------------------
    # Reactivation parameters
    # -------------------------
    "reactivation_delay": 1,        # minimum inactive periods before return is possible
    "reactivation_prob": 0.40,      # per-period probability of return after delay

    # --------------------------------------
    # Model-specific calibration parameters
    # --------------------------------------
    # Recovery parameters: common budgets, explicit coordination and search
    "es_repair_prob": 0.80,               # chance an ES firm acts on available repair
    "es_backup_activation_prob": 0.50,    # substitution when own pre-shock partner unavailable
    "da_adaptive_resource_share": 0.45,   # fraction of each period's pooled funds reserved for novelty
    "da_target_novelty_fraction": 0.50,   # upper bound on new ties / disrupted original ties
    "ea_target_novelty_fraction": 0.80,   # upper bound on new ties / disrupted original ties
    "ea_partner_rule": "uniform",      # community_gap or uniform
    "ea_exploration_prob": 0.35,         # global chance discovery vs two-hop local search
    "chain_first": "DS",                 # first recovery mode for CHAIN
    "chain_second": "EA",                # second recovery mode for CHAIN
    "chain_switch_rule": "time",         # time, connectivity, restoration
    "chain_switch_time": 5,             # number of full periods in first mode
    "chain_switch_threshold": 0.80,     # connectivity/restoration threshold
    # Deprecated parameters retained for backwards compatibility with old scripts

    "da_new_tie_cost": 1.00,
    "ea_new_tie_cost": 1.5,
    "ea_repair_prob": 0.35,
    "ea_backup_prob": 0.25,
    "ea_new_tie_prob": 0.80,
    "ea_cross_comm_prob": 0.65,

    # -------------------------
    # Revised knowledge production (formal innovation specification)
    # -------------------------
    "K": 6,                          # knowledge dimensions
    "innovation_horizon": 15,       # equal PRE and POST evaluation durations
    "include_uninterrupted_reference": False, # optional diagnostic only
    "initial_knowledge_total": 1.0, # identical initial total for all firms
    "sigma_within": 0.05,           # Gaussian initialization for sensitivity checks
    "delta_between": 0.70,         # structured Dirichlet focal-center separation
    "knowledge_init_rule": "dirichlet",   # structured Dirichlet or Gaussian
    "alpha_between": 40.0,         # Dirichlet community-profile concentration
    "alpha_within": 40.0,          # Dirichlet within-community concentration
    "collaboration_prob": 0.50,    # p_C, fixed across recovery modes
    "knowledge_production_rule": "bounded_surplus", # bounded_surplus, bounded_ces, or ces_stock
    "knowledge_scale": 0.01,       # A for legacy stock-proportional CES
    "bounded_max_increment": 0.24, # upper bound A for bounded CES (initial = A/4)
    "bounded_initial_fraction": 0.25,
    "collab_surplus_strength": 2.0, # saturating premium for complementary partner portfolios
    "knowledge_rho": 0.50,         # CES exponent (both production rules)
    "decomposability": 0.75,        # theta, weight on dimension-wise maximum
    "pi_self_success": 0.55,        # stochastic self-search success
    "pi_collab_success": 0.60,      # stochastic collaborative success

    # -------------------------
    # Performance weights
    # -------------------------
    "w_reach": 0.50,
    "w_degree": 0.30,
    "w_slack": 0.10,
    "w_innovation": 0.10,
    "perf_max": 1.5,
    "innovation_value": 0.05,

    # -------------------------
    # Simulation control
    # -------------------------
    "collect_agent_data": False,    # keep False for speed in early testing
}


# =========================
# FirmAgent class
# =========================

class FirmAgent(Agent):
    """
    Firm agent in the network recovery ABM.

    Each firm carries:
    - structural state
    - recovery state
    - knowledge state
    - performance state
    """

    def __init__(self, model, unique_id, community, knowledge_vector):
        """
        Initialize a firm agent.

        Parameters
        ----------
        model : RecoveryModel
            The parent Mesa model.
        community : int
            Community membership of the firm.
        knowledge_vector : np.ndarray
            Initial K-dimensional knowledge vector.
        """
        super().__init__(model)

        # -------------------------
        # Identity and structure
        # -------------------------
        self.unique_id = unique_id
        self.community = community
        self.pre_neighbors = set()       # pre-shock partners
        self.backup_neighbors = set()    # dormant backup ties
        self.new_neighbors = set()       # post-shock ties formed after shock

        # -------------------------
        # Recovery state
        # -------------------------
        self.active = True
        self.damage = 0.0
        self.slack = self.model.params["slack_init"]
        self.repair_capacity = self.model.params["repair_capacity"]
        self.used_backup_substitution = False
        self.inactive_time = 0

        # -------------------------
        # Knowledge state
        # -------------------------
        self.knowledge = knowledge_vector.astype(float)
        self.innovation_stock = 0

        # -------------------------
        # Performance state
        # -------------------------
        self.performance = 1.0
        self.recovered = False

    # =========================================================
    # Convenience methods
    # =========================================================

    def current_neighbors(self):
        """
        Return the set of currently active neighbors in the main graph.
        """
        return set(self.model.G.neighbors(self.unique_id))

    def current_degree(self):
        """
        Return current active degree.
        """
        return self.model.G.degree(self.unique_id)

    def pre_degree(self):
        """
        Return pre-shock degree.
        """
        return len(self.pre_neighbors)

    def reachable_pre_neighbors(self):
        """
        Return the set of pre-shock partners that are currently reachable
        through the active graph.
        """
        reachable = set()
        if not self.active:
            return reachable

        for nbr in self.pre_neighbors:
            if nbr in self.model.G.nodes:
                try:
                    if nx.has_path(self.model.G, self.unique_id, nbr):
                        reachable.add(nbr)
                except nx.NetworkXError:
                    pass
        return reachable

    def reach_ratio(self):
        """
        Share of pre-shock neighbors currently reachable.
        """
        if len(self.pre_neighbors) == 0:
            return 1.0
        return len(self.reachable_pre_neighbors()) / len(self.pre_neighbors)

    # =========================================================
    # Recovery and performance methods
    # =========================================================

    def reset_period_budgets(self):
        """
        Reset per-period recovery action capacity.
        """
        self.repair_capacity = self.model.params["repair_capacity"]

    def update_performance(self):
        """
        Update local performance using the default additive rule.
        """
        p = self.model.params

        reach = self.reach_ratio()

        if self.pre_degree() > 0:
            degree_recovery = min(self.current_degree() / self.pre_degree(), 1.0)
        else:
            degree_recovery = 1.0

        slack_ratio = min(self.slack / p["slack_init"], 1.0)

        perf = (
            p["w_reach"] * reach
            + p["w_degree"] * degree_recovery
            + p["w_slack"] * slack_ratio
            + p["w_innovation"] * p["innovation_value"] * self.innovation_stock
        )

        self.performance = min(max(perf, 0.0), p["perf_max"])
        self.recovered = self.performance >= p["restore_threshold"]

    def can_pay_cost(self, cost):
        """
        Check whether the firm has enough slack for an action.
        """
        return self.slack >= cost

    def pay_cost(self, cost):
        """
        Deduct slack for an action.
        """
        self.slack = max(self.slack - cost, 0.0)

    # Knowledge production is implemented in innovation_model.py.
    # Firm knowledge changes only in the pre-shock and post-recovery windows.


# =========================
# Helper functions
# Network initialization, backup ties, shocks, and metrics
# =========================

def build_initial_network(model):
    """
    Build the initial active network based on the selected topology.

    Supported network_type values:
    - modular_small_world
    - random_erdos_renyi
    - small_world
    - scale_free
    - core_periphery
    """
    n = model.params["N"]
    network_type = model.params["network_type"]
    c = model.params["C"]
    avg_degree = model.params["avg_degree"]
    rewire_prob = model.params["rewire_prob"]
    bridge_prob = model.params["bridge_prob"]

    model.G.clear()
    model.G.add_nodes_from(range(n))

    if network_type == "modular_small_world":
        communities = {comm: [] for comm in range(c)}
        for node in range(n):
            comm = node % c
            communities[comm].append(node)

        for comm_nodes in communities.values():
            m = len(comm_nodes)
            if m < 3:
                continue

            k = min(avg_degree, m - 1)
            if k % 2 == 1:
                k -= 1
            if k < 2:
                k = 2 if m >= 3 else 1

            subgraph = nx.watts_strogatz_graph(
                n=m,
                k=k,
                p=rewire_prob,
                seed=model.seed
            )

            relabel_map = {i: comm_nodes[i] for i in range(m)}
            subgraph = nx.relabel_nodes(subgraph, relabel_map)
            model.G.add_edges_from(subgraph.edges())

        nodes = list(model.G.nodes())
        for i in nodes:
            for j in nodes:
                if i < j and (i % c) != (j % c) and not model.G.has_edge(i, j):
                    if model.random.random() < bridge_prob:
                        model.G.add_edge(i, j)

    elif network_type == "random_erdos_renyi":
        p = avg_degree / max(n - 1, 1)
        g = nx.erdos_renyi_graph(n=n, p=p, seed=model.seed)
        model.G.add_edges_from(g.edges())

    elif network_type == "small_world":
        k = min(avg_degree, n - 1)
        if k % 2 == 1:
            k -= 1
        if k < 2:
            k = 2 if n >= 3 else 1

        g = nx.watts_strogatz_graph(
            n=n,
            k=k,
            p=rewire_prob,
            seed=model.seed
        )
        model.G.add_edges_from(g.edges())

    elif network_type == "scale_free":
        m = max(1, avg_degree // 2)
        m = min(m, n - 1)
        g = nx.barabasi_albert_graph(n=n, m=m, seed=model.seed)
        model.G.add_edges_from(g.edges())

    elif network_type == "core_periphery":
        core_frac = 0.2
        core_size = max(2, int(n * core_frac))
        core_nodes = list(range(core_size))
        periphery_nodes = list(range(core_size, n))

        for i in core_nodes:
            for j in core_nodes:
                if i < j:
                    model.G.add_edge(i, j)

        p_core_link = min(1.0, avg_degree / max(core_size, 1))
        p_periphery_link = bridge_prob

        for i in periphery_nodes:
            for j in core_nodes:
                if model.random.random() < p_core_link:
                    model.G.add_edge(i, j)

            for j in periphery_nodes:
                if i < j and model.random.random() < p_periphery_link:
                    model.G.add_edge(i, j)

    else:
        raise ValueError(f"Unsupported network_type: {network_type}")


def create_agents(model):
    """
    Initialize community-centered knowledge profiles, then create agents.
    Every firm starts with the same total initial knowledge.
    """
    n = model.params["N"]
    c = model.params["C"]
    k = model.params["K"]
    communities = [node_id % c for node_id in range(n)]
    profiles, reference_profiles = initialize_community_knowledge(
        n=n,
        k=k,
        communities=communities,
        initial_total=model.params["initial_knowledge_total"],
        within_sd=model.params["sigma_within"],
        between_diff=model.params["delta_between"],
        rng=model.knowledge_rng,
        method=model.params["knowledge_init_rule"],
        alpha_between=model.params["alpha_between"],
        alpha_within=model.params["alpha_within"],
    )
    model.community_knowledge_profiles = reference_profiles
    model.initial_firm_knowledge = profiles.copy()
    model.agents_by_id = {}

    for node_id in range(n):
        agent = FirmAgent(
            model=model,
            unique_id=node_id,
            community=communities[node_id],
            knowledge_vector=profiles[node_id],
        )
        model.agents_by_id[node_id] = agent


def assign_pre_neighbors(model):
    """
    Store pre-shock active partners in each agent.
    """
    for agent_id, agent in model.agents_by_id.items():
        agent.pre_neighbors = set(model.G.neighbors(agent_id))


def generate_backup_ties(model):
    """
    Generate dormant backup ties that are not active in the main graph.
    """
    model.backup_edges = set()

    n_backup_target = int(model.G.number_of_edges() * model.params["backup_density"])
    all_nodes = list(model.G.nodes())

    attempts = 0
    max_attempts = 20000

    while len(model.backup_edges) < n_backup_target and attempts < max_attempts:
        i, j = model.random.sample(all_nodes, 2)
        edge = tuple(sorted((i, j)))

        if (
            i != j
            and not model.G.has_edge(i, j)
            and edge not in model.backup_edges
            and model.agents_by_id[i].community == model.agents_by_id[j].community
        ):
            model.backup_edges.add(edge)
            model.agents_by_id[i].backup_neighbors.add(j)
            model.agents_by_id[j].backup_neighbors.add(i)

        attempts += 1


def store_pre_shock_graph(model):
    """
    Freeze a copy of the pre-shock active network.
    """
    model.G_pre = model.G.copy()


def compute_pre_shock_metrics(model):
    """
    Compute pre-shock node metrics and store them on agents.
    """
    centrality = nx.betweenness_centrality(model.G_pre)

    for agent_id, agent in model.agents_by_id.items():
        agent.pre_centrality = centrality.get(agent_id, 0.0)
        agent.pre_degree_value = len(agent.pre_neighbors)


def apply_initial_shock(model):
    """
    Dispatch to the requested shock type.
    """
    shock_type = model.params["shock_type"]

    if shock_type == "targeted_hub":
        apply_targeted_hub_shock(model)
    elif shock_type == "random":
        apply_random_shock(model)
    elif shock_type == "clustered":
        apply_clustered_shock(model)
    else:
        raise NotImplementedError(f"Shock type not implemented: {shock_type}")


def apply_targeted_hub_shock(model):
    """Target high-centrality firms while honoring the requested count."""
    n = model.params["N"]
    n_shocked = 0 if model.params["shock_severity"] == 0 else max(
        1, round(n * model.params["shock_severity"])
    )
    ranked = sorted(
        model.agents_by_id.values(),
        key=lambda a: (-a.pre_centrality, a.unique_id),
    )
    pool_size = max(1, round(n * model.params["hub_target_frac"]))
    selected = model.random.sample(ranked[:pool_size], min(n_shocked, pool_size))
    remaining = n_shocked - len(selected)
    if remaining:
        selected.extend(model.random.sample(ranked[pool_size:], remaining))
    for a in selected:
        shock_agent(model, a, model.params["p_inactive"], model.params["p_tie_fail"])


def apply_random_shock(model):
    """
    Shock a random subset of firms.
    """
    n = model.params["N"]
    severity = model.params["shock_severity"]
    p_inactive = model.params["p_inactive"]
    p_tie_fail = model.params["p_tie_fail"]

    all_agents = list(model.agents_by_id.values())
    n_shocked = 0 if severity == 0 else max(1, int(n * severity))
    shocked_agents = model.random.sample(all_agents, n_shocked)

    for agent in shocked_agents:
        shock_agent(model, agent, p_inactive, p_tie_fail)


def apply_clustered_shock(model):
    """Concentrate shocks in one community, expanding to others as needed."""
    n = model.params["N"]
    n_shocked = 0 if model.params["shock_severity"] == 0 else max(
        1, round(n * model.params["shock_severity"])
    )
    first = model.random.randrange(model.params["C"])
    other_communities = [c for c in range(model.params["C"]) if c != first]
    model.random.shuffle(other_communities)
    selected = []
    for comm in [first] + other_communities:
        nodes = [a for a in model.agents_by_id.values() if a.community == comm]
        model.random.shuffle(nodes)
        selected.extend(nodes[:max(0, n_shocked-len(selected))])
        if len(selected) >= n_shocked:
            break
    for a in selected:
        shock_agent(model, a, model.params["p_inactive"], model.params["p_tie_fail"])


def shock_agent(model, agent, p_inactive, p_tie_fail):
    """
    Apply disruption to one firm and its incident ties.
    If a firm becomes inactive, all incident active ties fail.
    Otherwise, incident ties fail probabilistically.
    """
    agent.damage = 1.0

    became_inactive = (model.random.random() < p_inactive)
    if became_inactive:
        agent.active = False
        agent.inactive_time = 0

    current_neighbors = list(model.G.neighbors(agent.unique_id))

    if became_inactive:
        for nbr in current_neighbors:
            if model.G.has_edge(agent.unique_id, nbr):
                model.G.remove_edge(agent.unique_id, nbr)
    else:
        for nbr in current_neighbors:
            if model.random.random() < p_tie_fail:
                if model.G.has_edge(agent.unique_id, nbr):
                    model.G.remove_edge(agent.unique_id, nbr)


def count_active_pre_shock_ties(model):
    """
    Count active ties that also existed in the pre-shock graph.
    """
    count = 0
    for i, j in model.G.edges():
        if model.G_pre.has_edge(i, j):
            count += 1
    return count


def count_new_ties(model):
    """
    Count active ties that did not exist in the pre-shock graph.
    """
    count = 0
    for i, j in model.G.edges():
        if not model.G_pre.has_edge(i, j):
            count += 1
    return count


def update_system_metrics(model):
    """
    Recompute model-level metrics from the current graph and agents.
    """
    for agent in model.agents_by_id.values():
        agent.update_performance()

    model.perf_net = float(np.mean([a.performance for a in model.agents_by_id.values()]))

    if model.G.number_of_nodes() == 0:
        model.lcc_share = 0.0
        model.frag = model.params["N"]
    else:
        active_nodes = list(model.G.nodes())
        if model.G.number_of_edges() == 0:
            model.lcc_share = 1.0 / model.params["N"]
            model.frag = len(active_nodes)
        else:
            components = list(nx.connected_components(model.G))
            model.frag = len(components)
            model.lcc_share = max(len(c) for c in components) / model.params["N"]

    model.active_pre_shock_ties = count_active_pre_shock_ties(model)
    model.new_ties_count = count_new_ties(model)


# =========================
# RecoveryModel class
# =========================

class RecoveryModel(Model):
    """
    Mesa model for network recovery modes and resilience.
    """

    def __init__(self, params):
        super().__init__()

        self.params = dict(params)
        self.seed = self.params["seed"]
        self.random.seed(self.seed)
        # Exogenous reactivation must remain paired across mode runs,
        # irrespective of how many random decisions the recovery rule consumes.
        self.exogenous_rng = random.Random(self.seed + 65537)
        # Recovery draws and knowledge draws use separate generators.
        # This ensures identical PRE benchmarks and no-shock references
        # across modes run with the same seed.
        self.knowledge_rng = np.random.default_rng(self.seed + 104729)

        self._validate_innovation_parameters()

        self.t = 0                   # recovery time only: 0...T
        self.mode = self.params["mode"]
        self.baseline_perf = self.params["baseline_perf"]
        self.running = True
        self.hybrid_switched = False
        self.chain_switched = False
        self.switch_at = None
        self.active_mode = None
        self.recovery_spend = 0.0
        self.recovery_actions = 0
        self.mode_history = []
        self.new_edges_created = 0
        self.initial_broken_edges = 0
        self._validate_recovery_parameters()

        self.G = nx.Graph()
        self.G_pre = nx.Graph()
        self.backup_edges = set()
        self.agents_by_id = {}
        self.community_ids = [i % self.params["C"] for i in range(self.params["N"])]

        self.perf_net = self.baseline_perf
        self.lcc_share = 1.0
        self.frag = 1
        self.active_pre_shock_ties = 0
        self.new_ties_count = 0

        self.innovation_trajectories = []
        self.innovation_results = None
        self._innovation_finalized = False

        build_initial_network(self)
        create_agents(self)
        assign_pre_neighbors(self)
        generate_backup_ties(self)
        store_pre_shock_graph(self)
        compute_pre_shock_metrics(self)

        # Pre-shock benchmark: all firms are active on the intact network.
        self.pre_knowledge_start = self._knowledge_snapshot()
        all_active = np.ones(self.params["N"], dtype=bool)
        pre_end, self.pre_innovation_counts, _ = self._run_knowledge_window(
            knowledge=self.pre_knowledge_start,
            graph=self.G_pre,
            active=all_active,
            horizon=self.params["innovation_horizon"],
            rng=self.knowledge_rng,
            phase="pre",
        )
        self.pre_knowledge_end = pre_end.copy()
        self._write_knowledge(pre_end)
        self._knowledge_rng_after_benchmark = copy.deepcopy(
            self.knowledge_rng.bit_generator.state
        )

        # The shock changes firms' activity and their ties, but never
        # the knowledge accumulated through the pre-shock benchmark.
        apply_initial_shock(self)
        self.initial_broken_edges = self.G_pre.number_of_edges() - count_active_pre_shock_ties(self)
        self.affected_firms = set()
        for i, j in self.G_pre.edges():
            if not self.G.has_edge(i, j):
                self.affected_firms.update((i, j))
        for i, agent in self.agents_by_id.items():
            if not agent.active:
                self.affected_firms.add(i)
        update_system_metrics(self)

        # Structural measures are recorded for recovery trajectories.
        # The legacy composite performance index is excluded from reports.
        self.datacollector = DataCollector(
            model_reporters={
                "time": lambda m: m.t,
                "mode": lambda m: m.mode,
                "perf_net": lambda m: m.perf_net,
                "lcc_share": lambda m: m.lcc_share,
                "fragmentation": lambda m: m.frag,
                "active_pre_shock_ties": lambda m: m.active_pre_shock_ties,
                "new_ties_count": lambda m: m.new_ties_count,
                "active_mode": lambda m: m.active_mode or m._initial_mode(),
                "active_firms": lambda m: sum(a.active for a in m.agents_by_id.values()),
                "restoration_share": lambda m: m.active_pre_shock_ties / max(1, m.G_pre.number_of_edges()),
                "cross_community_ties": lambda m: sum(m.agents_by_id[i].community != m.agents_by_id[j].community for i,j in m.G.edges()),
                "recovery_actions": lambda m: m.recovery_actions,
                "recovery_spend": lambda m: m.recovery_spend,
            }
        )
        self.datacollector.collect(self)

    def _validate_innovation_parameters(self):
        p = self.params
        for key in ("N", "K", "C", "T", "innovation_horizon"):
            if int(p[key]) != p[key] or p[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in (
            "collaboration_prob", "decomposability", "delta_between",
            "pi_self_success", "pi_collab_success",
        ):
            if not np.isfinite(p[key]) or not 0 <= p[key] <= 1:
                raise ValueError(f"{key} must be finite and in [0, 1]")
        if p["knowledge_init_rule"] not in ("dirichlet", "gaussian"):
            raise ValueError("knowledge_init_rule must be dirichlet or gaussian")
        if p["knowledge_production_rule"] not in ("bounded_ces", "bounded_surplus", "ces_stock"):
            raise ValueError("knowledge_production_rule must be bounded_ces, bounded_surplus or ces_stock")
        if not np.isfinite(p["collab_surplus_strength"]) or p["collab_surplus_strength"] < 0:
            raise ValueError("collab_surplus_strength must be nonnegative")
        if (p["knowledge_init_rule"] == "dirichlet"
                and p["delta_between"] > 0 and p["C"] > p["K"]):
            raise ValueError("Structured Dirichlet requires C <= K when delta_between > 0")
        for key in ("alpha_between", "alpha_within", "bounded_max_increment"):
            if not np.isfinite(p[key]) or p[key] <= 0:
                raise ValueError(f"{key} must be positive")
        if not (np.isfinite(p["bounded_initial_fraction"])
                and 0 < p["bounded_initial_fraction"] <= 1):
            raise ValueError("bounded_initial_fraction must lie in (0,1]")
        if not np.isfinite(p["sigma_within"]) or p["sigma_within"] < 0:
            raise ValueError("sigma_within must be finite and nonnegative")
        for key in ("initial_knowledge_total", "knowledge_scale", "knowledge_rho"):
            if not np.isfinite(p[key]) or p[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not isinstance(p.get("include_uninterrupted_reference", False), bool):
            raise ValueError("include_uninterrupted_reference must be a boolean")
        if not np.isfinite(p["shock_severity"]) or not 0 <= p["shock_severity"] <= 1:
            raise ValueError("shock_severity must be finite and in [0, 1]")

    def _knowledge_snapshot(self):
        return np.stack([
            self.agents_by_id[i].knowledge.copy()
            for i in range(self.params["N"])
        ])

    def _write_knowledge(self, knowledge):
        for i in range(self.params["N"]):
            self.agents_by_id[i].knowledge = knowledge[i].copy()

    def _run_knowledge_window(
        self, knowledge, graph, active, horizon, rng, phase
    ):
        """Evolve a knowledge population with the network held fixed."""
        state = knowledge.copy()
        event_totals = {}
        participation = np.zeros(self.params["N"], dtype=int)

        self.innovation_trajectories.append({
            "phase": phase, "period": 0, **knowledge_metrics(state),
        })
        for period in range(1, horizon + 1):
            state, period_counts, successes = innovation_period(
                state, graph, active, self.params, rng,
                communities=self.community_ids,
                pre_graph=self.G_pre,
                reference_knowledge=self.initial_firm_knowledge,
            )
            participation += successes
            for key, value in period_counts.items():
                event_totals[key] = event_totals.get(key, 0) + value
            self.innovation_trajectories.append({
                "phase": phase, "period": period,
                **knowledge_metrics(state), **period_counts,
            })

        return state, event_totals, participation

    def _finalize_innovation_experiment(self):
        """Evaluate innovation on recovered and intact networks.

        Knowledge remains fixed during the T-period structural recovery.
        The primary intact-network control shares this same pause, the
        PRE-end knowledge profiles, and the post-benchmark RNG state.
        Optionally, an uninterrupted intact-network continuation provides
        an additional measure of the opportunity cost of the pause.
        """
        if self._innovation_finalized:
            return

        n = self.params["N"]
        horizon = self.params["innovation_horizon"]
        post_start = self._knowledge_snapshot()
        recovered_graph = self.G.copy()
        recovered_active = np.array(
            [self.agents_by_id[i].active for i in range(n)], dtype=bool
        )

        post_end, post_counts, post_participation = self._run_knowledge_window(
            post_start, recovered_graph, recovered_active,
            horizon, self.knowledge_rng, "post"
        )
        self._write_knowledge(post_end)
        for i, successes in enumerate(post_participation):
            # Preserve this agent-level diagnostic for existing consumers.
            self.agents_by_id[i].innovation_stock = int(successes)

        all_active = np.ones(n, dtype=bool)
        # Common post-benchmark state makes this an exact matched-start
        # reference for every recovery mode using the same seed.
        paused_rng = np.random.default_rng()
        paused_rng.bit_generator.state = copy.deepcopy(
            self._knowledge_rng_after_benchmark
        )
        intact_start = self.pre_knowledge_end.copy()
        intact_end, intact_counts, _ = self._run_knowledge_window(
            intact_start, self.G_pre, all_active, horizon,
            paused_rng, "intact_paused"
        )

        uninterrupted_start = uninterrupted_end = None
        uninterrupted_interim_counts = uninterrupted_post_counts = None
        if self.params.get("include_uninterrupted_reference", False):
            uninterrupted_rng = np.random.default_rng()
            uninterrupted_rng.bit_generator.state = copy.deepcopy(
                self._knowledge_rng_after_benchmark
            )
            uninterrupted_start, uninterrupted_interim_counts, _ = (
                self._run_knowledge_window(
                    self.pre_knowledge_end, self.G_pre, all_active,
                    self.params["T"], uninterrupted_rng,
                    "uninterrupted_interim",
                )
            )
            uninterrupted_end, uninterrupted_post_counts, _ = (
                self._run_knowledge_window(
                    uninterrupted_start, self.G_pre, all_active,
                    horizon, uninterrupted_rng, "uninterrupted_post",
                )
            )

        self.innovation_results = compare_knowledge_windows(
            self.pre_knowledge_start, self.pre_knowledge_end,
            post_start, post_end, intact_start, intact_end,
            uninterrupted_start, uninterrupted_end,
        )
        self.innovation_results.update({
            "seed": self.seed,
            "mode": self.mode,
            "recovery_horizon": self.params["T"],
            "switch_at": self.switch_at,
            "final_recovery_mode": self.active_mode,
            "recovery_actions": self.recovery_actions,
            "recovery_spend": self.recovery_spend,
            "initial_broken_ties": self.initial_broken_edges,
            "final_restoration_share": self.active_pre_shock_ties / max(1, self.G_pre.number_of_edges()),
            "evaluation_horizon": horizon,
            "knowledge_init_rule": self.params["knowledge_init_rule"],
            "knowledge_production_rule": self.params["knowledge_production_rule"],
            "collab_surplus_strength": self.params["collab_surplus_strength"],
            "ea_partner_rule": self.params["ea_partner_rule"],
            "alpha_between": self.params["alpha_between"],
            "alpha_within": self.params["alpha_within"],
            "bounded_max_increment": self.params["bounded_max_increment"],
            "bounded_initial_fraction": self.params["bounded_initial_fraction"],
            "active_firms_at_evaluation": int(recovered_active.sum()),
            "recovered_network_edges": recovered_graph.number_of_edges(),
            "recovered_cross_community_edges": sum(
                self.community_ids[i] != self.community_ids[j]
                for i, j in recovered_graph.edges()
            ),
            "recovered_new_edges": sum(
                not self.G_pre.has_edge(i, j)
                for i, j in recovered_graph.edges()
            ),
            "pre_event_counts": self.pre_innovation_counts,
            "post_event_counts": post_counts,
            "intact_paused_event_counts": intact_counts,
        })
        if uninterrupted_interim_counts is not None:
            self.innovation_results.update({
                "uninterrupted_interim_event_counts":
                    uninterrupted_interim_counts,
                "uninterrupted_post_event_counts": uninterrupted_post_counts,
            })
        self._innovation_finalized = True

    def _validate_recovery_parameters(self):
        """Validate controllable recovery rules before the scenario is initialized."""
        p = self.params
        valid = ("DS", "DA", "ES", "EA", "HY", "CHAIN")
        if p["mode"] not in valid:
            raise ValueError(f"Unknown mode: {p['mode']}")
        if p["ea_partner_rule"] not in ("uniform", "community_gap"):
            raise ValueError("ea_partner_rule must be uniform or community_gap")
        if p["chain_first"] not in valid[:4] or p["chain_second"] not in valid[:4]:
            raise ValueError("Chain stages must be DS, DA, ES or EA.")
        if p["chain_switch_rule"] not in ("time", "connectivity", "restoration"):
            raise ValueError("Unknown chain switch rule.")
        if int(p["chain_switch_time"]) != p["chain_switch_time"] or p["chain_switch_time"] < 1:
            raise ValueError("chain_switch_time must be a positive integer.")
        if not 0 <= p["chain_switch_threshold"] <= 1:
            raise ValueError("chain_switch_threshold must be between 0 and 1.")
        for key in (
            "central_pool_frac", "da_adaptive_resource_share",
            "da_target_novelty_fraction", "ea_target_novelty_fraction",
            "ea_repair_prob", "ea_backup_prob", "ea_new_tie_prob",
            "ea_cross_comm_prob", "ea_exploration_prob",
            "es_repair_prob", "es_backup_activation_prob",
        ):
            if not 0 <= p[key] <= 1:
                raise ValueError(f"{key} must be between 0 and 1.")
        for key in ("slack_init", "repair_cost", "backup_cost", "new_tie_cost"):
            if p[key] <= 0:
                raise ValueError(f"{key} must be positive.")
        for key in ("repair_capacity", "rewiring_budget", "local_search_radius"):
            if int(p[key]) != p[key] or p[key] < 1:
                raise ValueError(f"{key} must be a positive integer.")

    def _initial_mode(self):
        return "DS" if self.mode == "HY" else (
            self.params["chain_first"] if self.mode == "CHAIN" else self.mode
        )

    def _mode_for_period(self):
        """Switches are evaluated between recovery periods; no retroactive actions."""
        if self.mode not in ("HY", "CHAIN"):
            return self.mode
        if self.mode == "HY":
            first, second, rule, when, threshold = (
                "DS", "EA", "connectivity", None, self.params["hybrid_switch_perf"]
            )
        else:
            p = self.params
            first, second = p["chain_first"], p["chain_second"]
            rule, when, threshold = (
                p["chain_switch_rule"], p["chain_switch_time"],
                p["chain_switch_threshold"],
            )
        already = self.chain_switched
        if not already:
            if rule == "time":
                trigger = self.t > when
            elif rule == "connectivity":
                trigger = self.t > 1 and self.lcc_share >= threshold
            else:
                share = self.active_pre_shock_ties / max(1, self.G_pre.number_of_edges())
                trigger = self.t > 1 and share >= threshold
            if trigger:
                self.chain_switched = True
                self.hybrid_switched = self.mode == "HY"
                self.switch_at = self.t
        return second if self.chain_switched else first

    def _central_contributions(self):
        """Pool a share of active firms' *existing* resources, no external funds."""
        frac = self.params["central_pool_frac"]
        contributors = [a for a in self.agents_by_id.values() if a.active]
        pool = 0.0
        for agent in contributors:
            contribution = frac * agent.slack
            agent.slack -= contribution
            pool += contribution
        return pool, contributors

    @staticmethod
    def _refund_pool(pool, contributors):
        if contributors and pool > 0:
            refund = pool / len(contributors)
            for a in contributors:
                a.slack += refund

    def _edge_eligible(self, u, v):
        return (
            u != v and not self.G.has_edge(u,v)
            and self.agents_by_id[u].active and self.agents_by_id[v].active
            and self.agents_by_id[u].repair_capacity > 0
            and self.agents_by_id[v].repair_capacity > 0
        )

    def _activate_edge(self, u, v, kind, cost, payer=None):
        """One undirected tie uses one action at both endpoints and costs one resource unit."""
        if not self._edge_eligible(u, v):
            return False
        if payer is not None and not payer.can_pay_cost(cost):
            return False
        self.G.add_edge(u, v)
        for i in (u, v):
            self.agents_by_id[i].repair_capacity -= 1
        if payer is not None:
            payer.pay_cost(cost)
        if kind != "repair":
            self.agents_by_id[u].new_neighbors.add(v)
            self.agents_by_id[v].new_neighbors.add(u)
            self.new_edges_created += 1
        self.recovery_spend += cost
        self.recovery_actions += 1
        return True

    def _repair_candidates(self, adaptive=False):
        """Central actor ranks globally visible disrupted relationships."""
        comps = {}
        for component_idx, nodes in enumerate(nx.connected_components(self.G)):
            comps.update({node: component_idx for node in nodes})
        candidates = []
        for u,v in self.G_pre.edges():
            if not self._edge_eligible(u,v):
                continue
            a, b = self.agents_by_id[u], self.agents_by_id[v]
            score = a.pre_centrality + b.pre_centrality
            score += self.params["bridge_bonus"] * (a.community != b.community)
            # Adaptation emphasizes restoring access between currently separated components.
            if adaptive and comps[u] != comps[v]:
                score += 2.0
            candidates.append((score, u, v))
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        return candidates

    def _step_directed_stabilization(self):
        pool, contributors = self._central_contributions()
        cost = self.params["repair_cost"]
        for _, u, v in self._repair_candidates(adaptive=False):
            if pool + 1e-9 < cost:
                break
            if self._activate_edge(u, v, "repair", cost):
                pool -= cost
        self._refund_pool(pool, contributors)

    def _directed_novelty_candidates(self):
        """System-wide search for new cross-community ties with structural reach."""
        comps = {}
        for k, nodes in enumerate(nx.connected_components(self.G)):
            comps.update({node: k for node in nodes})
        n = self.params["N"]
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if self.G_pre.has_edge(u, v) or not self._edge_eligible(u, v):
                    continue
                a, b = self.agents_by_id[u], self.agents_by_id[v]
                if a.community == b.community:
                    continue
                # Scores reflect global structural information, not private knowledge.
                score = (
                    4.0 * (comps[u] != comps[v])
                    + 1.0
                    + 0.3 * (a.pre_centrality + b.pre_centrality)
                    + 0.04 * (a.pre_degree() + b.pre_degree() -
                              a.current_degree() - b.current_degree())
                )
                # Penalize concentration of bridges in a handful of firms.
                score -= 0.08 * (a.current_degree() + b.current_degree())
                candidates.append((score, u, v))
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        return candidates

    def _step_directed_adaptation(self):
        pool, contributors = self._central_contributions()
        p = self.params
        repair_cost, new_cost = p["repair_cost"], p["new_tie_cost"]
        reserved = pool * p["da_adaptive_resource_share"]
        repairs_pool = pool - reserved
        # Resource allocation makes renewal compete directly with restoration.
        for _, u, v in self._repair_candidates(adaptive=True):
            if repairs_pool + 1e-9 < repair_cost:
                break
            if self._activate_edge(u, v, "repair", repair_cost):
                repairs_pool -= repair_cost
        novelty_pool = reserved + repairs_pool
        target = int(np.ceil(p["da_target_novelty_fraction"] * self.initial_broken_edges))
        remaining = max(0, target - self.new_edges_created)
        new_budget = min(p["rewiring_budget"], remaining)
        n_added = 0
        for _, u, v in self._directed_novelty_candidates():
            if n_added >= new_budget or novelty_pool + 1e-9 < new_cost:
                break
            if self._activate_edge(u, v, "new", new_cost):
                novelty_pool -= new_cost
                n_added += 1
        self._refund_pool(novelty_pool, contributors)

    def _step_emergent_stabilization(self):
        """Local repair first; substitute dormant ties if original partners unavailable."""
        p = self.params
        ids = list(self.agents_by_id)
        self.random.shuffle(ids)
        for u in ids:
            a = self.agents_by_id[u]
            if not a.active or a.repair_capacity <= 0:
                continue
            can_repair = [
                v for v in sorted(a.pre_neighbors)
                if self._edge_eligible(u, v)
            ]
            if (can_repair and a.can_pay_cost(p["repair_cost"])
                    and self.random.random() < p["es_repair_prob"]):
                v = self.random.choice(can_repair)
                if self._activate_edge(u, v, "repair", p["repair_cost"], payer=a):
                    continue
            missing_unavailable = any(
                not self.agents_by_id[v].active and not self.G.has_edge(u,v)
                for v in a.pre_neighbors
            )
            backups = [
                v for v in sorted(a.backup_neighbors)
                if self._edge_eligible(u,v)
            ]
            if (missing_unavailable and backups
                    and a.current_degree() < a.pre_degree()
                    and a.can_pay_cost(p["backup_cost"])
                    and self.random.random() < p["es_backup_activation_prob"]):
                v = self.random.choice(backups)
                self._activate_edge(u, v, "backup", p["backup_cost"], payer=a)

    def _ea_candidates(self, u):
        """Current local neighborhoods plus occasional distant exploration."""
        p = self.params
        a = self.agents_by_id[u]
        if self.random.random() < p["ea_exploration_prob"]:
            candidates = [
                v for v in sorted(self.agents_by_id)
                if v != u and not self.G_pre.has_edge(u,v)
                and self._edge_eligible(u,v)
            ]
        else:
            distances = nx.single_source_shortest_path_length(
                self.G, u, cutoff=p["local_search_radius"]
            )
            candidates = [
                v for v, dist in sorted(distances.items())
                if 1 < dist <= p["local_search_radius"]
                and not self.G_pre.has_edge(u,v)
                and self._edge_eligible(u,v)
            ]
        if not candidates:
            return []
        # A local preference for accessing other communities, without knowledge visibility.
        if self.random.random() < p["ea_cross_comm_prob"]:
            cross = [v for v in candidates
                     if a.community != self.agents_by_id[v].community]
            if cross:
                candidates = cross
        return candidates

    def _select_ea_partner(self, u, candidates):
        """Select a local candidate using only observable community memberships.

        The community-gap heuristic seeks new partners in communities
        underrepresented among firm u's existing direct neighbors. The legacy
        rule chooses uniformly. Neither method observes private knowledge.
        """
        if self.params["ea_partner_rule"] == "uniform":
            return self.random.choice(candidates)
        neighbors_by_comm = {}
        for v in self.G.neighbors(u):
            comm = self.agents_by_id[v].community
            neighbors_by_comm[comm] = neighbors_by_comm.get(comm, 0) + 1
        min_coverage = min(
            neighbors_by_comm.get(self.agents_by_id[v].community, 0)
            for v in candidates
        )
        shortlist = [
            v for v in candidates
            if neighbors_by_comm.get(self.agents_by_id[v].community, 0)
               == min_coverage
        ]
        return self.random.choice(shortlist)

    def _step_emergent_adaptation(self):
        """Decentralized exploration using local resources and evolving connections."""
        p = self.params
        target = int(np.ceil(p["ea_target_novelty_fraction"] * self.initial_broken_edges))
        ids = list(self.agents_by_id)
        self.random.shuffle(ids)
        for u in ids:
            a = self.agents_by_id[u]
            if not a.active or a.repair_capacity <= 0 or u not in self.affected_firms:
                continue
            repairable = [v for v in sorted(a.pre_neighbors)
                          if self._edge_eligible(u,v)]
            if (repairable and a.can_pay_cost(p["repair_cost"])
                    and self.random.random() < p["ea_repair_prob"]):
                v = self.random.choice(repairable)
                if self._activate_edge(u,v,"repair",p["repair_cost"],payer=a):
                    continue
            backups = [v for v in sorted(a.backup_neighbors)
                       if self._edge_eligible(u,v)]
            if (backups and a.can_pay_cost(p["backup_cost"])
                    and self.random.random() < p["ea_backup_prob"]):
                v = self.random.choice(backups)
                if self._activate_edge(u,v,"backup",p["backup_cost"],payer=a):
                    continue
            if (self.new_edges_created < target
                    and a.can_pay_cost(p["new_tie_cost"])
                    and self.random.random() < p["ea_new_tie_prob"]):
                candidates = self._ea_candidates(u)
                if candidates:
                    v = self._select_ea_partner(u, candidates)
                    self._activate_edge(u,v,"new",p["new_tie_cost"],payer=a)

    def _run_recovery_phase(self):
        self.active_mode = self._mode_for_period()
        self.mode_history.append((self.t, self.active_mode))
        if self.active_mode == "DS":
            self._step_directed_stabilization()
        elif self.active_mode == "DA":
            self._step_directed_adaptation()
        elif self.active_mode == "ES":
            self._step_emergent_stabilization()
        elif self.active_mode == "EA":
            self._step_emergent_adaptation()
        else:
            raise ValueError(f"Unknown recovery mode: {self.active_mode}")

    def _attempt_reactivation(self):
        """
        Allow previously inactive firms to return after a minimum delay.
        Reactivated firms return without active ties and must reconnect
        through the normal recovery mechanisms.
        """
        delay = self.params["reactivation_delay"]
        p_reactivate = self.params["reactivation_prob"]

        for agent in self.agents_by_id.values():
            if agent.active:
                continue

            agent.inactive_time += 1

            if agent.inactive_time < delay:
                continue

            if self.exogenous_rng.random() < p_reactivate:
                agent.active = True
                agent.damage = 0.0
                agent.inactive_time = 0

    def step(self):
        if not self.running:
            return
        self.t += 1

        self._attempt_reactivation()

        for agent in self.agents_by_id.values():
            agent.reset_period_budgets()

        self._run_recovery_phase()
        update_system_metrics(self)
        self.datacollector.collect(self)

        if self.t >= self.params["T"]:
            self.running = False
            self._finalize_innovation_experiment()
