
# =========================
# Shared ABM core module
# =========================

import numpy as np
import networkx as nx
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
    "T": 50,                        # number of post-shock periods
    "seed": 1,                      # random seed
    "regime": "HY",                 # DS, DA, ES, EA, HY
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
    "central_pool_frac": 0.40,      # share of slack pooled in centralized regimes
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
    "es_backup_activation_prob": 0.10,
    "da_new_tie_cost": 1.00,
    "ea_new_tie_cost": 1.5,
    "ea_repair_prob": 0.35,
    "ea_backup_prob": 0.25,
    "ea_new_tie_prob": 0.30,
    "ea_cross_comm_prob": 0.20,

    # -------------------------
    # Knowledge and innovation
    # -------------------------
    "K": 6,                         # number of knowledge dimensions
    "p_innovate": 0.05,             # probability of marginal innovation per period
    "absorptive_optimal": 0.40,     # ideal cognitive distance
    "absorptive_width": 0.25,       # tolerance around ideal distance
    "absorb_gain": 0.20,            # gain from successful absorption
    "exchange_prob": 0.20,          # probability of complementary exchange
    "recombine_min": 0.20,          # minimum distance for recombination
    "recombine_max": 0.80,          # maximum distance for recombination
    "p_recombine": 0.08,            # probability of recombination
    "attention_budget": 3,          # max partners interacted with per period
    "forgetting_rate": 0.01,        # decay in nonreinforced knowledge gains

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
        self.attention_budget = self.model.params["attention_budget"]

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
        Reset per-period repair and attention limits.
        """
        self.repair_capacity = self.model.params["repair_capacity"]
        self.attention_budget = self.model.params["attention_budget"]

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

    # =========================================================
    # Knowledge and innovation methods
    # =========================================================

    def try_incremental_innovation(self):
        """
        With probability p_innovate, generate a small improvement
        in one knowledge dimension.
        """
        p = self.model.params

        if self.model.random.random() < p["p_innovate"]:
            k = self.model.random.randrange(len(self.knowledge))
            self.knowledge[k] += p["absorb_gain"]
            self.innovation_stock += 1
            return True
        return False

    def cognitive_distance(self, other_agent):
        """
        Compute normalized cognitive distance from another firm.
        """
        diff = np.abs(self.knowledge - other_agent.knowledge)
        return float(np.mean(diff))

    def can_absorb_from(self, other_agent):
        """
        Check whether incoming knowledge falls within the absorptive band.
        """
        p = self.model.params
        d = self.cognitive_distance(other_agent)

        lower = p["absorptive_optimal"] - p["absorptive_width"]
        upper = p["absorptive_optimal"] + p["absorptive_width"]

        return lower <= d <= upper

    def absorb_from(self, other_agent):
        """
        Absorb part of another firm's knowledge if within the absorptive band.
        """
        p = self.model.params

        if self.can_absorb_from(other_agent):
            gains = np.where(other_agent.knowledge > self.knowledge, p["absorb_gain"], 0.0)
            self.knowledge = self.knowledge + gains

    def try_exchange_with(self, other_agent):
        """
        Exchange complementary knowledge with a connected partner.
        """
        p = self.model.params

        if self.model.random.random() < p["exchange_prob"]:
            self.knowledge = np.maximum(self.knowledge, other_agent.knowledge)
            other_agent.knowledge = np.maximum(other_agent.knowledge, self.knowledge)

    def try_recombine(self):
        """
        Attempt a simple recombination event within the firm's own knowledge stock.
        """
        p = self.model.params

        if len(self.knowledge) < 2:
            return False

        i, j = self.model.random.sample(range(len(self.knowledge)), 2)
        d = abs(self.knowledge[i] - self.knowledge[j])

        if p["recombine_min"] <= d <= p["recombine_max"]:
            if self.model.random.random() < p["p_recombine"]:
                new_value = 0.5 * (self.knowledge[i] + self.knowledge[j])
                self.knowledge[i] = max(self.knowledge[i], new_value)
                self.innovation_stock += 1
                return True

        return False

    def apply_forgetting(self):
        """
        Apply mild decay to knowledge values.
        """
        p = self.model.params
        self.knowledge = np.maximum(
            self.knowledge * (1.0 - p["forgetting_rate"]),
            0.0
        )


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
    Create FirmAgent objects and register them by node id.
    """
    n = model.params["N"]
    c = model.params["C"]
    k = model.params["K"]

    model.agents_by_id = {}

    for node_id in range(n):
        community = node_id % c
        knowledge_vector = np.random.uniform(low=0.0, high=1.0, size=k)

        agent = FirmAgent(
            model=model,
            unique_id=node_id,
            community=community,
            knowledge_vector=knowledge_vector
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
    """
    Shock a subset of high-centrality firms.
    shock_severity is interpreted as a share of all firms,
    with shocked firms drawn from the hub target pool.
    """
    n = model.params["N"]
    severity = model.params["shock_severity"]
    target_frac = model.params["hub_target_frac"]
    p_inactive = model.params["p_inactive"]
    p_tie_fail = model.params["p_tie_fail"]

    ranked = sorted(
        model.agents_by_id.values(),
        key=lambda a: a.pre_centrality,
        reverse=True
    )

    n_target_pool = max(1, int(n * target_frac))
    target_pool = ranked[:n_target_pool]

    n_shocked = max(1, int(n * severity))
    n_shocked = min(n_shocked, len(target_pool))

    shocked_agents = model.random.sample(target_pool, n_shocked)

    for agent in shocked_agents:
        shock_agent(model, agent, p_inactive, p_tie_fail)


def apply_random_shock(model):
    """
    Shock a random subset of firms.
    """
    n = model.params["N"]
    severity = model.params["shock_severity"]
    p_inactive = model.params["p_inactive"]
    p_tie_fail = model.params["p_tie_fail"]

    all_agents = list(model.agents_by_id.values())
    n_shocked = max(1, int(n * severity))
    shocked_agents = model.random.sample(all_agents, n_shocked)

    for agent in shocked_agents:
        shock_agent(model, agent, p_inactive, p_tie_fail)


def apply_clustered_shock(model):
    """
    Shock firms concentrated in one community.
    """
    n = model.params["N"]
    severity = model.params["shock_severity"]
    p_inactive = model.params["p_inactive"]
    p_tie_fail = model.params["p_tie_fail"]

    target_community = model.random.randrange(model.params["C"])
    candidates = [
        agent for agent in model.agents_by_id.values()
        if agent.community == target_community
    ]

    n_shocked = max(1, int(n * severity))
    n_select = min(n_shocked, len(candidates))
    shocked_agents = model.random.sample(candidates, n_select)

    for agent in shocked_agents:
        shock_agent(model, agent, p_inactive, p_tie_fail)


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
    Mesa model for network recovery regimes and resilience.
    """

    def __init__(self, params):
        super().__init__()

        self.params = params
        self.seed = params["seed"]
        self.random.seed(self.seed)
        np.random.seed(self.seed)

        self.t = 0
        self.regime = params["regime"]
        self.baseline_perf = params["baseline_perf"]
        self.running = True
        self.hybrid_switched = False

        self.G = nx.Graph()
        self.G_pre = nx.Graph()
        self.backup_edges = set()

        self.agents_by_id = {}

        self.perf_net = self.baseline_perf
        self.lcc_share = 1.0
        self.frag = 1
        self.active_pre_shock_ties = 0
        self.new_ties_count = 0

        build_initial_network(self)
        create_agents(self)
        assign_pre_neighbors(self)
        generate_backup_ties(self)
        store_pre_shock_graph(self)
        compute_pre_shock_metrics(self)
        apply_initial_shock(self)
        update_system_metrics(self)

        self.datacollector = DataCollector(
            model_reporters={
                "time": lambda m: m.t,
                "regime": lambda m: m.regime,
                "perf_net": lambda m: m.perf_net,
                "lcc_share": lambda m: m.lcc_share,
                "fragmentation": lambda m: m.frag,
                "active_pre_shock_ties": lambda m: m.active_pre_shock_ties,
                "new_ties_count": lambda m: m.new_ties_count,
            }
        )

        self.datacollector.collect(self)

    def _step_directed_stabilization(self):
        repair_cost = self.params["repair_cost"]
        pool_frac = self.params["central_pool_frac"]

        central_pool = 0.0
        for agent in self.agents_by_id.values():
            if agent.active and agent.slack > 0:
                contribution = pool_frac * agent.slack
                agent.slack -= contribution
                central_pool += contribution

        candidate_repairs = []
        seen_edges = set()

        for agent in self.agents_by_id.values():
            if not agent.active:
                continue

            broken_pre_ties = [
                nbr for nbr in agent.pre_neighbors
                if not self.G.has_edge(agent.unique_id, nbr)
            ]

            for nbr in broken_pre_ties:
                other = self.agents_by_id[nbr]
                if not other.active:
                    continue

                edge = tuple(sorted((agent.unique_id, nbr)))
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)

                candidate_score = agent.pre_centrality + other.pre_centrality
                if agent.community != other.community:
                    candidate_score += self.params["bridge_bonus"]

                candidate_repairs.append((edge, candidate_score))

        candidate_repairs.sort(key=lambda x: x[1], reverse=True)

        for (i, j), candidate_score in candidate_repairs:
            if central_pool < repair_cost:
                break

            if self.G.has_edge(i, j):
                continue

            if self.agents_by_id[i].active and self.agents_by_id[j].active:
                self.G.add_edge(i, j)
                central_pool -= repair_cost

        active_agents = [a for a in self.agents_by_id.values() if a.active]
        if active_agents and central_pool > 0:
            refund = central_pool / len(active_agents)
            for agent in active_agents:
                agent.slack += refund

    def _step_directed_adaptation(self):
        repair_cost = self.params["repair_cost"]
        pool_frac = self.params["central_pool_frac"]

        new_tie_cost = self.params["da_new_tie_cost"]
        new_tie_budget = 1
        adaptive_rewiring_pool = new_tie_budget * new_tie_cost

        central_pool = 0.0
        for agent in self.agents_by_id.values():
            if agent.active and agent.slack > 0:
                contribution = pool_frac * agent.slack
                agent.slack -= contribution
                central_pool += contribution

        candidate_repairs = []
        seen_edges = set()

        for agent in self.agents_by_id.values():
            if not agent.active:
                continue

            broken_pre_ties = [
                nbr for nbr in agent.pre_neighbors
                if not self.G.has_edge(agent.unique_id, nbr)
            ]

            for nbr in broken_pre_ties:
                other = self.agents_by_id[nbr]
                if not other.active:
                    continue

                edge = tuple(sorted((agent.unique_id, nbr)))
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)

                candidate_score = agent.pre_centrality + other.pre_centrality
                if agent.community != other.community:
                    candidate_score += self.params["bridge_bonus"]

                candidate_repairs.append((edge, candidate_score))

        candidate_repairs.sort(key=lambda x: x[1], reverse=True)

        repaired_nodes = set()

        for (i, j), candidate_score in candidate_repairs:
            if central_pool < repair_cost:
                break

            if self.G.has_edge(i, j):
                continue

            if i in repaired_nodes or j in repaired_nodes:
                continue

            if self.agents_by_id[i].active and self.agents_by_id[j].active:
                self.G.add_edge(i, j)
                central_pool -= repair_cost
                repaired_nodes.add(i)
                repaired_nodes.add(j)

        candidate_new_ties = []
        seen_new_edges = set()

        for agent in self.agents_by_id.values():
            if not agent.active:
                continue

            broken_count = sum(
                1 for nbr in agent.pre_neighbors
                if not self.G.has_edge(agent.unique_id, nbr)
            )

            if broken_count == 0:
                continue

            for other_id, other in self.agents_by_id.items():
                if other_id == agent.unique_id:
                    continue
                if not other.active:
                    continue
                if self.G.has_edge(agent.unique_id, other_id):
                    continue
                if other_id in agent.pre_neighbors:
                    continue
                if other_id in agent.backup_neighbors:
                    continue
                if agent.community == other.community:
                    continue

                edge = tuple(sorted((agent.unique_id, other_id)))
                if edge in seen_new_edges:
                    continue
                seen_new_edges.add(edge)

                candidate_score = 0.0

                if agent.community != other.community:
                    candidate_score += 2 * self.params["bridge_bonus"]

                candidate_score += 1.5 * other.pre_centrality
                candidate_score += 0.2 * broken_count

                shared_broken = len(set(agent.pre_neighbors) & set(other.pre_neighbors))
                candidate_score += 0.1 * shared_broken

                candidate_new_ties.append((edge, candidate_score))

        candidate_new_ties.sort(key=lambda x: x[1], reverse=True)

        new_ties_added = 0
        used_new_nodes = set()

        for (i, j), candidate_score in candidate_new_ties:
            if new_ties_added >= new_tie_budget:
                break
            if adaptive_rewiring_pool < new_tie_cost:
                break
            if self.G.has_edge(i, j):
                continue
            if i in used_new_nodes or j in used_new_nodes:
                continue

            if self.agents_by_id[i].active and self.agents_by_id[j].active:
                self.G.add_edge(i, j)
                self.agents_by_id[i].new_neighbors.add(j)
                self.agents_by_id[j].new_neighbors.add(i)

                adaptive_rewiring_pool -= new_tie_cost
                new_ties_added += 1
                used_new_nodes.add(i)
                used_new_nodes.add(j)

        active_agents = [a for a in self.agents_by_id.values() if a.active]
        if active_agents and central_pool > 0:
            refund = central_pool / len(active_agents)
            for agent in active_agents:
                agent.slack += refund

    def _step_emergent_stabilization(self):
        repair_cost = self.params["repair_cost"]
        backup_cost = 1.5
        backup_activation_prob = self.params["es_backup_activation_prob"]

        agent_ids = list(self.agents_by_id.keys())
        self.random.shuffle(agent_ids)

        for agent_id in agent_ids:
            agent = self.agents_by_id[agent_id]

            if not agent.active:
                continue

            if agent.repair_capacity <= 0:
                continue

            broken_pre_ties = [
                nbr for nbr in agent.pre_neighbors
                if not self.G.has_edge(agent.unique_id, nbr)
            ]

            if len(broken_pre_ties) == 0:
                continue

            repairable_pre_ties = [
                nbr for nbr in broken_pre_ties
                if self.agents_by_id[nbr].active
            ]

            unavailable_pre_ties = [
                nbr for nbr in broken_pre_ties
                if not self.agents_by_id[nbr].active
            ]

            if repairable_pre_ties and agent.can_pay_cost(repair_cost):
                nbr = self.random.choice(repairable_pre_ties)
                self.G.add_edge(agent.unique_id, nbr)
                agent.pay_cost(repair_cost)
                agent.repair_capacity -= 1
                continue

            available_backups = [
                nbr for nbr in agent.backup_neighbors
                if not self.G.has_edge(agent.unique_id, nbr)
            ]

            if (
                unavailable_pre_ties
                and not agent.used_backup_substitution
                and agent.current_degree() < agent.pre_degree()
                and available_backups
                and agent.can_pay_cost(backup_cost)
                and self.random.random() < backup_activation_prob
            ):
                nbr = self.random.choice(available_backups)
                other = self.agents_by_id[nbr]

                if other.active:
                    self.G.add_edge(agent.unique_id, nbr)
                    agent.pay_cost(backup_cost)
                    agent.repair_capacity -= 1
                    agent.used_backup_substitution = True

    def _step_emergent_adaptation(self):
        repair_cost = self.params["repair_cost"]
        backup_cost = self.params["backup_cost"]
        local_radius = self.params["local_search_radius"]

        new_tie_cost = self.params["ea_new_tie_cost"]
        repair_prob = self.params["ea_repair_prob"]
        backup_prob = self.params["ea_backup_prob"]
        new_tie_prob = self.params["ea_new_tie_prob"]
        cross_comm_prob = self.params["ea_cross_comm_prob"]

        agent_ids = list(self.agents_by_id.keys())
        self.random.shuffle(agent_ids)

        for agent_id in agent_ids:
            agent = self.agents_by_id[agent_id]

            if not agent.active:
                continue

            if agent.repair_capacity <= 0:
                continue

            broken_pre_ties = [
                nbr for nbr in agent.pre_neighbors
                if not self.G.has_edge(agent.unique_id, nbr)
            ]

            if len(broken_pre_ties) == 0:
                continue

            if agent.can_pay_cost(repair_cost) and self.random.random() < repair_prob:
                nbr = self.random.choice(broken_pre_ties)
                other = self.agents_by_id[nbr]

                if other.active and not self.G.has_edge(agent.unique_id, nbr):
                    self.G.add_edge(agent.unique_id, nbr)
                    agent.pay_cost(repair_cost)
                    agent.repair_capacity -= 1
                    continue

            available_backups = [
                nbr for nbr in agent.backup_neighbors
                if (
                    not self.G.has_edge(agent.unique_id, nbr)
                    and self.agents_by_id[nbr].active
                )
            ]

            if (
                available_backups
                and agent.can_pay_cost(backup_cost)
                and self.random.random() < backup_prob
            ):
                nbr = self.random.choice(available_backups)
                self.G.add_edge(agent.unique_id, nbr)
                agent.pay_cost(backup_cost)
                agent.repair_capacity -= 1
                continue

            if not agent.can_pay_cost(new_tie_cost):
                continue

            if self.random.random() >= new_tie_prob:
                continue

            candidates = []

            for other_id, other in self.agents_by_id.items():
                if other_id == agent.unique_id:
                    continue
                if not other.active:
                    continue
                if self.G.has_edge(agent.unique_id, other_id):
                    continue
                if other_id in agent.pre_neighbors:
                    continue
                if other_id in agent.backup_neighbors:
                    continue

                try:
                    dist = nx.shortest_path_length(self.G_pre, agent.unique_id, other_id)
                except nx.NetworkXNoPath:
                    continue

                if dist > local_radius:
                    continue

                score = 0.0

                shared_broken = len(set(broken_pre_ties) & set(other.pre_neighbors))
                score += shared_broken

                if other.community != agent.community:
                    if self.random.random() < cross_comm_prob:
                        score += self.params["bridge_bonus"]

                other_degree = len(list(self.G.neighbors(other_id)))
                score += 0.05 * other_degree

                candidates.append((other_id, score))

            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                top_score = candidates[0][1]
                top_candidates = [cid for cid, score in candidates if score == top_score]

                nbr = self.random.choice(top_candidates)
                self.G.add_edge(agent.unique_id, nbr)
                agent.new_neighbors.add(nbr)
                self.agents_by_id[nbr].new_neighbors.add(agent.unique_id)

                agent.pay_cost(new_tie_cost)
                agent.repair_capacity -= 1

    def _run_recovery_phase(self):
        if self.regime == "DS":
            self._step_directed_stabilization()
        elif self.regime == "DA":
            self._step_directed_adaptation()
        elif self.regime == "ES":
            self._step_emergent_stabilization()
        elif self.regime == "EA":
            self._step_emergent_adaptation()
        elif self.regime == "HY":
            self._step_hybrid()
        else:
            raise ValueError(f"Unknown regime: {self.regime}")

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

            if self.random.random() < p_reactivate:
                agent.active = True
                agent.damage = 0.0
                agent.inactive_time = 0

    def _run_learning_phase(self):
        for agent in self.agents_by_id.values():
            if not agent.active:
                continue
            agent.try_incremental_innovation()

        for agent in self.agents_by_id.values():
            if not agent.active:
                continue

            neighbors = list(self.G.neighbors(agent.unique_id))

            k = min(agent.attention_budget, len(neighbors))
            sampled_neighbors = self.random.sample(neighbors, k)

            for nbr_id in sampled_neighbors:
                other = self.agents_by_id[nbr_id]
                if other.active:
                    agent.absorb_from(other)
                    agent.try_exchange_with(other)

            agent.try_recombine()
            agent.apply_forgetting()

    def _check_hybrid_switch(self):
        if self.regime != "HY" or self.hybrid_switched:
            return

        if self.perf_net < self.params["hybrid_switch_perf"]:
            return

        remaining_broken_pre_ties = 0

        for agent in self.agents_by_id.values():
            if not agent.active:
                continue

            for nbr in agent.pre_neighbors:
                if not self.G.has_edge(agent.unique_id, nbr):
                    remaining_broken_pre_ties += 1

        remaining_broken_pre_ties = remaining_broken_pre_ties / 2

        if remaining_broken_pre_ties > 0:
            self.hybrid_switched = True

    def _step_hybrid(self):
        """
        Hybrid trajectory:
        Recovery begins with directed stabilization and later shifts
        to directed adaptation.

        Order of action:
        1. start in directed stabilization
        2. monitor collective performance across periods
        3. once the hybrid switch threshold is reached and unresolved
           pre-shock structural disruption remains, shift to directed
           adaptation in the next period
        """
        if not self.hybrid_switched:
            self._step_directed_stabilization()
        else:
            self._step_directed_adaptation()

    def step(self):
        self.t += 1

        self._attempt_reactivation()

        for agent in self.agents_by_id.values():
            agent.reset_period_budgets()

        self._run_recovery_phase()
        update_system_metrics(self)

        self._run_learning_phase()
        update_system_metrics(self)

        self._check_hybrid_switch()

        self.datacollector.collect(self)

        if self.t >= self.params["T"]:
            self.running = False
