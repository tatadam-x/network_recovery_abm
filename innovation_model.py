"""
Knowledge production and evaluation for the network recovery experiment.

This module deliberately does not depend on Mesa or Streamlit. Its functions
implement the revised formal innovation specification and can be unit-tested
independently of the recovery model.
"""
from __future__ import annotations

import numpy as np


def initialize_community_knowledge(
    n, k, communities, initial_total, within_sd, between_diff, rng,
    method="dirichlet", alpha_between=40.0, alpha_within=40.0,
):
    """Initialize equally endowed firms with heterogeneous knowledge shares.

    ``method="dirichlet"`` implements Scott Page's structured-center
    Alternative 2. Community c has expected center
       q_c = (1-delta)*uniform + delta*unit_vector(focal_dimension_c).
    Realized community profile ~ Dirichlet(alpha_between*q_c), followed by
       firm profile ~ Dirichlet(alpha_within*community_profile).
    Larger alpha values mean less dispersion. Focal dimensions are distinct;
    C <= K is required when delta > 0.

    ``method="gaussian"`` provides a Gaussian initialization benchmark:
    community profiles blend a uniform center with Dirichlet(1,...,1)
    and firm deviations are Gaussian, truncated and renormalized.

    All firms start with the same strictly positive total knowledge.
    """
    if n < 1 or k < 1:
        raise ValueError("n and k must be positive")
    if not np.isfinite(initial_total) or initial_total <= 0:
        raise ValueError("initial_total must be finite and positive")
    if not np.isfinite(within_sd) or within_sd < 0:
        raise ValueError("within_sd must be finite and nonnegative")
    if not np.isfinite(between_diff) or not 0 <= between_diff <= 1:
        raise ValueError("between_diff must lie in [0, 1]")
    community_ids = np.asarray(communities, dtype=int)
    if len(community_ids) != n or np.min(community_ids) < 0:
        raise ValueError("communities must contain n nonnegative identifiers")
    unique_communities, index = np.unique(community_ids, return_inverse=True)
    c = len(unique_communities)
    uniform = np.full(k, 1.0 / k)

    if method == "dirichlet":
        if not (np.isfinite(alpha_between) and alpha_between > 0
                and np.isfinite(alpha_within) and alpha_within > 0):
            raise ValueError("Dirichlet concentrations must be finite and positive")
        if c > k and between_diff > 0:
            raise ValueError("Structured Dirichlet centers require C <= K when delta > 0")
        centers = np.tile(uniform, (c, 1)) * (1.0 - between_diff)
        if between_diff > 0:
            for idx in range(c):
                centers[idx, idx] += between_diff
        # At delta=1 the reference center has exact zeros. The correct
        # interpretation is its deterministic, degenerate Dirichlet limit.
        reference = np.empty_like(centers)
        for idx in range(c):
            positive = centers[idx] > 0
            reference[idx] = 0.0
            if positive.sum() == 1:
                reference[idx, positive] = 1.0
            else:
                reference[idx, positive] = rng.dirichlet(
                    alpha_between * centers[idx, positive]
                )
        shares = np.empty((n, k), dtype=float)
        for i, cidx in enumerate(index):
            positive = reference[cidx] > 0
            shares[i] = 0.0
            if positive.sum() == 1:
                shares[i, positive] = 1.0
            else:
                shares[i, positive] = rng.dirichlet(
                    alpha_within * reference[cidx, positive]
                )
    elif method == "gaussian":
        differentiated = rng.dirichlet(np.ones(k), size=c)
        reference = ((1.0 - between_diff) * uniform[None, :]
                     + between_diff * differentiated)
        noise = rng.normal(loc=0.0, scale=within_sd, size=(n, k))
        shares = np.maximum(reference[index] + noise, 0.0)
        totals = shares.sum(axis=1)
        bad = totals <= 0
        if np.any(bad):
            shares[bad] = reference[index[bad]]
            totals = shares.sum(axis=1)
        shares /= totals[:, None]
    else:
        raise ValueError("knowledge initialization must be 'dirichlet' or 'gaussian'")

    # At positive concentration the theoretical Dirichlet is strictly
    # positive on positive-support dimensions; floating-point extremes can
    # underflow to zero. This is a numerical rather than conceptual limit.
    return initial_total * shares, reference


def ces_aggregate(inputs, rho):
    """Unscaled homogeneous CES aggregator; positive rho, nonnegative inputs."""
    values = np.asarray(inputs, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("CES inputs must be a nonempty one-dimensional vector")
    if np.any(values < 0) or np.any(~np.isfinite(values)):
        raise ValueError("CES inputs must be finite and nonnegative")
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("CES rho must be finite and strictly positive")
    return float(np.sum(values ** rho) ** (1.0 / rho))


def knowledge_increment(inputs, params, initial_inputs=None):
    """Knowledge added after a successful self-search or collaboration.

    ``ces_stock``: original Q(k) = A * CES(k).
    ``bounded_ces`` and ``bounded_surplus`` use
    s = cap * [1 - (1-fraction) * CES(k_0)/CES(k)] for their base.
    The bounded_surplus mode adds a capped collaboration-only complementarity
    premium; self-search remains the same as bounded_ces.
    This equals ``fraction*cap`` when evaluated at the initial knowledge
    input, and tends toward ``cap`` as the input stock grows. ``k_0`` is
    fixed at initialization (for a pair, pooled INITIAL firm vectors).
    In an evolving model positive innovations make CES(k)>=CES(k_0).
    """
    rho = params["knowledge_rho"]
    mode = params.get("knowledge_production_rule", "ces_stock")
    current = ces_aggregate(inputs, rho)
    if mode == "ces_stock":
        return params["knowledge_scale"] * current
    if mode not in ("bounded_ces", "bounded_surplus"):
        raise ValueError(f"Unknown knowledge production rule: {mode}")
    cap = params["bounded_max_increment"]
    frac = params["bounded_initial_fraction"]
    if not np.isfinite(cap) or cap <= 0 or not 0 < frac <= 1:
        raise ValueError("Bounded cap must be positive and initial fraction in (0, 1]")
    original = current if initial_inputs is None else ces_aggregate(initial_inputs, rho)
    if current <= 0:
        return 0.0
    return float(np.clip(cap * (1 - (1-frac) * original / current), 0.0, cap))


def collaborative_increment(pooled, own, partner, params, initial_pool=None):
    """Capped collaborative increment and its complementarity diagnostic.

    The baseline bounded increment compares the joint pool with its initial value.
    The complementarity extension also compares the CES of
    the pooled KNOWLEDGE SHARES to the average CES of standalone shares:

        chi = max(0, CES(pool(shares_i, shares_j)) /
                     mean(CES(shares_i), CES(shares_j)) - 1).
        amount = base + (cap-base) * (1-exp(-gamma*chi)).

    This preserves Scott's finite per-event cap; equal-composition partners
    receive no premium, irrespective of knowledge scale. Pooling is controlled
    by the existing decomposability parameter theta. The multiplier gamma
    is shared by every mode and does not depend on partner communities.
    """
    base = knowledge_increment(pooled, params, initial_pool)
    if params.get("knowledge_production_rule", "ces_stock") != "bounded_surplus":
        return base, 0.0, 0.0
    strength = float(params["collab_surplus_strength"])
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("collab_surplus_strength must be finite and nonnegative")
    a, b = np.asarray(own, dtype=float), np.asarray(partner, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or np.any(a < 0) or np.any(b < 0):
        raise ValueError("Knowledge profiles must be matching nonnegative vectors")
    sa = a / a.sum() if a.sum() > 0 else np.zeros_like(a)
    sb = b / b.sum() if b.sum() > 0 else np.zeros_like(b)
    theta = params["decomposability"]
    shares_pool = ((1.0-theta)*np.minimum(sa, sb)
                   + theta*np.maximum(sa, sb))
    rho = params["knowledge_rho"]
    standalone = 0.5 * (ces_aggregate(sa, rho) + ces_aggregate(sb, rho))
    chi = max(0.0, ces_aggregate(shares_pool, rho) / standalone - 1.0) if standalone > 0 else 0.0
    cap = float(params["bounded_max_increment"])
    premium = max(0.0, cap-base) * (-np.expm1(-strength*chi))
    return min(cap, base+premium), chi, premium


def ces_production(inputs, scale, rho):
    """Stock-proportional innovation increment for a sensitivity benchmark."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("CES scale must be finite and strictly positive")
    return float(scale * ces_aggregate(inputs, rho))


def _select_dimension(pool, rng):
    total = float(np.sum(pool))
    if total <= 0:
        return int(rng.integers(len(pool)))
    return int(rng.choice(len(pool), p=pool / total))


def gini_coefficient(values):
    """Population Gini, with Gini(0,...,0) defined as zero."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or np.any(x < 0) or np.any(~np.isfinite(x)):
        raise ValueError("Gini inputs must be a finite nonnegative 1D vector")
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0
    sorted_x = np.sort(x)
    n = len(sorted_x)
    result = (
        (2 * np.dot(np.arange(1, n + 1), sorted_x) / (n * sorted_x.sum()))
        - (n + 1) / n
    )
    return float(np.clip(result, 0.0, 1.0))


def knowledge_metrics(knowledge):
    """System knowledge, mean frontier, total-stock Gini, and dimension Ginis."""
    x = np.asarray(knowledge, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("knowledge must be a nonempty N-by-K array")
    if np.any(x < 0) or np.any(~np.isfinite(x)):
        raise ValueError("knowledge must be finite and nonnegative")
    frontier = np.max(x, axis=0)
    per_dimension_gini = [gini_coefficient(x[:, m]) for m in range(x.shape[1])]
    stocks = x.sum(axis=1)
    shares = np.divide(x, stocks[:, None],
                       out=np.zeros_like(x), where=stocks[:, None] > 0)
    concentration = (shares ** 2).sum(axis=1)
    effective = np.divide(1.0, concentration,
                          out=np.zeros_like(concentration), where=concentration > 0)
    return {
        "total_knowledge": float(x.sum()),
        "frontier_by_dimension": frontier.tolist(),
        "mean_frontier": float(np.mean(frontier)),
        "gini": gini_coefficient(stocks),
        "gini_by_dimension": per_dimension_gini,
        "mean_dimension_gini": float(np.mean(per_dimension_gini)),
        "mean_effective_domains": float(effective.mean()),
        "min_effective_domains": float(effective.min()),
        "fraction_near_specialist": float(np.mean(effective < 1.5)),
    }


def innovation_period(
    knowledge, graph, active, params, rng, communities=None, pre_graph=None,
    reference_knowledge=None,
):
    """One period of knowledge production with optional collaboration diagnostics.

    Every active firm initiates one opportunity; selected partners may participate
    repeatedly. All updates use beginning-of-period knowledge and are applied
    simultaneously. `pre_graph` is used only to label ties as pre-existing or
    newly established; it does not alter innovation decisions.

    `total_knowledge_produced` counts the system-wide knowledge increment.
    A successful collaboration contributes the same amount to BOTH firms.
    Knowledge-distance diagnostics use total variation between normalized
    partner profiles (0 = identical composition; 1 = disjoint composition).
    """
    previous = np.asarray(knowledge, dtype=float)
    is_active = np.asarray(active, dtype=bool)
    if previous.ndim != 2 or len(is_active) != len(previous):
        raise ValueError("Inconsistent knowledge and active-firm dimensions")
    if communities is not None and len(communities) != len(previous):
        raise ValueError("communities must have one identifier per firm")

    p_collab = params["collaboration_prob"]
    p_self = params["pi_self_success"]
    p_joint = params["pi_collab_success"]
    theta = params["decomposability"]
    for name, value in (
        ("collaboration_prob", p_collab),
        ("pi_self_success", p_self),
        ("pi_collab_success", p_joint),
        ("decomposability", theta),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must lie in [0, 1]")

    if reference_knowledge is None:
        # For isolated one-period unit tests, use the current stocks as k_0.
        # Full simulations always pass frozen initial firm profiles.
        original = previous
    else:
        original = np.asarray(reference_knowledge, dtype=float)
        if original.shape != previous.shape or np.any(original < 0):
            raise ValueError("reference_knowledge must match current knowledge")
    gains = np.zeros_like(previous)
    successes = np.zeros(previous.shape[0], dtype=int)
    counts = dict(
        opportunities=0, self_attempts=0, self_successes=0,
        collab_attempts=0, collab_successes=0,
        lapsed_collab=0, total_knowledge_produced=0.0,
        self_knowledge_produced=0.0, collab_knowledge_produced=0.0,
        collab_attempts_cross_community=0,
        collab_successes_cross_community=0,
        collab_knowledge_produced_cross_community=0.0,
        collab_attempts_new_ties=0,
        collab_successes_new_ties=0,
        collab_knowledge_produced_new_ties=0.0,
        collab_profile_distance_sum=0.0,
        collab_pooled_advantage_sum=0.0,
        collab_complementarity_sum=0.0,
        collab_premium_sum=0.0,
    )
    scale = params["knowledge_scale"]
    rho = params["knowledge_rho"]

    for i in np.flatnonzero(is_active):
        i = int(i)
        counts["opportunities"] += 1

        if rng.random() >= p_collab:
            counts["self_attempts"] += 1
            if rng.random() < p_self:
                amount = knowledge_increment(previous[i], params, original[i])
                dimension = _select_dimension(previous[i], rng)
                gains[i, dimension] += amount
                successes[i] += 1
                counts["self_successes"] += 1
                counts["total_knowledge_produced"] += amount
                counts["self_knowledge_produced"] += amount
            continue

        neighbors = sorted(
            int(j) for j in graph.neighbors(i) if is_active[int(j)]
        )
        if not neighbors:
            counts["lapsed_collab"] += 1
            continue

        counts["collab_attempts"] += 1
        j = neighbors[int(rng.integers(len(neighbors)))]
        cross_comm = (
            communities is not None and communities[i] != communities[j]
        )
        new_tie = pre_graph is not None and not pre_graph.has_edge(i, j)
        if cross_comm:
            counts["collab_attempts_cross_community"] += 1
        if new_tie:
            counts["collab_attempts_new_ties"] += 1

        a, b = previous[i], previous[j]
        a_share = a / a.sum() if a.sum() else np.zeros_like(a)
        b_share = b / b.sum() if b.sum() else np.zeros_like(b)
        counts["collab_profile_distance_sum"] += (
            0.5 * float(np.abs(a_share - b_share).sum())
        )
        pooled = (
            (1.0 - theta) * np.minimum(a, b)
            + theta * np.maximum(a, b)
        )
        initial_pool = (
            (1.0 - theta) * np.minimum(original[i], original[j])
            + theta * np.maximum(original[i], original[j])
        )
        potential, complementarity, premium = collaborative_increment(
            pooled, a, b, params, initial_pool
        )
        counts["collab_complementarity_sum"] += complementarity
        counts["collab_premium_sum"] += premium
        independent_mean = 0.5 * (
            knowledge_increment(a, params, original[i])
            + knowledge_increment(b, params, original[j])
        )
        counts["collab_pooled_advantage_sum"] += potential - independent_mean

        if rng.random() < p_joint:
            dimension = _select_dimension(pooled, rng)
            gains[i, dimension] += potential
            gains[j, dimension] += potential
            successes[i] += 1
            successes[j] += 1
            counts["collab_successes"] += 1
            produced = 2.0 * potential
            counts["total_knowledge_produced"] += produced
            counts["collab_knowledge_produced"] += produced
            if cross_comm:
                counts["collab_successes_cross_community"] += 1
                counts["collab_knowledge_produced_cross_community"] += produced
            if new_tie:
                counts["collab_successes_new_ties"] += 1
                counts["collab_knowledge_produced_new_ties"] += produced

    return previous + gains, counts, successes


def compare_knowledge_windows(
    pre_start, pre_end, post_start, post_end, intact_start, intact_end,
    uninterrupted_start=None, uninterrupted_end=None
):
    """Evaluate matched PRE/POST windows against a paused intact-network control.

    The primary counterfactual starts with exactly the same knowledge and
    stochastic state as the treated network at the end of the pre-shock
    benchmark, and experiences the same innovation-free recovery interval.
    Optional uninterrupted continuation is retained as an opportunity-cost
    diagnostic only and must not be interpreted as a matched-start comparison.
    """
    if (uninterrupted_start is None) != (uninterrupted_end is None):
        raise ValueError("Supply both uninterrupted endpoints or neither")
    snapshots = {
        "pre_start": knowledge_metrics(pre_start),
        "pre_end": knowledge_metrics(pre_end),
        "post_start": knowledge_metrics(post_start),
        "post_end": knowledge_metrics(post_end),
        "intact_paused_start": knowledge_metrics(intact_start),
        "intact_paused_end": knowledge_metrics(intact_end),
    }
    if uninterrupted_start is not None:
        snapshots["uninterrupted_start"] = knowledge_metrics(uninterrupted_start)
        snapshots["uninterrupted_end"] = knowledge_metrics(uninterrupted_end)

    def difference(start, end, name):
        if name == "frontier_by_dimension":
            return (
                np.asarray(snapshots[end][name])
                - np.asarray(snapshots[start][name])
            ).tolist()
        return float(snapshots[end][name] - snapshots[start][name])

    results = {"snapshots": snapshots}
    stages = [
        ("pre", "pre_start", "pre_end"),
        ("post", "post_start", "post_end"),
        ("intact_paused", "intact_paused_start", "intact_paused_end"),
    ]
    if uninterrupted_start is not None:
        stages.append(("uninterrupted", "uninterrupted_start", "uninterrupted_end"))
    for stage, left, right in stages:
        results[f"{stage}_knowledge_gain"] = difference(left, right, "total_knowledge")
        results[f"{stage}_frontier_gain"] = difference(left, right, "mean_frontier")
        results[f"{stage}_gini_change"] = difference(left, right, "gini")
        results[f"{stage}_effective_domains_change"] = difference(
            left, right, "mean_effective_domains"
        )
        results[f"{stage}_frontier_gain_by_dimension"] = difference(
            left, right, "frontier_by_dimension"
        )
        results[f"{stage}_mean_dimension_gini_change"] = difference(
            left, right, "mean_dimension_gini"
        )

    for prefix, name in [
        ("R_K", "knowledge_gain"),
        ("R_F", "frontier_gain"),
        ("R_G", "gini_change"),
        ("R_D", "effective_domains_change"),
    ]:
        results[prefix] = results[f"post_{name}"] - results[f"pre_{name}"]
        results[f"{prefix}_vs_intact_paused"] = (
            results[f"post_{name}"] - results[f"intact_paused_{name}"]
        )
        if uninterrupted_start is not None:
            results[f"{prefix}_vs_uninterrupted"] = (
                results[f"post_{name}"] - results[f"uninterrupted_{name}"]
            )

    # Paused reference and treated should begin their evaluation with
    # identical stocks (even if some treated firms are still inactive).
    if not np.allclose(post_start, intact_start, rtol=1e-12, atol=1e-12):
        raise AssertionError("Matched-start reference and treated knowledge differ")

    baseline_gain = results["pre_knowledge_gain"]
    results["R_K_relative"] = (
        results["R_K"] / baseline_gain if baseline_gain > 0 else None
    )
    results["R_F_by_dimension"] = (
        np.asarray(results["post_frontier_gain_by_dimension"])
        - np.asarray(results["pre_frontier_gain_by_dimension"])
    ).tolist()

    return results
