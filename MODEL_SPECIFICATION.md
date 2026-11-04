# Network Recovery ABM — Model Specification

This document records the implemented computational assumptions for the shareable research model. Parameters are preliminary and may be revised following coauthor discussion.

## 1. Experimental sequence

1. Generate an undirected pre-shock network and assign firms to communities.
2. Initialize each firm with the same total knowledge endowment but a community-dependent knowledge composition.
3. Run the innovation mechanism for a pre-shock benchmark window.
4. Apply an exogenous shock that can disrupt relationships and inactivate firms.
5. Apply a selected recovery mode or chain for a fixed recovery interval. Knowledge does **not** change during this interval.
6. Resume innovation on the recovered network over a post-recovery evaluation window, carrying forward each firm's pre-shock knowledge stock.
7. Compare outcomes with a **paused intact-network counterfactual** that has the same starting knowledge and innovation-free interval. An uninterrupted reference is available as an optional diagnostic.

The pre-shock and post-recovery innovation windows have equal length.

## 2. Recovery structure

| Recovery mode | Information and authority | Main recovery actions |
| --- | --- | --- |
| **Directed Stabilization (DS)** | Centralized, network-wide prioritization | Use pooled resources to restore important broken original ties; no novelty |
| **Directed Adaptation (DA)** | Centralized, network-wide prioritization | Restore important ties and deliberately form cross-community ties, subject to a novelty budget |
| **Emergent Stabilization (ES)** | Firm-level information and local resources | Restore prior ties and selectively activate backup ties |
| **Emergent Adaptation (EA)** | Firm-level information and local resources | Restore, substitute, and form new ties through current-network local search and occasional distant exploration |

Recovery is constrained by firms' available resources, per-period endpoint action capacities, and tie-formation costs. Centrally coordinated recovery modes pool a portion of available resources; decentralized modes pay from local resources. Recovery does not inspect firms' knowledge vectors.

The recovery environment supports random, hub-targeted, and clustered shocks, and multiple initial network topologies. Exogenous firm-reactivation draws are paired across recovery modes for a given seed.

### Recovery chains

A chain specifies an initial mode and a subsequent mode from {DS, DA, ES, EA}. A single permanent transition may occur after a set number of periods, when the largest connected-component share reaches a threshold, or when original-tie restoration reaches a threshold. Network ties, resources, and inactive-firm states carry forward; there is no reset at transition. The standard switching point is after five completed periods of a ten-period recovery interval.

Recommended comparisons include DS → EA, ES → EA, DA → EA, and EA → DA. The exhaustive screen includes all 16 ordered pairs alongside four standalone recovery modes.

## 3. Initial knowledge distribution

Firms have K nonnegative knowledge dimensions and identical initial total knowledge. Each community c has a distinct expected knowledge center:

\[
q_c=(1-\delta)u+\delta e_{d(c)},
\]

where u is uniform over K dimensions and e is a focal-dimension unit vector.

Community reference profiles and firm profiles are drawn hierarchically:

\[
\mu_c\sim \mathrm{Dirichlet}(\alpha_B q_c),\qquad
s_i\sim \mathrm{Dirichlet}(\alpha_W\mu_{c(i)}).
\]

Initial firm knowledge is its share vector multiplied by the common initial knowledge total. Larger concentrations imply less dispersion; the separation parameter \(\delta\) increases systematic differentiation of community centers. Distinct focal dimensions require C ≤ K when \(\delta>0\).

A Gaussian initialization rule remains an optional sensitivity setting.

## 4. Knowledge production

Each active firm initiates one innovation opportunity per evaluation period: independent self-search with probability 1 − p_C or collaboration with probability p_C. A collaborating firm chooses one active current neighbor uniformly; partners may participate in more than one innovation event per period. An attempted collaboration lapses if the initiating firm has no available partner. Success probabilities and knowledge-production parameters are identical across recovery modes.

A common CES aggregator acts on the relevant input profile:

\[
Q(x)=\left(\sum_{m=1}^{K}x_m^\rho\right)^{1/\rho}.
\]

For collaboration, the pooled profile is constructed dimension-wise as

\[
p_m=(1-\theta)\min(k_{im},k_{jm})+\theta\max(k_{im},k_{jm}),
\]

where \(\theta\) represents task decomposability.

The default **bounded** increment is

\[
B(x;x_0)=\mathrm{clip}\left(A\left[1-(1-f)\frac{Q(x_0)}{Q(x)}\right],0,A\right),
\]

where \(x_0\) is the corresponding frozen initial knowledge input, A is the per-recipient cap, and f controls the initial fraction of that cap. For collaboration, x and x₀ denote current and initial pooled inputs.

Collaborative innovation may also obtain a bounded complementarity premium. Define normalized partner shares u and v, and a pooled normalized-input vector w using the same min/max pooling rule. Then

\[
\chi=\max\left(0,\frac{Q(w)}{[Q(u)+Q(v)]/2}-1\right),
\]

\[
B_{joint}=B+(A-B)\left(1-e^{-\gamma\chi}\right).
\]

Here γ is a common complementarity-sensitivity parameter. This construction keeps the increment at or below A. Self-search uses the bounded base increment without a collaborative premium.

A successful event adds the increment to one selected dimension, with the selection probability proportional to existing knowledge shares (self-search) or pooled shares (collaboration). Both partners receive the increment in successful collaboration. Knowledge updates are applied simultaneously at the end of each period. The bounded-only and stock-proportional CES formulations remain available for sensitivity analysis.

## 5. Outcomes and diagnostics

The three system-level knowledge outcomes compare **changes over equal-length windows**:

- **Aggregate knowledge production:** change in the sum of all firms' knowledge stocks.
- **Knowledge-frontier advancement:** change in the **mean** across dimensions of the maximum firm-level knowledge in each dimension. This is the sum of domain frontiers divided by the fixed number of knowledge dimensions, K.
- **Knowledge inequality:** change in the Gini coefficient of total knowledge across firms.

Each is evaluated relative to its pre-shock benchmark and, for the principal comparison, relative to the matched-start paused intact-network reference.

### Window-aligned reporting in the application

Let \(X_w(t)\) denote a system-level knowledge metric measured \(t\) periods into
innovation window \(w\in\{PRE, POST, INTACT\}\). Each trajectory is plotted as
\(Δ X_w(t)=X_w(t)-X_w(0)\), for \(t=0,\ldots,H\). This avoids comparing raw
knowledge stocks observed at different points in the model's history.

Two paired differences are retained at every evaluation period:
- **Principal matched-start comparison:** \(Δ X_{POST}(t)-Δ X_{INTACT}(t)\), shown prominently in the application.
- **Earlier historical pre-shock comparison:** \(Δ X_{POST}(t)-Δ X_{PRE}(t)\), available under **Pre-shock benchmark diagnostics**.

The main innovation charts present post-recovery gaps relative to the paused
intact network, with zero meaning equal change since the evaluation window
began. Absolute post-recovery and paused-intact trajectories are available in a
collapsed secondary panel. The pre-shock trajectory and post-minus-pre gap
are available on request in the diagnostic panel. The pre-shock window begins
with lower accumulated knowledge, which affects growth independently of
network reconstruction.

The pre-shock and paused-intact references are common to all recovery modes
within each paired seed and each reference is included **once** in averages.
Endpoint values at \(t=H\) equal the reported benchmark-adjusted system
outcomes. Production percentages use the paired reference's
within-window production as their denominator.

The paused intact network begins its innovation evaluation with the same
knowledge profiles as each recovered network, following an equally long
innovation-free recovery interval. The post-recovery network can still have
inactive firms or missing/novel ties. Innovation during structural recovery is
outside the current specification, so the charts assess functioning after
the recovery interval.

The structural recovery charts start at \(t=0\), immediately after disruption.
Their reference lines show the pre-shock restoration share (100%), largest
component share (100%), new-tie count (zero), and mean number of
cross-community ties in the intact starting network.

For Gini, a **positive** benchmark-adjusted value means a larger increase in
inequality (or smaller decrease); it is not automatically a beneficial outcome.

Diagnostics include mean effective knowledge domains, defined for firm i by \(1/\sum_m s_{im}^2\), cross-community collaboration, collaborations using new ties, average partner knowledge distance and complementarity, new and restored ties, the largest connected component, and recovery resources spent.

The legacy composite structural performance index in the core is not a principal system-level outcome and is not used for chain switching.

## 6. Default settings

| Setting | Default |
| --- | --- |
| Firms | 100 |
| Communities | 4 |
| Knowledge dimensions | 6 |
| Recovery periods | 10 |
| Pre-shock / post-recovery innovation periods | 15 / 15 |
| Initialization | Structured Dirichlet |
| Dirichlet concentrations \(\alpha_B, \alpha_W\) | 40, 40 |
| Community differentiation \(\delta\) | 0.70 |
| Collaboration probability | 0.50 |
| Bounded increment cap and initial fraction | 0.24, 0.25 |
| Collaboration premium strength \(\gamma\) | 2.0 |
| CES exponent \(\rho\) | 0.50 |
| Task decomposability \(\theta\) | 0.75 |
| Self-search / collaboration success probabilities | 0.55 / 0.60 |
| EA new-partner selection | Uniform over eligible candidates |
| Primary reference | Paused intact network |

## 7. Interpretation and limitations

The recovery modes jointly vary information access, allocation rules, search scope, and choice probabilities. They are stylized, interpretable treatments; identifying the independent causal contribution of any one design dimension would require additional decomposition. Recovery affects post-recovery innovation through the available collaboration graph and partner knowledge compositions. Innovation is not evaluated during structural recovery. Parameter calibration and substantive effect sizes require further robustness analysis across seeds, shocks, network structures, and knowledge-task settings.
