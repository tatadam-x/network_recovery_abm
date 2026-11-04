# Network Recovery and Collective Innovation

An agent-based computational model for studying how alternative recovery modes and sequential recovery chains affect the structural reconstruction and innovative functioning of interorganizational networks.

The project includes a Streamlit app for interactive comparisons and a command-line runner for reproducible experiments. Model parameters and calibration choices remain exploratory research assumptions.

## Research design

A common pre-shock network and knowledge distribution are subjected to a common disruption. Recovery then proceeds under one of four mechanisms:

| Recovery mode | Governance | Orientation |
| --- | --- | --- |
| DS — Directed Stabilization | Centralized | Restore pre-shock ties |
| DA — Directed Adaptation | Centralized | Coordinate restoration and selected new ties |
| ES — Emergent Stabilization | Decentralized | Local repair and backup substitution |
| EA — Emergent Adaptation | Decentralized | Local repair, search, and new relationships |

Recovery chains apply one mechanism followed by another. Switching may depend on elapsed recovery periods, connectivity, or restoration progress. Resource and network states carry forward across the transition.

Knowledge is initialized by community-structured Dirichlet draws. The default innovation rule uses a bounded knowledge increment with a complementarity-sensitive premium for collaborative production. Knowledge accumulates in a pre-shock innovation window and a matched post-recovery window; innovation is paused during structural recovery. A paused intact-network reference provides the principal comparison.

## Run the app locally

Use a Python environment with a supported version of Python (3.11 or 3.12 recommended).

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

The app provides three comparison modes:

1. **Recommended chains:** four foundational recovery modes plus DS → EA, ES → EA, DA → EA, and EA → DA.
2. **Custom chain:** four foundational recovery modes and one chosen two-stage chain.
3. **All recovery chains:** four foundational recovery modes and all 16 ordered two-stage combinations (including four same-mode checks).

The standard experiment uses **100 firms**, **10 recovery periods**, and **15 periods in each innovation window**. Paired seeds, alternative innovation formulations, and other technical assumptions appear under **Advanced settings**.

Results prominently report **pre-shock, post-recovery, and paused-intact knowledge production** for equal-length windows, alongside paired percentage differences. Frontier advancement, changes in knowledge inequality, and effective knowledge diversity are also benchmark-adjusted. The charts show **cumulative within-window changes**, directly overlaying both pre-shock and intact-network references. Additional gap plots isolate differences against either reference. Structural recovery charts include pre-shock benchmark lines. All underlying observations are downloadable as CSV files.

## Command-line experiments

Run the recommended set of recovery modes and chains:

```bash
python run_comparison.py --seeds 10
```

Run all ordered chains:

```bash
python run_comparison.py --mode all --seeds 10
```

The runner saves per-seed results, benchmark-adjusted summary tables, recovery and innovation trajectories, **`comparison_benchmark_trajectories.csv`** containing both reference comparisons at every evaluation period, and experiment settings in `comparison_results/`. The app and runner share their comparison logic in `comparison_engine.py`.

## Test the model

```bash
python -m unittest discover -p "test_*.py" -v
```

A local smoke test using the installed Mesa and Streamlit packages is recommended before publishing or sharing a hosted app.

## Deploy online

Upload these files to the **root of your GitHub repository**. In [Streamlit Community Cloud](https://share.streamlit.io/), select that repository and set the main application file to `streamlit_app.py`.

See [DEPLOYMENT.md](DEPLOYMENT.md) for details.

## Documentation and code organization

- `MODEL_SPECIFICATION.md` — modeling assumptions, mechanisms, and measures
- `DEPLOYMENT.md` — local validation and Streamlit Community Cloud checklist
- `model_core.py` — network generation, shocks, recovery modes, and recovery chains
- `innovation_model.py` — knowledge initialization, production, and system-level outcomes
- `comparison_engine.py` — paired experiments shared by app and runner
- `benchmark_reporting.py` — aligned benchmark trajectories and endpoint consistency checks
- `chain_experiments.py` — available standalone recovery modes and recovery-chain sets
- `streamlit_app.py` — interactive application
- `run_comparison.py` — batch experiments and CSV exports
- `test_*.py` — automated checks

## Interpretation

The model produces simulated outcomes conditional on its assumptions. The principal knowledge measures are aggregate knowledge production, knowledge-frontier advancement, and change in the Gini coefficient of firm knowledge stocks. Effective knowledge dimensions and structural outcomes are additional diagnostics. Differences between recovery modes should be assessed across paired seeds and sensitivity conditions; the model is not calibrated to external empirical data.

## License

MIT. See [LICENSE](LICENSE).
