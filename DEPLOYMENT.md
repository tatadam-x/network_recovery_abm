# Deploy the Network Recovery ABM

The Streamlit entrypoint is **`streamlit_app.py`**, located at the repository root.

## Local validation

Create a fresh virtual environment (Python 3.11 or 3.12 recommended), then run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -p "test_*.py" -v
python -m streamlit run streamlit_app.py
```

Start with the recommended-chain comparison and one or two paired seeds. Check the results table, figures, and CSV export. An exhaustive comparison with many seeds can be computationally demanding on hosted infrastructure.

## GitHub layout

Upload **the contents of the extracted archive** to the root of your GitHub repository:

```text
repository/
    streamlit_app.py
    comparison_engine.py
    chain_experiments.py
    model_core.py
    innovation_model.py
    run_comparison.py
    requirements.txt
    README.md
    MODEL_SPECIFICATION.md
    DEPLOYMENT.md
    LICENSE
    .streamlit/
        config.toml
    test_*.py
```

Make sure `.streamlit/config.toml` is included; this folder may be hidden in file browsers. Do not upload the extracted folder as an additional nested directory unless you adjust the application path.

## Streamlit Community Cloud

1. Sign in with GitHub at https://share.streamlit.io/.
2. Choose **Create app** and select your repository and branch.
3. Specify `streamlit_app.py` as the main application file.
4. Select a Python runtime compatible with the dependencies.
5. Deploy and check the resulting hosted URL with a small recommended-chain run.
6. Share the hosted app URL and repository URL with collaborators.

Streamlit installs dependencies from `requirements.txt`. The app does not require API keys or credentials for its simulation calculations.

## Troubleshooting

- **Import errors:** Verify that the application and four model/comparison modules are in the repository root.
- **Dependency errors:** Review Streamlit's deployment logs and the selected Python version.
- **Slow computations:** Reduce paired seeds or use the recommended comparison before the exhaustive screen.
- **Unexpected mechanism settings:** Confirm structured Dirichlet initialization, bounded complementarity-sensitive production, and uniform EA partner selection in **Advanced settings**.
- **Old results still showing:** Select **Run paired comparison** after changing parameters.

For systematic or longer experiments, run `python run_comparison.py --seeds 10` locally and inspect the CSV exports.

Official deployment documentation:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization


## Verify the scientific comparison after deploying

Run the recommended comparison with two paired seeds as a smoke test.
Confirm that the **Aggregate knowledge production** table reports pre-shock,
post-recovery and paused-intact production in separate columns.
The four innovation trajectory charts should include both reference lines,
and the gap charts should offer pre-shock and paused-intact comparisons.

All three innovation windows should begin at zero on the change plots. The
final gap at the last evaluation period should agree with the corresponding
benchmark-adjusted outcome in the tables and CSV exports.
The downloadable `benchmark_trajectories.csv` includes every within-window
change and paired difference for auditing.
