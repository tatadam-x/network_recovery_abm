# Network Recovery Models

An agent-based model and browser app for comparing post-shock network recovery regimes, resilience trajectories, and innovation outcomes in interorganizational networks.

## Overview

This project implements a Python/Mesa agent-based model (ABM) of network recovery following disruptive shocks. It compares how different recovery regimes shape short-term stability, medium-run resilience, structural repair, and innovation outcomes in interorganizational networks.

The project includes:

- `model_core.py`: core ABM logic
- `streamlit_app.py`: browser-based Streamlit interface for running model comparisons interactively

## Recovery Regimes

The model compares five recovery regimes:

- **DS** — Directed Stabilization
- **DA** — Directed Adaptation
- **ES** — Emergent Stabilization
- **EA** — Emergent Adaptation
- **HY** — Hybrid

These regimes differ in governance mode and recovery orientation.

## Main Features

- Compare all five recovery regimes under the same scenario
- Adjust shock type, shock severity, network type, network size, average degree, and simulation settings
- Display compact or full results tables
- Visualize:
  - trajectory plots
  - bar comparison plots
  - joint regime comparison plot
- Track recovery, performance, structural change, and innovation outcomes

## Files

```text
network_recovery_models/
    model_core.py
    streamlit_app.py
    requirements.txt
    README.md
    LICENSE
```

## Installation

Create and activate a Python environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

If you do not yet have a `requirements.txt`, the core packages are:

```bash
python -m pip install streamlit mesa networkx numpy pandas matplotlib
```

## Run the App

From the project folder, run:

```bash
python -m streamlit run streamlit_app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Typical Workflow

1. Adjust scenario inputs in the sidebar
2. Click **Run model comparison**
3. Review:
   - scenario summary
   - plots
   - compact or full results tables
   - joint regime comparison

## Model Scope

The current implementation focuses on:

- post-shock network recovery
- comparative resilience across recovery regimes
- medium-run performance dynamics
- structural repair and tie formation
- innovation accumulation over time

## Notes

- The app is designed as an interface layer on top of the shared model logic in `model_core.py`
- All five regimes are run under the same scenario settings for controlled comparison
- The Streamlit app is intended for interactive exploration and presentation, while `model_core.py` remains the main source of model logic

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
