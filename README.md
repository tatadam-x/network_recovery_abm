# Network Recovery ABM

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
network_recovery_abm/
    model_core.py
    streamlit_app.py
    requirements.txt
    README.md
