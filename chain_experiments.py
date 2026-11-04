"""Enumerate comparable two-stage recovery experiments.

Each ordered pair is a stage-1 mechanism followed by a stage-2 mechanism.
The four same-mode pairs serve as invariance checks against the single modes.
"""
from itertools import product

BASE_MODES = ("DS", "DA", "ES", "EA")


def all_chain_scenarios(include_baselines=True):
    """Return (label, core_mode, stage_one, stage_two) scenario tuples."""
    baselines = [(mode, mode, mode, mode) for mode in BASE_MODES] if include_baselines else []
    chains = [
        (f"{first}->{second}", "CHAIN", first, second)
        for first, second in product(BASE_MODES, repeat=2)
    ]
    return tuple(baselines + chains)


def one_chain_scenarios(first, second):
    return tuple(
        [(mode, mode, mode, mode) for mode in BASE_MODES] +
        [(f"{first}->{second}", "CHAIN", first, second)]
    )


def display_label(label):
    return label.replace("->", " → ")


# A concise theoretical shortlist for coauthor presentations.
# The four fixed modes are always included for paired comparisons.
PRIORITY_CHAINS = (
    ("DS", "EA"),  # central stabilization -> local experimentation
    ("ES", "EA"),  # local stabilization -> local experimentation
    ("DA", "EA"),  # central adaptation -> local experimentation
    ("EA", "DA"),  # local experimentation -> coordinated adaptation
)


def priority_chain_scenarios(include_baselines=True):
    """Four theoretically salient recovery chains, plus fixed-mode benchmarks."""
    baselines = [(mode, mode, mode, mode) for mode in BASE_MODES] if include_baselines else []
    chains = [
        (f"{first}->{second}", "CHAIN", first, second)
        for first, second in PRIORITY_CHAINS
    ]
    return tuple(baselines + chains)
