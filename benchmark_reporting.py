"""Window-aligned innovation reports for scientific comparison.

Every curve is an accumulated change from period zero of its own 15-period
window. The pre-shock and paused-intact references are common to all modes
for a given paired seed, so they are deduplicated before averaging.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Label, original stock/level column, accumulated-change output suffix.
WINDOW_METRICS = {
    "knowledge": "total_knowledge",
    "frontier": "mean_frontier",
    "gini": "gini",
    "diversity": "mean_effective_domains",
}
PHASES = {"pre": "pre", "post": "post", "intact_paused": "intact"}


def prepare_benchmark_trajectories(innovation: pd.DataFrame) -> pd.DataFrame:
    """Pair each mode's post-recovery trajectory with pre and intact windows.

    Returns one row per (seed, mode, period) with within-window changes for
    all four outcomes and differences vs the two references. Comparison
    periods are aligned by *time since the respective window began*.
    """
    required = {"seed", "mode", "phase", "period", *WINDOW_METRICS.values()}
    absent = required.difference(innovation.columns)
    if absent:
        raise ValueError(f"Innovation trajectories missing columns: {sorted(absent)}")

    frames = {}
    for phase, prefix in PHASES.items():
        part = innovation.loc[
            innovation["phase"].eq(phase),
            ["seed", "mode", "period", *WINDOW_METRICS.values()],
        ].copy()
        if part.empty:
            raise ValueError(f"Missing innovation phase {phase!r}")
        if phase != "post":
            # Model references should be precisely matched across modes under
            # the same seed. Detect inadvertent changes before deduplicating.
            grouped = part.groupby(["seed", "period"], sort=False)
            for col in WINDOW_METRICS.values():
                spans = grouped[col].agg(lambda x: x.max() - x.min())
                if (spans.abs() > 1e-9).any():
                    raise ValueError(
                        f"{phase} reference {col} differs across paired modes"
                    )
            part = part.drop_duplicates(["seed", "period"]).drop(columns="mode")
            id_cols = ["seed"]
        else:
            id_cols = ["seed", "mode"]
            if part.duplicated(["seed", "mode", "period"]).any():
                raise ValueError("Duplicate post-recovery observations")
        if not (part["period"] == 0).any():
            raise ValueError(f"Phase {phase} lacks window-start observation")
        part = part.sort_values(id_cols + ["period"])
        for output, col in WINDOW_METRICS.items():
            base_values = part.loc[part["period"].eq(0), id_cols + [col]]
            if base_values.duplicated(id_cols).any():
                raise ValueError(f"Ambiguous window start for {phase}")
            base_values = base_values.rename(columns={col: "_initial"})
            part = part.merge(base_values, on=id_cols, how="left",
                              validate="many_to_one")
            if part["_initial"].isna().any():
                raise ValueError(f"No starting value for some {phase} rows")
            part[f"{prefix}_{output}_change"] = part[col] - part["_initial"]
            part = part.drop(columns="_initial")
        selected = id_cols + ["period"] + [
            f"{prefix}_{out}_change" for out in WINDOW_METRICS
        ]
        frames[phase] = part[selected]

    merged = frames["post"].merge(
        frames["pre"], on=["seed", "period"], how="left",
        validate="many_to_one"
    ).merge(
        frames["intact_paused"], on=["seed", "period"], how="left",
        validate="many_to_one"
    )
    expected = len(frames["post"])
    if len(merged) != expected or merged.isna().any().any():
        raise ValueError("Benchmark and post-recovery windows are not aligned")
    for outcome in WINDOW_METRICS:
        for ref in ("pre", "intact"):
            merged[f"{outcome}_gap_vs_{ref}"] = (
                merged[f"post_{outcome}_change"] -
                merged[f"{ref}_{outcome}_change"]
            )
    return merged


def check_summary_alignment(summary: pd.DataFrame,
                            trajectories: pd.DataFrame,
                            atol: float = 1e-8) -> None:
    """Ensure the published endpoint numbers equal the plot endpoints."""
    final = (trajectories.sort_values("period")
             .groupby(["seed", "mode"], as_index=False).tail(1))
    final = final.rename(columns={
        col: f"_trajectory_{col}" for col in final.columns
        if col not in ("seed", "mode")
    })
    combined = summary.merge(final, on=["seed", "mode"],
                             validate="one_to_one")
    if len(combined) != len(summary):
        raise AssertionError("Missing endpoint data for some runs")
    metrics = {
        "knowledge": ("knowledge_gain", "pre_knowledge_gain",
                      "intact_knowledge_gain",
                      "knowledge_gap_vs_pre", "knowledge_gap_vs_intact"),
        "frontier": ("frontier_gain", "pre_frontier_gain",
                     "intact_frontier_gain",
                     "frontier_gap_vs_pre", "frontier_gap_vs_intact"),
        "gini": ("gini_change", "pre_gini_change",
                 "intact_gini_change", "gini_gap_vs_pre", "gini_gap_vs_intact"),
        "diversity": ("effective_domains_change", "pre_effective_domains_change",
                      "intact_effective_domains_change",
                      "effective_domains_gap_vs_pre",
                      "effective_domains_gap_vs_intact"),
    }
    for outcome, fields in metrics.items():
        measured = [
            f"post_{outcome}_change", f"pre_{outcome}_change",
            f"intact_{outcome}_change",
            f"{outcome}_gap_vs_pre", f"{outcome}_gap_vs_intact",
        ]
        for col, expected in zip(fields, measured):
            if not np.allclose(combined[col], combined[f"_trajectory_{expected}"],
                               rtol=1e-9, atol=atol, equal_nan=True):
                raise AssertionError(
                    f"Reported {col} does not match trajectory {expected}"
                )
