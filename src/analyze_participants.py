#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_participants.py
=======================

Participant-level profiling based on the enriched event-level data produced
by analyze_events.py.

Pipeline
--------
1.  Load enriched event-level data (events_enriched.csv).
        PIDs are mapped to human-readable labels (P1–P20) for tables and
        printouts while the raw Prolific IDs are preserved for traceability.
2.  Aggregate to participant level:
        - PB construct rates  (SB_rate, CI_rate, LD_rate, O_rate)
        - PB construct counts (SB_count, CI_count, LD_count, O_count)
        - Context proportions (TEMP, INTENT, TRIGGERS)
        - Mean Likert scores  (ALGO_CAUSE, FUTURE_USE, ARTIST_HARM)
3.  Export the participant summary to CSV.

Why rates AND counts?
---------------------
Rates capture the *relative* salience of each construct (proportion of
events where a flag was raised).  Counts capture *reporting intensity* —
a participant who flagged CI 20 times is behaviourally different from one
who flagged it 5 times even if both have the same CI rate.  Both dimensions
are included in the participant summary.

Note on sample size
-------------------
N=9 participants.  All participant-level results are exploratory and
hypothesis-generating only.  No inferential claims about generalisability
are warranted at this sample size.

Input
-----
outputs/tables/events_enriched.csv  — output of analyze_events.py step 7b
(see config.ENRICHED_EVENTS_PATH)

Output
------
outputs/tables/participant_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CONSTRUCT_COLORS,
    ENRICHED_EVENTS_PATH,
    PID_MAP,
    TABLES_DIR,
    ensure_output_dirs,
    setup_plot_style,
)

setup_plot_style()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: PB item columns in the enriched file (revised model)
SB_ITEMS = ["SB1", "SB2", "SB3"]
CI_ITEMS = ["CI1", "CI2", "CI3", "CI4", "CI5"]
LD_ITEMS = ["LD1", "LD2", "LD3", "LD4"]
O_ITEMS  = ["O1"]

#: Construct rate column names (mean across items, then mean across events)
RATE_COLS = ["SB_rate", "CI_rate", "LD_rate", "O_rate"]

#: Construct count column names (total item flags summed across all events)
COUNT_COLS = ["SB_count", "CI_count", "LD_count", "O_count"]

#: Likert outcome column names
LIKERT_COLS = ["ALGO_CAUSE", "FUTURE_USE", "ARTIST_HARM"]

# NOTE: CONSTRUCT_COLORS and PID_MAP now live in config.py (imported above)
# so that both analysis scripts stay in sync; they were previously
# duplicated verbatim in each script.

# ---------------------------------------------------------------------------
# Step 1 — Load & validate
# ---------------------------------------------------------------------------

def load_enriched(path: str | Path) -> pd.DataFrame:
    """
    Load the enriched event-level CSV produced by analyze_events.py.

    Each raw Prolific PID is mapped to a human-readable label via
    ``PID_MAP``.  The combined label (``{PID}_{Px}``) is stored in
    ``PID_labeled`` for exports, while ``P_label`` (e.g. ``"P8"``) is
    used in tables and printouts.  Unmapped PIDs receive ``"unknown"``
    and a warning is printed.

    Parameters
    ----------
    path:
        Path to events_enriched.csv.

    Returns
    -------
    Event-level DataFrame with PB binary columns and participant labels.

    Raises
    ------
    ValueError
        If any expected columns are missing from the file.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = (
        SB_ITEMS + CI_ITEMS + LD_ITEMS + O_ITEMS
        + LIKERT_COLS
        + ["PID", "TRIGGERS", "TEMP", "INTENT"]
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in enriched file: {missing}")

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"], dayfirst=True, errors="coerce"
        )
        df = df.sort_values(["PID", "Timestamp"]).reset_index(drop=True)

    # Map PIDs to human-readable labels
    df["P_label"] = df["PID"].map(PID_MAP)
    unmapped = df.loc[df["P_label"].isna(), "PID"].unique()
    if len(unmapped) > 0:
        print(f"  Warning: {len(unmapped)} PID(s) not in PID_MAP: {unmapped}")
    df["P_label"]     = df["P_label"].fillna("unknown")
    df["PID_labeled"] = df["PID"] + "_" + df["P_label"]

    df["event_n"] = df.groupby("PID").cumcount() + 1

    print(f"Loaded {len(df)} events from {len(df['PID'].unique())} participants.")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Aggregate to participant level
# ---------------------------------------------------------------------------

def build_participant_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Aggregate event-level data to one row per participant.

    PB construct **rates** are the mean across items then mean across events,
    keeping all rates on a fair 0–1 scale regardless of construct size.

    PB construct **counts** are the total number of item flags raised across
    all events, capturing absolute reporting intensity.

    Context proportions are normalised to sum to 1 per participant.

    Parameters
    ----------
    df:
        Enriched event-level DataFrame (output of :func:`load_enriched`).

    Returns
    -------
    summary:
        One row per participant with all aggregate features.
    ctx_cols:
        List of context proportion column names added to ``summary``.
    """
    # --- Event counts ---
    event_counts = df.groupby("PID").size().rename("events").reset_index()

    # --- PB rates: mean across items per event, then mean across events ---
    pb_rates = pd.DataFrame({"PID": df["PID"].unique()})
    for construct, items in (
        ("SB", SB_ITEMS), ("CI", CI_ITEMS),
        ("LD", LD_ITEMS),  ("O",  O_ITEMS),
    ):
        rate = (
            df.assign(**{f"__{construct}": df[items].mean(axis=1)})
              .groupby("PID")[f"__{construct}"]
              .mean()
              .rename(f"{construct}_rate")
              .reset_index()
        )
        pb_rates = pb_rates.merge(rate, on="PID")

    # --- PB counts: total flags summed across items and events ---
    pb_counts = pd.DataFrame({"PID": df["PID"].unique()})
    for construct, items in (
        ("SB", SB_ITEMS), ("CI", CI_ITEMS),
        ("LD", LD_ITEMS),  ("O",  O_ITEMS),
    ):
        count = (
            df.groupby("PID")[items]
              .sum()
              .sum(axis=1)
              .rename(f"{construct}_count")
              .reset_index()
        )
        pb_counts = pb_counts.merge(count, on="PID")

    # --- Likert means ---
    likert_means = (
        df.groupby("PID")[LIKERT_COLS]
          .mean()
          .rename(columns={c: f"mean_{c}" for c in LIKERT_COLS})
          .reset_index()
    )

    # --- Context proportions (normalised to sum to 1 per participant) ---
    ctx_parts = []
    for col in ("TEMP", "INTENT", "TRIGGERS"):
        prop = (
            df.groupby(["PID", col])
              .size()
              .unstack(fill_value=0)
              .pipe(lambda x: x.div(x.sum(axis=1).replace(0, 1), axis=0))
        )
        prop.columns = [f"{col}_{c}" for c in prop.columns]
        ctx_parts.append(prop)

    ctx_df   = pd.concat(ctx_parts, axis=1).fillna(0).reset_index()
    ctx_cols = [c for c in ctx_df.columns if c != "PID"]

    # --- Merge all blocks in order ---
    summary = (
        event_counts
        .merge(pb_rates,     on="PID")
        .merge(pb_counts,    on="PID")
        .merge(likert_means, on="PID")
        .merge(ctx_df,       on="PID")
    )

    # Human-readable label and combined PID string
    summary["P_label"]  = summary["PID"].map(PID_MAP).fillna("unknown")
    summary["PID_short"] = summary["PID"] + "_" + summary["P_label"]

    summary["dominant_PB"] = (
        summary[RATE_COLS].idxmax(axis=1).str.replace("_rate", "")
    )

    likert_mean_cols = [f"mean_{c}" for c in LIKERT_COLS]
    print(f"\n=== Participant Summary ({len(summary)} participants) ===")
    print(
        summary[
            ["PID_short", "P_label", "events"]
            + RATE_COLS + COUNT_COLS + likert_mean_cols
        ].to_string(index=False)
    )

    return summary, ctx_cols


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_results(
    summary: pd.DataFrame,
    output_dir: str | Path = TABLES_DIR,
) -> None:
    """
    Export the participant summary to CSV.

    The participant summary retains the full ``PID_labeled`` column
    (e.g. ``5e68bd3b..._P8``) for traceability.

    Parameters
    ----------
    summary:
        Full participant summary.
    output_dir:
        Directory for the output file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Drop plot-only helper column; keep PID_labeled for traceability
    summary.drop(columns=["PID_short"], errors="ignore").to_csv(
        out / "participant_summary.csv", index=False
    )

    print(f"\n[export] participant_summary -> {out / 'participant_summary.csv'}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(
    input_path: str | Path = ENRICHED_EVENTS_PATH,
    output_dir: str | Path = TABLES_DIR,
) -> None:
    """
    Run the full participant-level profiling pipeline.

    Steps
    -----
    1.  Load events_enriched.csv; map PIDs to P-labels.
    2.  Aggregate to participant level (rates + counts + context + Likert).
    3.  Export CSV.

    Parameters
    ----------
    input_path:
        Path to events_enriched.csv. Defaults to the file produced by
        analyze_events.py (see config.ENRICHED_EVENTS_PATH). Run
        analyze_events.py first if this file does not exist yet.
    output_dir:
        Directory for the output CSV.
    """
    ensure_output_dirs()

    if not Path(input_path).exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run analyze_events.py first to "
            f"generate the enriched events file it depends on."
        )

    # 1. Load
    df = load_enriched(input_path)

    # 2. Aggregate
    summary, ctx_cols = build_participant_summary(df)

    # 3. Export
    export_results(summary, output_dir)


if __name__ == "__main__":
    main()