#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py
=========
Shared configuration for the analysis pipeline: file paths, the PB
taxonomy, participant ID mapping and shared colour palettes. Figure
typography lives in ``figstyle.py``. All analysis scripts import from
this module so that constants are defined once and stay consistent.

Nothing in this file changes any analysis result. It only centralises
values that were previously duplicated across the scripts.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

#: Project root (the ``ema/`` directory containing this ``src/`` package).
ROOT_DIR = Path(__file__).resolve().parent.parent

#: Raw input data.
DATA_DIR = ROOT_DIR / "data"
RAW_EVENTS_PATH = DATA_DIR / "events_20260324.csv"

#: All pipeline outputs (figures + tables) live under outputs/, kept
#: separate from the raw data directory.
OUTPUTS_DIR = ROOT_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
EVENTS_FIGURES_DIR = FIGURES_DIR / "events"
PARTICIPANTS_FIGURES_DIR = FIGURES_DIR / "participants"

#: Enriched event-level file produced by analyze_events.py and consumed by
#: analyze_participants.py. Kept in outputs/tables/ rather than data/ since
#: it is a generated artifact, not raw data.
ENRICHED_EVENTS_PATH = TABLES_DIR / "events_enriched.csv"


def require_input_file(path: str | Path, hint: str = "") -> Path:
    """
    Stop with a clear message, without a traceback, if an input file is missing.

    Parameters
    ----------
    path:
        File the script needs.
    hint:
        Extra sentence telling the user how to provide the file.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"Input file not found at {path}. {hint}".strip())
    return path


#: Message shown when the diary data file is missing.
DATA_FILE_HINT = (
    "The diary data file is expected at data/events_20260324.csv in the "
    "repository root (see README, section Data)."
)


def ensure_output_dirs() -> None:
    """Create all output directories if they do not already exist."""
    for d in (TABLES_DIR, EVENTS_FIGURES_DIR, PARTICIPANTS_FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# PB taxonomy
# ---------------------------------------------------------------------------

#: Maps Italian survey statements to (construct, item_id) tuples.
PB_DICT: dict[str, tuple[str, int]] = {
    "Ho notato una sovrarappresentazione di artisti maschili o generi mainstream":           ("SB", 1),
    "Ho visto pochissime artiste donne o artisti non occidentali essere consigliati":        ("SB", 2),
    "Le raccomandazioni mi sono sembrate stereotipate o influenzate da bias":                ("SB", 3),
    "Il sistema sembrava ignorare scene musicali diverse o di nicchia a cui tengo.":         ("SB", 4),
    "La piattaforma sembrava promuovere canzoni già virali o molto popolari":                ("CI", 1),
    "Ho percepito che artisti commerciali o di grandi etichette erano favoriti rispetto agli indipendenti": ("CI", 2),
    "Sembrava che la raccomandazione fosse pensata per promuovere qualcosa piuttosto che adattarsi ai miei gusti": ("CI", 3),
    "Ho trovato lo stesso contenuto \u201cdi tendenza\u201d su diverse piattaforme":         ("CI", 4),
    "Il sistema continuava a mostrarmi artisti o brani simili risultando ripetitivo":        ("LD", 1),
    "Ho trovato difficile scoprire qualcosa di veramente nuovo o diverso":                   ("LD", 2),
    "Mi sono sentito \u201cintrappolat\u0259\u201d in un loop di contenuti già familiari":   ("LD", 3),
    "Le raccomandazioni non sembravano evolvere nemmeno dopo aver cambiato ciò che ascoltavo": ("LD", 4),
    "Non ne sono sicur\u0259 ma ha catturato la mia attenzione per qualche motivo":          ("O",  1),
}

# ---------------------------------------------------------------------------
# Diary period
# ---------------------------------------------------------------------------

# DOCUMENTED DECISIONS (confirmed by the study lead on 11 November 2026).
# Week 1 runs from 1 December to the end of 7 December 2025. Week 2 runs
# from 8 December to the end of 14 December 2025. Events logged before
# 1 December were an exception, because some participants started before
# they were told to. They count as week 1 and, for events per day, as
# belonging to the first day, 1 December.

#: First day of the diary period (documented decision).
STUDY_START_DATE: datetime = datetime(2025, 12, 1)

#: Last timestamp counted as week 1 (documented decision). Events whose
#: effective date is on or before this value are week 1 (W1), later events
#: are week 2 (W2).
WEEK1_END: datetime = datetime(2025, 12, 7, 23, 59, 59)

#: Last timestamp of week 2 (documented decision). Three events from one
#: participant (P20) were logged on 15 December 2025, after the end of
#: week 2. They are counted in week 2 by decision.
WEEK2_END: datetime = datetime(2025, 12, 14, 23, 59, 59)

#: Treatment of events logged before STUDY_START_DATE (documented
#: decision). "first_day" assigns them the effective date STUDY_START_DATE,
#: so they count as week 1 and as events of the first day.
EARLY_START_TREATMENT: str = "first_day"

#: Length of the diary period in days, from STUDY_START_DATE to WEEK2_END.
#: Used as the denominator for events per participant per day.
DIARY_PERIOD_DAYS: int = (WEEK2_END.date() - STUDY_START_DATE.date()).days + 1

#: Seed for the horizontal jitter of the strip plot in
#: fig00_likert_distributions.png. Seaborn draws the jitter from NumPy's
#: global random state, so the figure changes on every run unless it is
#: seeded. The value is arbitrary. It moves points sideways within their
#: category for display only and affects no statistic.
JITTER_SEED: int = 0

#: Effect sizes (rank-biserial r, Kendall's W) are exploratory. The paper
#: reports no effect sizes and no significance tests. They are kept in the
#: conditional_probs CSVs under this column prefix only, with a companion
#: note. No figure shows them. This is the prefix for exported columns that
#: are not reported in the paper.
EXPLORATORY_PREFIX: str = "exploratory_"

#: Explicit ordering of context levels for sign-consistent effect sizes.
#: ACTIVE before PASSIVE; PRE before POST.
CONTEXT_LEVEL_ORDER: dict[str, list[str]] = {
    "INTENT": ["ACTIVE", "PASSIVE"],  # positive r -> more frequent when ACTIVE
    "TEMP":   ["PRE",    "POST"],     # positive r -> more frequent when PRE
}

#: Display colours per construct, used consistently across all plots in
#: both analyze_events.py and analyze_participants.py.
CONSTRUCT_COLORS: dict[str, str] = {
    "SB": "#4C72B0",
    "CI": "#DD8452",
    "LD": "#55A868",
    "O":  "#808080",
}

#: Mapping from raw Prolific PID to human-readable participant label.
#: Covers all 20 registered participants; only those present in the
#: enriched file will be used. PIDs not in this map receive "unknown".
PID_MAP: dict[str, str] = {
    "5fad4c1caae022024c431a87": "P1",
    "601705a0246e51313e8ed38e": "P2",
    "6061aaa4240115afd962c429": "P3",
    "60ddf71e95896d2595f0e1a5": "P4",
    "6734c4c9540c828f442d4308": "P5",
    "67235c9d9d3b60d42012c74a": "P6",
    "5e98bb599a58620dacc368ee": "P7",
    "5e68bd3b003be1000c6c4d25": "P8",
    "6056beff1b9debccd57761e0": "P9",
    "6744b3a1bcb197cc5ba7ae7b": "P10",
    "5e650addc1f6e527c4689310": "P11",
    "65aebb06582e0bb7d84d3e5d": "P12",
    "5cac98a9aab6e30001e6f297": "P13",
    "6646052f36c0bd14aafa0f0f": "P14",
    "5c5854ac40378900013b70f7": "P15",
    "6582f5dfb83c230e72711bbd": "P16",
    "5eee1b1961f27903554a7bf1": "P17",
    "5e9f15b5c6681408668da700": "P18",
    "5d9091ff391a60058a7711b5": "P19",
    "60e5d3a91561c1746aa12a5e": "P20",
}
