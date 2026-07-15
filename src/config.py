#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py
=========
Shared configuration for the analysis pipeline: file paths, the PB
taxonomy, participant ID mapping, shared colour palettes, and matplotlib
styling. Both ``analyze_events.py`` and ``analyze_participants.py`` import
from this module so that constants are defined once and stay consistent
across the two scripts.

Nothing in this file changes any analysis result. It only centralises
values that were previously duplicated (e.g. ``CONSTRUCT_COLORS`` and the
Linux Libertine font setup appeared, verbatim, in both original scripts).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

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

#: Cluster colour palette (up to 8 clusters), used in analyze_participants.py.
CLUSTER_PALETTE: list[str] = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#264653", "#A8DADC", "#1D3557",
]

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


# ---------------------------------------------------------------------------
# Matplotlib / font setup
# ---------------------------------------------------------------------------

#: Candidate paths for the Linux Libertine font used in the original
#: figures (matches the ACM template body font). The first path that
#: exists on the current machine is used.
_LIBERTINE_CANDIDATES: list[Path] = [
    Path("/usr/share/fonts/opentype/linux-libertine/LinLibertine_R.otf"),
    Path.home() / ".fonts" / "LinLibertine_R.otf",
]
_LIBERTINE_VARIANTS = {
    "R":  "LinLibertine_R.otf",
    "RB": "LinLibertine_RB.otf",
    "RI": "LinLibertine_RI.otf",
    "RBI": "LinLibertine_RBI.otf",
}


def setup_plot_style(font_size: int = 16) -> None:
    """
    Configure matplotlib to match the figures used in the paper.

    Tries to register Linux Libertine (regular, bold, italic, bold-italic)
    so that figures visually match the ACM ``acmsmall`` template body font.
    If the font is not installed on the current machine, this falls back to
    matplotlib's default font and prints a one-line warning instead of
    raising ``FileNotFoundError`` (which is what the original scripts did,
    since the font path was hardcoded to a specific machine).

    This only affects font *rendering*; it has no effect on any computed
    statistic, table, or CSV output.

    Parameters
    ----------
    font_size:
        Base font size applied to text, axis labels, tick labels, and
        legends (matches the original scripts' value of 16).
    """
    libertine_root = next((p.parent for p in _LIBERTINE_CANDIDATES if p.exists()), None)

    if libertine_root is not None:
        for variant_file in _LIBERTINE_VARIANTS.values():
            fm.fontManager.addfont(str(libertine_root / variant_file))
        plt.rcParams["font.family"] = "Linux Libertine O"
    else:
        warnings.warn(
            "Linux Libertine font not found on this machine; falling back "
            "to matplotlib's default font. Figures will render correctly "
            "but with a different typeface than in the paper. Install "
            "Linux Libertine (e.g. `apt install fonts-linuxlibertine`) to "
            "match the original figures exactly.",
            stacklevel=2,
        )

    plt.rcParams["figure.figsize"] = (3.3, 2.5)  # inches, matches ACM single-column width
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size
