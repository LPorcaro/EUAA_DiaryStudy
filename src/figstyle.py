#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figstyle.py
===========
Single place for figure typography. Every plotting script imports this
module. It sets the text font family, the mathematics font and the font
embedding settings.

Fonts
-----
The camera-ready layout of Taylor & Francis sets body text in Minion Pro
(serif) with Myriad Pro as the sans-serif companion. These are the
intended targets, ``SERIF_FAMILY`` and ``SANS_FAMILY``. Neither is
installed on the machine used for the audit.

``FONT_SOURCE`` selects between the target and the fallback in use.

- ``"fallback"`` (default) uses ``FALLBACK_SERIF_FAMILY``, currently
  Crimson Pro, a static TrueType serif.
- ``"target"`` uses the target family chosen by ``FONT_CHOICE``
  (``"serif"`` for Minion Pro, ``"sans"`` for Myriad Pro).

Both can be overridden without editing code with the environment
variables ``FIGSTYLE_FONT_SOURCE`` and ``FIGSTYLE_FONT``.

The Crimson Pro files are registered from where they are found, through
``FALLBACK_FONT_DIR`` or TeX Live's ``kpsewhich``, with
``matplotlib.font_manager.addfont``. No copy is made,
no system font setting is changed and no cache rebuild is needed. To use a
different fallback that fontconfig and Matplotlib already know (for example
Crimson Text once it is installed), change ``FALLBACK_SERIF_FAMILY`` only.
Families without an entry in ``FALLBACK_FONT_FILES`` are looked up in
Matplotlib's font list.

No silent substitution
----------------------
If the fallback cannot be found, ``setup_plot_style`` raises
``FileNotFoundError`` and no figure is drawn. If a target family is not
installed, a ``UserWarning`` names the family and the substitute.

Embedding
---------
``pdf.fonttype`` and ``ps.fonttype`` are 42, so fonts are embedded as
TrueType (Type 42) instead of Type 3. Crimson Pro has TrueType (glyf)
outlines, so the embedded data matches the declared font type. Fonts with
CFF outlines (most .otf files, including Minion Pro and Myriad Pro) are
written by Matplotlib in a TrueType font dictionary although the embedded
data is CFF. pdffonts reports this as a mismatch between font type and
embedded font file. Check the exported PDFs before using a CFF font.

Mathematics font
----------------
``mathtext.fontset`` is ``custom`` with the text family for roman, italic
and bold, so mathematical text matches the surrounding text. Glyphs the
text font lacks come from ``MATH_FALLBACK``. With the fallback font,
italic mathematics uses ``FALLBACK_MATH_ITALIC`` (STIXGeneral Italic), so
the ρ in fig06 is italic, as in the manuscript. Roman and bold mathematics
stay in Crimson Pro. STIX is a serif designed for mathematics and ships
with Matplotlib as TrueType. Review this setting if the targets are used.

This module only affects rendering. It changes no statistic or table.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Font families
# ---------------------------------------------------------------------------

#: Intended target, serif family of the Taylor & Francis camera-ready layout.
SERIF_FAMILY: str = "Minion Pro"

#: Intended target, sans-serif companion family of the same layout.
SANS_FAMILY: str = "Myriad Pro"

#: Fallback in use while the targets are not installed. Change this one
#: constant to switch to another installed serif (for example Crimson Text).
FALLBACK_SERIF_FAMILY: str = "Crimson Pro"

#: ``"fallback"`` (default) or ``"target"``.
FONT_SOURCE: str = os.environ.get("FIGSTYLE_FONT_SOURCE", "fallback")

#: Target face used when ``FONT_SOURCE`` is ``"target"``, ``"serif"``
#: (default) or ``"sans"``.
FONT_CHOICE: str = os.environ.get("FIGSTYLE_FONT", "serif")

#: Folder that holds the fallback font files. Set it here or through the
#: environment variable ``FIGSTYLE_FONT_DIR``. A relative path is read
#: from the repository root. If it is None, each file is located with
#: TeX Live's ``kpsewhich``, which finds the crimsonpro package wherever
#: TeX Live is installed.
_FONT_DIR_SETTING = os.environ.get("FIGSTYLE_FONT_DIR")
FALLBACK_FONT_DIR: Path | None = Path(_FONT_DIR_SETTING) if _FONT_DIR_SETTING else None

#: Font files registered with ``addfont`` for families that fontconfig does
#: not list. These Crimson Pro files are static TrueType files with glyf
#: outlines (checked with fontTools). They are read in place, not copied.
FALLBACK_FONT_FILES: dict[str, list[str]] = {
    "Crimson Pro": [
        "CrimsonPro-Regular.ttf",
        "CrimsonPro-Italic.ttf",
        "CrimsonPro-Bold.ttf",
        "CrimsonPro-BoldItalic.ttf",
    ],
}

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _locate_font_file(name: str) -> Path | None:
    """Return the path of a fallback font file, or None if it is not found."""
    if FALLBACK_FONT_DIR is not None:
        folder = FALLBACK_FONT_DIR
        if not folder.is_absolute():
            folder = _REPO_ROOT / folder
        path = folder / name
        return path if path.is_file() else None
    if shutil.which("kpsewhich") is None:
        return None
    found = subprocess.run(["kpsewhich", name], capture_output=True, text=True).stdout.strip()
    return Path(found) if found and Path(found).is_file() else None

#: Mathtext fallback for glyphs missing from the text font. STIX for a
#: serif face, STIX Sans for a sans-serif face. Both ship with Matplotlib
#: as TrueType files and embed as Type 42.
MATH_FALLBACK: dict[str, str] = {"serif": "stix", "sans": "stixsans"}

#: Italic mathematics face for the fallback font only. Crimson Pro has no
#: Greek letters, and the manuscript sets rho in italic, so italic
#: mathematics comes from STIXGeneral Italic. Roman and bold mathematics
#: stay in the fallback family.
#: REVIEW THIS SETTING IF THE TARGET FONTS (Minion Pro, Myriad Pro) ARE USED.
#: It is not applied when FONT_SOURCE is "target".
FALLBACK_MATH_ITALIC: str = "STIXGeneral:italic"

#: Embedding settings. 42 means TrueType (Type 42), never Type 3.
EMBEDDING_RCPARAMS: dict[str, int] = {"pdf.fonttype": 42, "ps.fonttype": 42}

#: Base font size and default figure size, unchanged from the original
#: setup. Individual plots override the figure size explicitly.
BASE_FONT_SIZE: int = 16
DEFAULT_FIGSIZE: tuple[float, float] = (3.3, 2.5)


def selected_family(source: str | None = None,
                    choice: str | None = None) -> tuple[str, str]:
    """
    Return ``(family, kind)`` for the given source and choice.

    ``kind`` is ``"serif"`` or ``"sans"`` and selects the mathtext fallback.
    The fallback is a serif face.
    """
    source = source or FONT_SOURCE
    choice = choice or FONT_CHOICE
    if source not in ("fallback", "target"):
        raise ValueError(f"FONT_SOURCE must be 'fallback' or 'target', not {source!r}")
    if choice not in ("serif", "sans"):
        raise ValueError(f"FONT_CHOICE must be 'serif' or 'sans', not {choice!r}")
    if source == "fallback":
        return FALLBACK_SERIF_FAMILY, "serif"
    return (SERIF_FAMILY if choice == "serif" else SANS_FAMILY), choice


def resolved_font_file(family: str) -> str | None:
    """Path of the regular face Matplotlib uses for ``family``, or None."""
    try:
        return fm.findfont(fm.FontProperties(family=family),
                           fallback_to_default=False)
    except ValueError:
        return None


def register_fallback(family: str = FALLBACK_SERIF_FAMILY) -> None:
    """
    Make the fallback family available to Matplotlib, or fail clearly.

    Raises
    ------
    FileNotFoundError
        If a listed font file is missing, or if the family cannot be found
        after registration.
    """
    names = FALLBACK_FONT_FILES.get(family, [])
    located = {name: _locate_font_file(name) for name in names}
    missing = [name for name, path in located.items() if path is None]
    if missing:
        where = (f"in {FALLBACK_FONT_DIR}" if FALLBACK_FONT_DIR is not None
                 else "through kpsewhich (TeX Live)")
        raise FileNotFoundError(
            f"Fallback font {family!r} cannot be registered. File(s) {missing} "
            f"not found {where}. Figures are not drawn in a substitute font. "
            "Set FIGSTYLE_FONT_DIR (or FALLBACK_FONT_DIR in src/figstyle.py) "
            "to the folder that holds the files, or change FALLBACK_SERIF_FAMILY."
        )
    for path in located.values():
        fm.fontManager.addfont(str(path))
    if resolved_font_file(family) is None:
        raise FileNotFoundError(
            f"Fallback font {family!r} is not known to Matplotlib. Add its "
            "files to FALLBACK_FONT_FILES in src/figstyle.py, or install it "
            "and rebuild the Matplotlib font cache. Figures are not drawn in "
            "a substitute font."
        )


def setup_plot_style(font_size: int = BASE_FONT_SIZE,
                     source: str | None = None,
                     choice: str | None = None) -> str:
    """
    Apply font family, mathematics font, embedding and size settings.

    Parameters
    ----------
    font_size:
        Base size for text, axis labels, tick labels and legends.
    source:
        ``"fallback"`` or ``"target"``. Defaults to ``FONT_SOURCE``.
    choice:
        ``"serif"`` or ``"sans"`` for the target. Defaults to ``FONT_CHOICE``.

    Returns
    -------
    The family name in use.
    """
    source = source or FONT_SOURCE
    family, kind = selected_family(source, choice)

    if source == "fallback":
        register_fallback(family)
    elif resolved_font_file(family) is None:
        used = fm.findfont(fm.FontProperties(family=family))
        warnings.warn(
            f"Target font family {family!r} is not installed. Matplotlib "
            f"will substitute {used}. Install the font or set FONT_SOURCE to "
            "'fallback' in src/figstyle.py before producing final figures.",
            stacklevel=2,
        )
        # The warning above states the substitution once. Matplotlib would
        # otherwise log the same "findfont" message for every text element.
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    plt.rcParams["font.family"] = family
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = family
    plt.rcParams["mathtext.it"] = (
        FALLBACK_MATH_ITALIC if source == "fallback" else f"{family}:italic"
    )
    plt.rcParams["mathtext.bf"] = f"{family}:bold"
    plt.rcParams["mathtext.fallback"] = MATH_FALLBACK[kind]
    plt.rcParams.update(EMBEDDING_RCPARAMS)

    plt.rcParams["figure.figsize"] = DEFAULT_FIGSIZE
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size
    return family
