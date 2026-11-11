#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plotting.py
===========
Tiny helper used by both analysis scripts to save the current matplotlib
figure to disk *and* display it, instead of only calling ``plt.show()``.

In the original scripts, 14 of 15 figures were only shown interactively
(e.g. in a Jupyter session) and never written to disk; the PNGs bundled in
figs/ were exported by hand at some point and are not reproducible by
re-running the scripts. ``save_and_show`` closes that gap without touching
any plot's content, styling, or data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import figstyle  # noqa: F401  (every plotting module imports the shared style)


def save_and_show(
    fig: plt.Figure,
    out_dir: str | Path,
    filename: str,
    *,
    dpi: int = 300,
    show: bool = True,
) -> Path:
    """
    Save ``fig`` to ``out_dir/filename`` and optionally display it.

    Parameters
    ----------
    fig:
        The figure to save (e.g. the return value of ``plt.subplots()`` or
        ``plt.gcf()``).
    out_dir:
        Destination directory; created if it does not exist.
    filename:
        Output filename, e.g. ``"fig7_trigger_temp_intent.png"``.
    dpi:
        Resolution in dots per inch (default 300, matching the one
        ``savefig`` call present in the original scripts).
    show:
        Whether to also call ``plt.show()`` after saving. Set to ``False``
        for headless / batch runs.

    Returns
    -------
    Path to the saved PNG file. A PDF with the same stem is saved too.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[figure] {filename} -> {out_path}")

    # Vector copy with the same name and a .pdf suffix. Fonts are embedded
    # according to figstyle.EMBEDDING_RCPARAMS (TrueType, Type 42).
    pdf_path = out_path.with_suffix(".pdf")
    if pdf_path != out_path:
        # No creation date, so that repeated runs give byte-identical PDFs.
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight",
                    metadata={"CreationDate": None})
        print(f"[figure] {pdf_path.name} -> {pdf_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path
