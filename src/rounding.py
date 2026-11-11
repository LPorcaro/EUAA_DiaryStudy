#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rounding.py
===========
Half-up rounding for every number the pipeline reports.

Python's built-in ``round`` and ``format(x, ".2f")`` round halves to even
and operate on the exact binary value of a float, so ``round(3.25, 1)``
gives ``3.2`` and ``round(0.235, 2)`` gives ``0.23`` (because 0.235 is
stored as 0.23499999...). Reported tables are rounded by hand with the
conventional half-up rule, so the code uses the same rule.

Each value is first converted to its shortest round-trip decimal string
(``repr``), which removes binary representation error, and then quantised
with ``decimal.ROUND_HALF_UP``.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd


def round_half_up(x: float, ndigits: int) -> float:
    """Round ``x`` to ``ndigits`` decimals, halves away from zero.

    NaN and infinite values are returned unchanged.
    """
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return x
    quantum = Decimal(1).scaleb(-ndigits)
    return float(Decimal(repr(x)).quantize(quantum, rounding=ROUND_HALF_UP))


def format_half_up(x: float, ndigits: int) -> str:
    """Format ``x`` with exactly ``ndigits`` decimals using half-up rounding."""
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return str(x)
    quantum = Decimal(1).scaleb(-ndigits)
    return str(Decimal(repr(x)).quantize(quantum, rounding=ROUND_HALF_UP))


def round_frame_half_up(df: pd.DataFrame, ndigits: int) -> pd.DataFrame:
    """Apply :func:`round_half_up` to every numeric cell of ``df``."""
    out = df.copy()
    for col in out.columns:
        if (pd.api.types.is_numeric_dtype(out[col])
                and not pd.api.types.is_integer_dtype(out[col])):
            out[col] = out[col].map(lambda v: round_half_up(v, ndigits))
    return out


def format_frame_half_up(df: pd.DataFrame, ndigits: int) -> pd.DataFrame:
    """Return a string DataFrame for heatmap annotations (half-up)."""
    return df.astype(float).map(lambda v: format_half_up(v, ndigits))
