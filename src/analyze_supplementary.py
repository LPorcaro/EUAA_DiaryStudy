#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_supplementary.py
========================
Regenerates the numbers in the paper that previously had no code path.

1.  Platform shares (Section 5.2).
2.  Events per participant per day (Section 5.2).
3.  Table A4, week 1 and week 2 counts, PB category rates and appraisal
    means per participant.
4.  Leave-one-out ranges of the participant-level Spearman correlations
    (Appendix B.3).

Each output is compared with the value quoted in the paper, using half-up
rounding at the precision printed in the paper.

The week split and the early-start treatment follow documented decisions
in ``config.py`` (``STUDY_START_DATE``, ``WEEK1_END``, ``WEEK2_END`` and
``EARLY_START_TREATMENT``).

Not covered, because the inputs are not in the repository
-----------------------------------------------------------
- The exclusion of 11 events (275 registered, 264 analysed).
- Table A1 and the participant characteristics in Section 5.1.
See the README section "Manual upstream steps".

Input
-----
data/events_20260324.csv  (see config.RAW_EVENTS_PATH)

Output
------
outputs/tables/platform_shares.csv
outputs/tables/events_per_day.csv
outputs/tables/table_A4_weekly.csv
outputs/tables/loo_spearman_B3.csv
outputs/tables/supplementary_comparison.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from analyze_events import (
    REVISED_PB_COLS_WITH_O,
    build_revised_binary_matrix,
    get_construct,
    parse_pb_selections,
)
from config import (
    DIARY_PERIOD_DAYS,
    PID_MAP,
    RAW_EVENTS_PATH,
    EARLY_START_TREATMENT,
    STUDY_START_DATE,
    WEEK1_END,
    WEEK2_END,
    TABLES_DIR,
    DATA_FILE_HINT,
    ensure_output_dirs,
    require_input_file,
)
from rounding import format_half_up

LIKERTS = ["ALGO_CAUSE", "FUTURE_USE", "ARTIST_HARM"]
CONSTRUCTS = ("SB", "CI", "LD", "O")

# ---------------------------------------------------------------------------
# Values quoted in the paper. Strings keep the printed precision.
# ---------------------------------------------------------------------------

PAPER_PLATFORM_PCT = {  # Section 5.2
    "Spotify": "74", "YouTube Music": "12", "Other": "14",
}
PAPER_EVENTS_PER_DAY = {  # Section 5.2
    "mean per participant": "2.1", "most active participant": "2.7",
}
# Table A4. Order per row is # ev. W1 W2, SB W1 W2, CI W1 W2,
# LD W1 W2, O W1 W2, ALGO W1 W2, FUT W1 W2, HARM W1 W2.
PAPER_TABLE_A4 = """
P2  15 18 .09 .11 .17 .20 .32 .31 .67 .83 3.1 3.3 2.7 2.6 3.6 3.3
P4  17 21 .27 .37 .12 .19 .15 .08 .35 .24 4.1 4.5 2.6 2.2 3.9 3.7
P5  11  6 .12 .06 .15 .17 .09 .13 .73 .83 3.5 3.8 3.1 3.3 3.2 3.0
P6  18 20 .20 .12 .51 .22 .63 .30 .33 .30 4.2 4.8 3.0 3.0 3.8 4.1
P8  12 11 .06 .00 .75 .76 .46 .30 .08 .00 1.8 1.2 2.5 2.9 4.3 4.5
P9  10  6 .17 .11 .32 .40 .30 .29 .20 .00 3.5 2.8 2.9 3.3 4.6 4.3
P14 16 22 .04 .00 .56 .55 .08 .10 .25 .14 2.3 2.1 3.0 3.0 3.8 4.0
P16 17 13 .49 .21 .65 .38 .60 .33 .41 .31 3.1 4.5 2.5 2.8 4.5 4.3
P20 15 16 .36 .50 .40 .30 .22 .20 .00 .06 4.2 4.1 2.2 2.4 3.9 3.7
"""
TABLE_A4_COLUMNS = (
    ["n_events"] + [f"{c}_rate" for c in CONSTRUCTS] + LIKERTS
)
PAPER_LOO = {  # Appendix B.3
    ("ALGO_CAUSE", "SB1"): ("0.83", ".75", ".90"),
    ("ARTIST_HARM", "LD4"): ("0.83", ".75", ".92"),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_events(path: str | Path = RAW_EVENTS_PATH) -> pd.DataFrame:
    """Load raw events, add binary PB columns, labels, and week."""
    df = pd.read_csv(require_input_file(path, DATA_FILE_HINT))
    pb_mapped, _ = parse_pb_selections(df)
    binary = build_revised_binary_matrix(pb_mapped)
    df = pd.concat([df, binary], axis=1)
    df["P_label"] = df["PID"].map(PID_MAP)
    if df["P_label"].isna().any():
        raise ValueError("PID without a label in config.PID_MAP")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True)
    if EARLY_START_TREATMENT != "first_day":
        raise ValueError(f"Unknown EARLY_START_TREATMENT {EARLY_START_TREATMENT!r}")
    # Events logged before the study start count as events of the first day.
    df["effective_time"] = df["Timestamp"].clip(lower=pd.Timestamp(STUDY_START_DATE))
    df["week"] = (df["effective_time"] > WEEK1_END).map({False: "W1", True: "W2"})
    for c in CONSTRUCTS:
        items = [p for p in REVISED_PB_COLS_WITH_O if get_construct(p) == c]
        df[f"{c}_event_rate"] = df[items].mean(axis=1)
    return df


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def platform_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Event counts and percentages per platform, plus the paper's grouping."""
    counts = df["PLAT"].value_counts()
    out = pd.DataFrame({"platform": counts.index, "n_events": counts.values})
    out["percent"] = 100 * out["n_events"] / len(df)
    out["paper_group"] = out["platform"].where(
        out["platform"].isin(["Spotify", "YouTube Music"]), "Other"
    )
    return out


def events_per_day(df: pd.DataFrame) -> pd.DataFrame:
    """Events per participant per day over DIARY_PERIOD_DAYS."""
    counts = df.groupby("P_label").size().rename("n_events").reset_index()
    counts["days"] = DIARY_PERIOD_DAYS
    counts["events_per_day"] = counts["n_events"] / DIARY_PERIOD_DAYS
    first = df.groupby("P_label")["Timestamp"].min()
    last = df.groupby("P_label")["Timestamp"].max()
    counts["first_event"] = counts["P_label"].map(first)
    counts["last_event"] = counts["P_label"].map(last)
    return counts


def table_a4(df: pd.DataFrame) -> pd.DataFrame:
    """Per participant and week, event counts, PB rates and Likert stats."""
    agg: dict = {"n_events": ("ID", "size")}
    for c in CONSTRUCTS:
        agg[f"{c}_rate"] = (f"{c}_event_rate", "mean")
    for q in LIKERTS:
        agg[q] = (q, "mean")
        agg[f"{q}_sum"] = (q, "sum")
    return df.groupby(["P_label", "week"]).agg(**agg).reset_index()


def loo_spearman(df: pd.DataFrame) -> pd.DataFrame:
    """Full and leave-one-participant-out Spearman rho for every pair."""
    part = df.groupby("P_label")[LIKERTS + REVISED_PB_COLS_WITH_O].mean()
    rows = []
    for q in LIKERTS:
        for pb in REVISED_PB_COLS_WITH_O:
            full = spearmanr(part[q], part[pb])[0]
            loo = [spearmanr(part.drop(p)[q], part.drop(p)[pb])[0]
                   for p in part.index]
            rows.append({"likert": q, "PB": pb, "rho_full": full,
                         "loo_min": min(loo), "loo_max": max(loo)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparison with the paper
# ---------------------------------------------------------------------------

def _decimals(paper_value: str) -> int:
    return len(paper_value.split(".")[1]) if "." in paper_value else 0


def _strip_zero(s: str) -> str:
    """Write 0.83 and .83 the same way for comparison."""
    return s[1:] if s.startswith("0.") else s.replace("-0.", "-.")


def compare(item: str, location: str, paper: str, value: float) -> dict:
    """One comparison row, rounding ``value`` half-up to the paper's precision."""
    code = format_half_up(value, _decimals(paper))
    return {
        "item": item, "location": location, "paper": paper,
        "code_rounded": code, "code_unrounded": value,
        "agrees": _strip_zero(code) == _strip_zero(paper),
    }


def build_comparison(
    plat: pd.DataFrame, epd: pd.DataFrame, a4: pd.DataFrame, loo: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    grouped = plat.groupby("paper_group")["percent"].sum()
    for name, paper in PAPER_PLATFORM_PCT.items():
        rows.append(compare(f"platform share {name} (%)", "Section 5.2",
                            paper, grouped[name]))

    n_total = epd["n_events"].sum()
    rows.append(compare("events per participant per day", "Section 5.2",
                        PAPER_EVENTS_PER_DAY["mean per participant"],
                        n_total / (len(epd) * DIARY_PERIOD_DAYS)))
    rows.append(compare("events per day, most active participant", "Section 5.2",
                        PAPER_EVENTS_PER_DAY["most active participant"],
                        epd["events_per_day"].max()))

    a4i = a4.set_index(["P_label", "week"])
    for line in PAPER_TABLE_A4.strip().splitlines():
        p, *vals = line.split()
        for k, col in enumerate(TABLE_A4_COLUMNS):
            for w, paper in zip(("W1", "W2"), vals[2 * k: 2 * k + 2]):
                rows.append(compare(f"Table A4 {p} {col} {w}", "Table A4",
                                    paper, a4i.loc[(p, w), col]))

    li = loo.set_index(["likert", "PB"])
    for (q, pb), (full, lo, hi) in PAPER_LOO.items():
        rows.append(compare(f"B.3 rho {q} x {pb}", "Appendix B.3", full,
                            li.loc[(q, pb), "rho_full"]))
        rows.append(compare(f"B.3 LOO min {q} x {pb}", "Appendix B.3", lo,
                            li.loc[(q, pb), "loo_min"]))
        rows.append(compare(f"B.3 LOO max {q} x {pb}", "Appendix B.3", hi,
                            li.loc[(q, pb), "loo_max"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_path: str | Path = RAW_EVENTS_PATH,
         output_dir: str | Path = TABLES_DIR) -> pd.DataFrame:
    ensure_output_dirs()
    out = Path(output_dir)
    df = load_events(input_path)

    plat = platform_shares(df)
    epd = events_per_day(df)
    a4 = table_a4(df)
    loo = loo_spearman(df)

    plat.to_csv(out / "platform_shares.csv", index=False)
    epd.to_csv(out / "events_per_day.csv", index=False)
    a4.to_csv(out / "table_A4_weekly.csv", index=False)
    loo.to_csv(out / "loo_spearman_B3.csv", index=False)

    comp = build_comparison(plat, epd, a4, loo)
    comp.to_csv(out / "supplementary_comparison.csv", index=False)

    print(f"Diary period {STUDY_START_DATE:%d %B %Y} to {WEEK2_END:%d %B %Y}, "
          f"week 1 ends {WEEK1_END:%d %B %Y} (documented decisions)")
    early = df[df["Timestamp"] < pd.Timestamp(STUDY_START_DATE)]
    late = df[df["Timestamp"] > pd.Timestamp(WEEK2_END)]
    def _who(sub: pd.DataFrame) -> str:
        n = sub["P_label"].nunique()
        return f"{n} participant" + ("" if n == 1 else "s")
    print(f"Early-start events counted as week 1, first day: {len(early)} ({_who(early)})")
    late_days = ", ".join(sorted({f"{t:%d %B %Y}" for t in late["Timestamp"]}))
    print(f"Events logged after the end of week 2 ({late_days}) and counted in "
          f"week 2 by decision, {len(late)} ({_who(late)})")
    print(f"Diary period for events per day: {DIARY_PERIOD_DAYS} days")
    n_ok = int(comp["agrees"].sum())
    print(f"\n=== Comparison with the paper: {n_ok}/{len(comp)} agree ===")
    disagree = comp[~comp["agrees"]]
    if disagree.empty:
        print("All compared values agree.")
    else:
        print(disagree.to_string(index=False))
        a4i = a4.set_index(["P_label", "week"])
        for item in disagree["item"]:
            parts = item.split()
            if parts[:2] == ["Table", "A4"] and parts[3] in LIKERTS:
                p, q, w = parts[2], parts[3], parts[4]
                r = a4i.loc[(p, w)]
                print(f"  {p} {q} {w} sum={r[f'{q}_sum']} count={r['n_events']} "
                      f"mean={r[q]!r}")
    for name in ("platform_shares", "events_per_day", "table_A4_weekly",
                 "loo_spearman_B3", "supplementary_comparison"):
        print(f"[export] {name} -> {out / (name + '.csv')}")
    return comp


if __name__ == "__main__":
    main()
