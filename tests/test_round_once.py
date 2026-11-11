"""Every reported value must be rounded once, half-up, from full precision.

Run with ``python tests/test_round_once.py``.

Part A feeds values such as 0.1234999 into the export and annotation code.
Rounded once to 3 decimals this is 0.123. Any intermediate rounding (to 4,
5, 6 ... decimals) turns it into 0.1235 first and then 0.124, so the test
fails.

Part B runs ``analyze_events.main`` into a temporary directory, captures
every figure annotation, and compares every reported number with an
independent full-precision recomputation rounded once.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pingouin as pg  # noqa: E402
from scipy.stats import friedmanchisquare, rankdata, spearmanr  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

import analyze_events as ae  # noqa: E402
import analyze_supplementary as sup  # noqa: E402
from config import RAW_EVENTS_PATH  # noqa: E402
from rounding import format_half_up, round_half_up  # noqa: E402

PB = ae.REVISED_PB_COLS_WITH_O
LIKERTS = ["ALGO_CAUSE", "FUTURE_USE", "ARTIST_HARM"]
TRAP3 = 0.1234999999   # once -> 0.123, twice -> 0.124
TRAP2 = 0.1249999999   # once -> 0.12,  twice -> 0.13


# ---------------------------------------------------------------------------
# Part A. Synthetic traps
# ---------------------------------------------------------------------------

def test_traps_in_export_functions() -> None:
    assert round_half_up(TRAP3, 3) == 0.123
    assert format_half_up(TRAP2, 2) == "0.12"

    idx = ["SB1"]
    prob = pd.DataFrame({"A": [TRAP3]}, index=idx)
    ci = pd.DataFrame({"A": [(TRAP3, TRAP3)]}, index=idx)
    eff = pd.DataFrame({"PB": ["SB1"], "effect_size": [TRAP3]})
    with tempfile.TemporaryDirectory() as tmp:
        out = ae.export_conditional_results_csv(
            prob, ci, None, eff, context_name="T", pb_order=idx,
            output_path=Path(tmp) / "c.csv",
        )
        assert (out.loc["SB1", ["A_prob", "A_ci_low", "A_ci_high",
                                "exploratory_effect_size"]] == 0.123).all()
        corr = pd.DataFrame({"SB1": [TRAP3]}, index=["ALGO_CAUSE"])
        long = ae.export_likert_pb_csv(corr, Path(tmp) / "l.csv")
        assert long["spearman_rho"].iloc[0] == 0.123

    annot = ae.format_frame_half_up(pd.DataFrame({"x": [TRAP2]}), 2)
    assert annot.iloc[0, 0] == "0.12"


# ---------------------------------------------------------------------------
# Part B. Full pipeline against independent recomputation
# ---------------------------------------------------------------------------

def _load() -> pd.DataFrame:
    df = pd.read_csv(RAW_EVENTS_PATH)
    mapped, _ = ae.parse_pb_selections(df)
    return pd.concat([df, ae.build_revised_binary_matrix(mapped)], axis=1)


def _levels(df: pd.DataFrame, ctx: str) -> list[str]:
    return ae.CONTEXT_LEVEL_ORDER.get(ctx, sorted(df[ctx].unique()))


def _effect_full(df: pd.DataFrame, ctx: str, pb: str) -> float:
    """Independent effect size, matching the documented definition."""
    lv = _levels(df, ctx)
    p = df.groupby(["PID", ctx])[pb].mean().unstack().reindex(columns=lv)
    if len(lv) == 2:
        d = (p[lv[0]] - p[lv[1]]).dropna()
        if len(d) < 3:
            return np.nan
        nz = d[d != 0].values
        if len(nz) < 3:
            return 0.0
        ranks = rankdata(np.abs(nz))
        return 1.0 - 2.0 * ranks[nz < 0].sum() / (len(nz) * (len(nz) + 1) / 2)
    p = p.dropna()
    groups = [p[x].values for x in lv]
    if all(np.array_equal(groups[0], g) for g in groups[1:]):
        return 0.0
    return friedmanchisquare(*groups)[0] / (len(p) * (len(lv) - 1))


def _run_main(tmp: Path) -> dict:
    """Run analyze_events.main into ``tmp`` and capture annotations."""
    captured: dict = {"heatmaps": [], "edges": None}
    orig_heatmap = ae.sns.heatmap
    orig_edges = ae.nx.draw_networkx_edge_labels
    saved = {
        "TABLES_DIR": ae.TABLES_DIR,
        "EVENTS_FIGURES_DIR": ae.EVENTS_FIGURES_DIR,
        "ENRICHED_EVENTS_PATH": ae.ENRICHED_EVENTS_PATH,
        "save_and_show": ae.save_and_show,
        "ensure_output_dirs": ae.ensure_output_dirs,
    }

    def heatmap(data, *a, **k):
        captured["heatmaps"].append((data.copy(), k.get("annot")))
        return orig_heatmap(data, *a, **k)

    def edges(G, pos, *a, **k):
        captured["edges"] = (G, k.get("edge_labels"))
        return orig_edges(G, pos, *a, **k)

    ae.sns.heatmap = heatmap
    ae.nx.draw_networkx_edge_labels = edges
    ae.TABLES_DIR = tmp
    ae.EVENTS_FIGURES_DIR = tmp
    ae.ENRICHED_EVENTS_PATH = tmp / "events_enriched.csv"
    ae.save_and_show = lambda fig, *a, **k: matplotlib.pyplot.close(fig)
    ae.ensure_output_dirs = lambda: None
    try:
        ae.main()
    finally:
        ae.sns.heatmap = orig_heatmap
        ae.nx.draw_networkx_edge_labels = orig_edges
        for name, value in saved.items():
            setattr(ae, name, value)
    return captured


def test_pipeline_rounds_once() -> None:
    df = _load()
    checked = 0
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        cap = _run_main(tmp)

        for ctx in ("INTENT", "TEMP", "TRIGGERS"):
            levels = _levels(df, ctx)
            cond = pd.read_csv(tmp / f"conditional_probs_{ctx}.csv", index_col="PB")
            cons = pd.read_csv(tmp / f"construct_probs_{ctx}.csv", index_col="construct")
            for lvl in levels:
                g = df[df[ctx] == lvl]
                n = len(g)
                for pb in PB:
                    k = int(g[pb].sum())
                    lo, hi = proportion_confint(k, n, method="beta")
                    for col, full in ((f"{lvl}_prob", k / n),
                                      (f"{lvl}_ci_low", lo),
                                      (f"{lvl}_ci_high", hi)):
                        assert cond.loc[pb, col] == round_half_up(full, 3), (ctx, pb, col)
                        checked += 1
                for c in ("SB", "CI", "LD", "O"):
                    items = [p for p in PB if ae.get_construct(p) == c]
                    full = float(np.mean([g[p].sum() / n for p in items]))
                    assert cons.loc[c, lvl] == round_half_up(full, 3), (ctx, c, lvl)
                    checked += 1
            sens = pd.read_csv(tmp / f"sensitivity_construct_probs_{ctx}.csv")
            assert not any("ci" in c.lower() for c in sens.columns)
            for _, r in sens.iterrows():
                g = df[df[ctx] == r["level"]]
                items = [p for p in PB if ae.get_construct(p) == r["construct"]]
                full_mean = float(np.mean([g[p].mean() for p in items]))
                full_any = (g[items].sum(axis=1) > 0).mean()
                assert r["mean_of_items"] == round_half_up(full_mean, 3)
                assert r["mean_of_items"] == cons.loc[r["construct"], r["level"]]
                assert r["any_item"] == round_half_up(full_any, 3)
                checked += 2
            for pb in PB:
                full = _effect_full(df, ctx, pb)
                got = cond.loc[pb, "exploratory_effect_size"]
                assert (np.isnan(full) and np.isnan(got)) or got == round_half_up(full, 3), (ctx, pb)
                checked += 1

        part = df.groupby("PID")[LIKERTS + PB].mean()
        rho = pd.read_csv(tmp / "likert_pb_correlations.csv")
        for _, r in rho.iterrows():
            full = spearmanr(part[r["likert"]], part[r["PB"]])[0]
            assert r["spearman_rho"] == round_half_up(full, 3), (r["likert"], r["PB"])
            checked += 1

        # Figure annotations. Every heatmap draws full-precision data and
        # labels it with strings rounded once from that data.
        assert len(cap["heatmaps"]) == 5   # fig03b, fig05 x3, fig06
        for data, annot in cap["heatmaps"]:
            data = data.astype(float)
            for r in data.index:
                for c in data.columns:
                    label = str(annot.loc[r, c]).split("\n")[0]
                    assert label == format_half_up(data.loc[r, c], 2), (r, c, label)
                    checked += 1
        G, labels = cap["edges"]
        for (u, v), label in labels.items():
            full = (df[u] & df[v]).sum() / (df[u] | df[v]).sum()
            assert label == format_half_up(full, 2), (u, v)
            checked += 1

    # Reliability tables (printed by the pipeline).
    rev = df[PB[:-1]]
    alpha_tab, item_tab = ae.compute_reliability_wide_table(rev)
    for _, r in alpha_tab.iterrows():
        items = [c for c in rev if c.startswith(r["Construct"])]
        a, _ = pg.cronbach_alpha(rev[items])
        # Independent full-precision Feldt bounds (pingouin's own bounds are
        # already rounded to 3 decimals, which would hide double rounding).
        from scipy.stats import f as f_dist
        n, k = rev[items].shape
        ci95 = (1 - (1 - a) * f_dist.isf(0.025, n - 1, (n - 1) * (k - 1)),
                1 - (1 - a) * f_dist.isf(0.975, n - 1, (n - 1) * (k - 1)))
        assert r["Alpha"] == round_half_up(a, 2)
        assert r["Alpha_95CI"] == f"[{format_half_up(ci95[0], 2)}, {format_half_up(ci95[1], 2)}]"
        for col in items:
            d = rev[items]
            assert item_tab.loc["Item-total correlations", col] == round_half_up(
                d[col].corr(d.sum(axis=1) - d[col]), 2)
            assert item_tab.loc["Alpha if Deleted", col] == round_half_up(
                pg.cronbach_alpha(d.drop(columns=[col]))[0], 2)
            checked += 2

    # Descriptive summary (derived prevalence must use unrounded inputs).
    summ = ae.compute_descriptive_summary(df.drop(columns=PB), df[PB], PB, LIKERTS)
    for pid, g in df.groupby("PID"):
        for c in ("SB", "CI", "LD", "O"):
            items = [p for p in PB if ae.get_construct(p) == c]
            full = float(np.mean([g[p].mean() for p in items]))
            assert summ.loc[pid, f"{c}_prevalence"] == round_half_up(full, 3), (pid, c)
            checked += 1

    # Supplementary comparison. The printed value must be the single
    # rounding of an unrounded value.
    with tempfile.TemporaryDirectory() as t:
        comp = sup.main(output_dir=t)
    for _, r in comp.iterrows():
        assert r["code_rounded"] == format_half_up(r["code_unrounded"],
                                                   sup._decimals(r["paper"]))
        checked += 1
    a4 = sup.table_a4(sup.load_events())
    p16 = a4[(a4.P_label == "P16") & (a4.week == "W2")].iloc[0]
    assert p16["FUTURE_USE"] == 37 / 13      # stored unrounded

    print(f"{checked} reported values checked")


if __name__ == "__main__":
    test_traps_in_export_functions()
    test_pipeline_rounds_once()
    print("round-once tests passed")
