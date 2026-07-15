#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_participants.py
=======================

Participant-level profiling and clustering based on the enriched event-level
data produced by analyze_events.py.

Pipeline
--------
1.  Load enriched event-level data (events_enriched.csv).
        PIDs are mapped to human-readable labels (P1–P20) for all plots
        and tables while the raw Prolific IDs are preserved for traceability.
2.  Aggregate to participant level:
        - PB construct rates  (SB_rate, CI_rate, LD_rate, O_rate)
        - PB construct counts (SB_count, CI_count, LD_count, O_count)
        - Context proportions (TEMP, INTENT, TRIGGERS)
        - Mean Likert scores  (ALGO_CAUSE, FUTURE_USE, ARTIST_HARM)
3.  PCA dimensionality reduction on all features before clustering.
        With N=9 and ~19 features the raw feature space is wider than the
        sample.  PCA projects participants into a lower-dimensional space
        (n_components chosen to retain >= 80% variance) so that Ward
        distances are meaningful and stable.
4.  Determine optimal number of clusters (elbow + silhouette on PCA space).
5.  Ward hierarchical clustering on PCA components.
6.  Visualise:
        - PCA scree plot + biplot
        - Dendrogram
        - Participant heatmap (PB rates / raw counts / context / Likert)
        - Radar plots per cluster
7.  Kruskal-Wallis exploratory comparison across clusters.
        Within-cluster SDs are also reported to quantify internal
        heterogeneity, with automatic flagging of features where cluster 1
        exceeds mean cross-cluster variability.
8.  Export participant summary, cluster profiles, and PCA loadings to CSV.

Why rates AND counts?
---------------------
Rates capture the *relative* salience of each construct (proportion of
events where a flag was raised).  Counts capture *reporting intensity* —
a participant who flagged CI 20 times is behaviourally different from one
who flagged it 5 times even if both have the same CI rate.  Both dimensions
are included in clustering (after PCA) and visualised side-by-side.

Note on sample size
-------------------
N=9 participants.  All clustering and statistical results are exploratory
and hypothesis-generating only.  No inferential claims about cluster
stability or generalisability are warranted at this sample size.

Input
-----
outputs/tables/events_enriched.csv  — output of analyze_events.py step 7b
(see config.ENRICHED_EVENTS_PATH)

Output
------
outputs/tables/participant_summary.csv
outputs/tables/cluster_profiles.csv
outputs/tables/pca_loadings.csv
outputs/figures/participants/*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import (
    CLUSTER_PALETTE,
    CONSTRUCT_COLORS,
    ENRICHED_EVENTS_PATH,
    PARTICIPANTS_FIGURES_DIR,
    PID_MAP,
    TABLES_DIR,
    ensure_output_dirs,
    setup_plot_style,
)
from plotting import save_and_show

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

#: Minimum cumulative explained variance to retain from PCA
PCA_VARIANCE_THRESHOLD = 0.80

# NOTE: CONSTRUCT_COLORS, CLUSTER_PALETTE, and PID_MAP now live in
# config.py (imported above) so that both analysis scripts stay in sync;
# they were previously duplicated verbatim in each script.


def _short_ctx_label(col: str) -> str:
    """Short tick label for a context column.
    TEMP_POST -> 'post', INTENT_ACTIVE -> 'active',
    TRIGGERS_TR1_Presentation -> 'tr1'.
    """
    parts = col.split("_")
    return parts[1] if parts[0] == "TRIGGERS" else parts[-1]

def _short_feature_label(col: str) -> str:
    """Short arrow label for any clustering feature.

    TEMP_POST                  -> 'post'
    INTENT_ACTIVE              -> 'active'
    TRIGGERS_TR3_Continuation  -> 'tr3'
    mean_ALGO_CAUSE            -> 'ALGO_CAUSE'
    CI_rate / O_count          -> unchanged
    """
    parts = col.split("_")
    if parts[0] == "TEMP":
        return parts[-1]                 # pre / post
    if parts[0] == "INTENT":
        return parts[-1]                 # active / passive
    if parts[0] == "TRIGGERS":
        return parts[1]                  # tr1 / tr2 / tr3
    if col.startswith("mean_"):
        return col[len("mean_"):]                # strip the 'mean_' prefix
    return col                                   # rates & counts as-is

# ---------------------------------------------------------------------------
# Step 1 — Load & validate
# ---------------------------------------------------------------------------

def load_enriched(path: str | Path) -> pd.DataFrame:
    """
    Load the enriched event-level CSV produced by analyze_events.py.

    Each raw Prolific PID is mapped to a human-readable label via
    ``PID_MAP``.  The combined label (``{PID}_{Px}``) is stored in
    ``PID_labeled`` for exports, while ``P_label`` (e.g. ``"P8"``) is
    used in plots.  Unmapped PIDs receive ``"unknown"`` and a warning is
    printed.

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
# Step 3 — PCA
# ---------------------------------------------------------------------------

def run_pca(
    summary: pd.DataFrame,
    feature_cols: list[str],
    variance_threshold: float = PCA_VARIANCE_THRESHOLD,
) -> tuple[np.ndarray, PCA, pd.DataFrame]:
    """
    Standardise all features and apply PCA, retaining enough components to
    explain at least ``variance_threshold`` of total variance.

    With N=9 and ~19 features, PCA compresses the feature space before
    clustering so Ward distances are meaningful.  Count features are
    standardised along with all others so their larger absolute scale does
    not distort the principal components.

    Parameters
    ----------
    summary:
        Participant summary DataFrame.
    feature_cols:
        Column names to include as PCA input features.
    variance_threshold:
        Minimum cumulative explained variance to retain (default 0.80).

    Returns
    -------
    X_pca:
        PCA-transformed coordinates (n_participants x n_components).
    pca:
        Fitted PCA object.
    loadings_df:
        Feature loadings (features x components), rounded to 3 d.p.
    """
    X        = summary[feature_cols].values.astype(float)
    X_scaled = StandardScaler().fit_transform(X)

    pca_full     = PCA()
    pca_full.fit(X_scaled)
    cumvar       = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n_components = max(2, min(n_components, len(summary) - 1))

    print(f"\n=== PCA ===")
    print(f"  Features:     {len(feature_cols)}")
    print(f"  Participants: {len(summary)}")
    print(f"  Components retained (>= {variance_threshold*100:.0f}% variance): "
          f"{n_components}")
    print(f"  Cumulative variance at {n_components} components: "
          f"{cumvar[n_components-1]:.3f}")

    # Scree plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(
        range(1, len(pca_full.explained_variance_ratio_) + 1),
        pca_full.explained_variance_ratio_,
        color="#4C72B0", alpha=0.8,
    )
    ax1.axvline(n_components + 0.5, color="red", linestyle="--",
                label=f"Retained: {n_components} PCs")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("Scree Plot")
    ax1.legend(fontsize=16)
    ax1.grid(alpha=0.3)

    ax2.plot(range(1, len(cumvar) + 1), cumvar, "o-", color="#DD8452")
    ax2.axhline(variance_threshold, color="gray", linestyle="--", alpha=0.6,
                label=f"{variance_threshold*100:.0f}% threshold")
    ax2.axvline(n_components + 0.5, color="red", linestyle="--",
                label=f"n={n_components} retained")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("Cumulative Variance")
    ax2.legend(fontsize=16)
    ax2.grid(alpha=0.3)

    # plt.suptitle("PCA — Variance Explained", fontsize=13, y=1.02)
    plt.tight_layout()
    save_and_show(fig, PARTICIPANTS_FIGURES_DIR, "fig11_pca_scree.png")

    pca       = PCA(n_components=n_components)
    X_pca     = pca.fit_transform(X_scaled)
    pc_labels = [f"PC{i+1}" for i in range(n_components)]

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=pc_labels,
    ).round(3)

    print(f"\n=== PCA Loadings (top 5 contributors per PC) ===")
    for i, pc in enumerate(pc_labels):
        top     = loadings_df[pc].abs().nlargest(5)
        var_pct = pca.explained_variance_ratio_[i] * 100
        print(
            f"  {pc} ({var_pct:.1f}% var):  "
            + ", ".join(
                f"{f}={loadings_df.loc[f, pc]:+.2f}" for f in top.index
            )
        )

    return X_pca, pca, loadings_df


def plot_pca_biplot(
    X_pca: np.ndarray,
    pca: PCA,
    loadings_df: pd.DataFrame,
    summary: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    """
    PCA biplot: participant scores on PC1 vs PC2 (coloured by cluster) with
    feature loading arrows for features with notable contributions.

    Participant points are labelled with their short P-label (e.g. P8).

    Parameters
    ----------
    X_pca:
        PCA-transformed coordinates.
    pca:
        Fitted PCA object.
    loadings_df:
        Feature loadings DataFrame.
    summary:
        Participant summary with ``cluster`` and ``P_label`` columns.
    feature_cols:
        Original feature names.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, row in summary.reset_index(drop=True).iterrows():
        c_id  = int(row["cluster"])
        color = CLUSTER_PALETTE[c_id - 1]
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], color=color, s=150, zorder=3)
        ax.annotate(
            row["P_label"],
            (X_pca[idx, 0], X_pca[idx, 1]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=16, color=color, fontweight="bold",
        )

    # Feature loading arrows (only notable contributors)
    scale = 1.5 * max(np.abs(X_pca[:, :2]).max(), 1)
    for feat in feature_cols:
        lx = float(loadings_df.loc[feat, "PC1"]) * scale
        ly = float(loadings_df.loc[feat, "PC2"]) * scale
        if np.sqrt(lx**2 + ly**2) < 0.3 * scale:
            continue
        ax.annotate(
            "", xy=(lx, ly), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
        )
        ax.text(lx * 1.08, ly * 1.08, _short_feature_label(feat),
                fontsize=16, color="gray", ha="center", va="center")

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=16
    )
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=16
    )
    # ax.set_title("PCA Biplot — Participant Scores + Feature Loadings", fontsize=13)

    n_clusters = summary["cluster"].nunique()
    handles    = [
        mpatches.Patch(color=CLUSTER_PALETTE[i], label=f"Cluster {i+1}")
        for i in range(n_clusters)
    ]
    ax.legend(handles=handles, fontsize=16, loc="best")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    save_and_show(fig, PARTICIPANTS_FIGURES_DIR, "fig12_pca_biplot.png")


# ---------------------------------------------------------------------------
# Step 4 — Optimal cluster number
# ---------------------------------------------------------------------------

def select_n_clusters(
    X_pca: np.ndarray,
    Z: np.ndarray,
    k_range: range | None = None,
) -> int:
    """
    Plot elbow curve (WCSS) and silhouette scores in PCA space and return
    the k with the highest silhouette score.

    With N=9, valid range is k=2 to k=N-1.  Results are exploratory.

    Parameters
    ----------
    X_pca:
        PCA-transformed participant coordinates.
    Z:
        Ward linkage matrix computed on X_pca.
    k_range:
        Range of k to evaluate. Defaults to range(2, min(6, N)).

    Returns
    -------
    Suggested k (highest silhouette score).
    """
    n = len(X_pca)
    if k_range is None:
        k_range = range(2, min(6, n))

    wcss, silhouettes = [], []

    for k in k_range:
        labels = fcluster(Z, t=k, criterion="maxclust")
        wcss_k = sum(
            float(
                np.sum(
                    (X_pca[labels == c] - X_pca[labels == c].mean(axis=0)) ** 2
                )
            )
            for c in np.unique(labels)
        )
        wcss.append(wcss_k)
        sil = silhouette_score(X_pca, labels) if 1 < k < n else np.nan
        silhouettes.append(sil)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(k_range), wcss, marker="o", color="#4C72B0")
    ax1.set_title("Elbow Curve (WCSS in PCA space)", fontsize=16)
    ax1.set_xlabel("Number of clusters k")
    ax1.set_ylabel("Within-cluster SS")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(alpha=0.3)

    ax2.plot(list(k_range), silhouettes, marker="s", color="#DD8452")
    ax2.set_title("Silhouette Score (PCA space)", fontsize=16)
    ax2.set_xlabel("Number of clusters k")
    ax2.set_ylabel("Silhouette score")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_and_show(fig, PARTICIPANTS_FIGURES_DIR, "fig13_cluster_selection.png")

    print("\n=== Cluster selection metrics ===")
    print(f"{'k':>4}  {'WCSS':>10}  {'Silhouette':>12}")
    for k, w, s in zip(k_range, wcss, silhouettes):
        s_str = f"{s:.4f}" if not np.isnan(s) else "   n/a"
        print(f"{k:>4}  {w:>10.3f}  {s_str:>12}")

    best_k = int(list(k_range)[int(np.nanargmax(silhouettes))])
    print(f"\nSuggested k (highest silhouette): {best_k}")
    return best_k


# ---------------------------------------------------------------------------
# Step 5 — Clustering
# ---------------------------------------------------------------------------

def run_clustering(
    summary: pd.DataFrame,
    X_pca: np.ndarray,
    k: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Ward hierarchical clustering on PCA-reduced coordinates.

    Parameters
    ----------
    summary:
        Participant summary DataFrame.
    X_pca:
        PCA-transformed coordinates used as clustering input.
    k:
        Number of clusters.

    Returns
    -------
    summary:
        Input DataFrame with ``cluster`` column added.
    Z:
        Ward linkage matrix (for dendrogram).
    """
    Z                  = linkage(X_pca, method="ward")
    summary            = summary.copy()
    summary["cluster"] = fcluster(Z, t=k, criterion="maxclust")

    print(f"\n=== Cluster assignments (k={k}) ===")
    for c in sorted(summary["cluster"].unique()):
        members = summary.loc[summary["cluster"] == c, "P_label"].tolist()
        print(f"  Cluster {c}: {members}")

    return summary, Z


# ---------------------------------------------------------------------------
# Step 6a — Dendrogram
# ---------------------------------------------------------------------------

def plot_dendrogram(
    Z: np.ndarray,
    labels: list[str],
    k: int,
) -> None:
    """
    Annotated Ward dendrogram with a dashed cut line that produces k clusters.
    Leaf labels use the human-readable P-label (e.g. P8).

    Parameters
    ----------
    Z:
        Ward linkage matrix.
    labels:
        Participant P_label values (e.g. ["P8", "P9", ...]).
    k:
        Number of clusters — determines the cut height.
    """
    heights   = sorted(Z[:, 2], reverse=True)
    threshold = (heights[k - 2] + heights[k - 1]) / 2

    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=16,
        color_threshold=threshold,
        ax=ax,
    )
    ax.axhline(
        threshold, color="black", linestyle="--",
        linewidth=1.0, alpha=0.7, label=f"Cut for k={k}",
    )
    # ax.set_title(
    #     f"Hierarchical Clustering of Participants\n"
    #     f"(Ward linkage on PCA components, k={k}, N={len(labels)})",
    #     fontsize=13,
    # )
    ax.set_ylabel("Euclidean distance (PCA space)", fontsize=16)
    # ax.set_xlabel("Participant", fontsize=11)
    ax.legend(fontsize=16)
    plt.tight_layout()
    save_and_show(fig, PARTICIPANTS_FIGURES_DIR, "fig14_dendrogram.png")


# ---------------------------------------------------------------------------
# Step 6b — Participant heatmap
# ---------------------------------------------------------------------------

def plot_participant_heatmap(
    summary: pd.DataFrame,
    ctx_cols: list[str],
) -> None:
    """
    Four-panel heatmap: PB rates / raw counts / context proportions / Likert.

    Row labels use the human-readable P_label.  Participants are sorted by
    cluster with horizontal separator lines and cluster colour labels on the
    left margin.

    Parameters
    ----------
    summary:
        Participant summary with ``cluster`` and ``P_label`` columns.
    ctx_cols:
        Context proportion column names.
    """
    heatmap_data = (
        summary.set_index("P_label")
               .sort_values("cluster")
    )
    likert_mean_cols = [f"mean_{c}" for c in LIKERT_COLS]
    cluster_sizes    = heatmap_data.groupby("cluster", sort=True).size().cumsum()

    fig, axes = plt.subplots(1, 4, figsize=(26, 7), sharey=True)

    # Panel 1: PB rates
    sns.heatmap(
        heatmap_data[RATE_COLS].astype(float),
        cmap="Blues", annot=True, fmt=".2f",
        ax=axes[0], cbar=False, vmin=0, vmax=1,
    )
    axes[0].set_title("PB Rates", fontsize=16)
    axes[0].set_xticklabels(["SB", "CI", "LD", "O"], rotation=0)

    # Panel 2: PB raw counts
    sns.heatmap(
        heatmap_data[COUNT_COLS].astype(float),
        cmap="Purples", annot=True, fmt=".0f",
        ax=axes[1], cbar=False,
    )
    axes[1].set_title("PB Counts", fontsize=16)
    axes[1].set_xticklabels(["SB", "CI", "LD", "O"], rotation=0)

    # Panel 3: Context proportions
    sns.heatmap(
        heatmap_data[ctx_cols].astype(float),
        cmap="Greens", annot=True, fmt=".2f",
        ax=axes[2], cbar=False, vmin=0, vmax=1,
    )
    axes[2].set_title("Interaction Context", fontsize=16)
    axes[2].set_xticklabels(
        [_short_ctx_label(c) for c in ctx_cols], rotation=25
    )

    # Panel 4: Likert means
    sns.heatmap(
        heatmap_data[likert_mean_cols].astype(float),
        cmap="Reds", annot=True, fmt=".2f",
        ax=axes[3], cbar=False, vmin=1, vmax=5,
    )
    axes[3].set_title("Appraisal", fontsize=16)
    axes[3].set_xticklabels(LIKERT_COLS, rotation=15, ha="right")

    # Cluster separator lines and left-margin labels
    prev = 0
    for cluster_id, boundary in cluster_sizes.items():
        for ax in axes:
            if boundary < len(heatmap_data):
                ax.axhline(boundary, color="black", linewidth=2)
        mid = (prev + boundary) / 2
        axes[0].text(
            -0.5, mid, f"C{int(cluster_id)}",
            va="center", ha="right", fontsize=11, fontweight="bold",
            color=CLUSTER_PALETTE[int(cluster_id) - 1],
            transform=axes[0].get_yaxis_transform(),
        )
        prev = boundary

    axes[0].set_ylabel("", fontsize=11)
    axes[1].set_ylabel("", fontsize=11)
    axes[2].set_ylabel("", fontsize=11)
    axes[3].set_ylabel("", fontsize=11)

    # plt.suptitle("Participant Profiles by Cluster", fontsize=14, y=1.01)
    plt.tight_layout()
    save_and_show(fig, PARTICIPANTS_FIGURES_DIR, "fig15_participant_heatmap.png")


# ---------------------------------------------------------------------------
# Step 6c — Radar plots
# ---------------------------------------------------------------------------

def _radar_ax(
    ax: plt.Axes,
    values: np.ndarray,
    labels: list[str],
    title: str,
    ylim: tuple[float, float],
    color: str,
) -> None:
    """Draw a single radar chart on a polar axis."""
    n      = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    vals   = values.tolist() + [values[0]]
    angs   = angles + [angles[0]]

    ax.plot(angs, vals, "o-", linewidth=2, color=color)
    ax.fill(angs, vals, alpha=0.25, color=color)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=10, pad=10)
    ax.grid(True, alpha=0.4)


def plot_cluster_radars(
    cluster_profile: pd.DataFrame,
    ctx_cols: list[str],
) -> None:
    """
    Four-row radar chart grid per cluster:
    PB rates / normalised counts / context proportions / Likert means.

    Counts are normalised to [0, 1] relative to the maximum across clusters
    so they share the same axis as rates.

    Parameters
    ----------
    cluster_profile:
        Cluster-level aggregate DataFrame.
    ctx_cols:
        Context proportion column names.
    """
    likert_mean_cols = [f"mean_{c}" for c in LIKERT_COLS]
    n_clusters       = len(cluster_profile)

    fig, axes = plt.subplots(
        4, n_clusters,
        figsize=(5 * n_clusters, 18),
        subplot_kw={"polar": True},
    )
    if n_clusters == 1:
        axes = axes.reshape(4, 1)

    max_counts = cluster_profile[COUNT_COLS].max().replace(0, 1)

    for col_idx, (_, row) in enumerate(cluster_profile.iterrows()):
        cluster_id  = int(row["cluster"])
        color       = CLUSTER_PALETTE[cluster_id - 1]
        n_members   = int(row["n_participants"])
        norm_counts = row[COUNT_COLS].values.astype(float) / max_counts.values

        _radar_ax(
            axes[0, col_idx],
            row[RATE_COLS].values.astype(float),
            ["SB", "CI", "LD", "O"],
            f"Cluster {cluster_id} (n={n_members})\nPB Rates",
            (0.0, 1.0), color,
        )
        _radar_ax(
            axes[1, col_idx],
            norm_counts,
            ["SB", "CI", "LD", "O"],
            f"Cluster {cluster_id}\nPB Counts (normalised)",
            (0.0, 1.0), color,
        )
        _radar_ax(
            axes[2, col_idx],
            row[ctx_cols].values.astype(float),
            [c.split("_", 1)[-1] for c in ctx_cols],
            f"Cluster {cluster_id}\nContext",
            (0.0, 1.0), color,
        )
        _radar_ax(
            axes[3, col_idx],
            row[likert_mean_cols].values.astype(float),
            LIKERT_COLS,
            f"Cluster {cluster_id}\nLikert",
            (1.0, 5.0), color,
        )

    plt.suptitle("Cluster Profiles — Radar Charts", fontsize=14, y=1.01)
    plt.tight_layout()
    save_and_show(fig, PARTICIPANTS_FIGURES_DIR, "fig16_cluster_radars.png")


# ---------------------------------------------------------------------------
# Step 7 — Cluster profiles + Kruskal-Wallis + within-cluster SDs
# ---------------------------------------------------------------------------

def build_cluster_profiles(
    summary: pd.DataFrame,
    ctx_cols: list[str],
) -> pd.DataFrame:
    """
    Aggregate participant summary to cluster level.

    Also runs Kruskal-Wallis tests and reports within-cluster standard
    deviations to quantify internal heterogeneity.  Features where cluster 1
    exceeds the mean cross-cluster SD are flagged automatically.

    With N=9 all tests are strictly exploratory.

    Parameters
    ----------
    summary:
        Participant summary with ``cluster`` column.
    ctx_cols:
        Context proportion column names.

    Returns
    -------
    cluster_profile:
        One row per cluster with n_participants, mean features, and
        dominant PB construct.
    """
    likert_mean_cols = [f"mean_{c}" for c in LIKERT_COLS]
    agg_cols         = RATE_COLS + COUNT_COLS + likert_mean_cols + ctx_cols

    cluster_profile = (
        summary.groupby("cluster", as_index=False)
               .agg(
                   n_participants = ("PID",    "count"),
                   mean_events    = ("events", "mean"),
                   **{c: (c, "mean") for c in agg_cols},
               )
    )

    cluster_profile["dominant_PB"] = (
        cluster_profile[RATE_COLS].idxmax(axis=1).str.replace("_rate", "")
    )

    # --- Kruskal-Wallis ---
    print("\n=== Kruskal-Wallis across clusters (exploratory, N=9) ===")
    print(f"{'Feature':>30}  {'H':>8}  {'p_raw':>8}  note")
    for col in RATE_COLS + COUNT_COLS + likert_mean_cols:
        groups = [
            grp[col].dropna().values
            for _, grp in summary.groupby("cluster")
        ]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            try:
                stat, pval = kruskal(*groups)
                note = "(*)" if pval < 0.05 else ""
                print(f"{col:>30}  {stat:>8.3f}  {pval:>8.4f}  {note}")
            except ValueError:
                print(f"{col:>30}  {'n/a':>8}  {'n/a':>8}")

    print("\n=== Cluster Profiles ===")
    print(
        cluster_profile[
            ["cluster", "n_participants", "mean_events"]
            + RATE_COLS + COUNT_COLS + likert_mean_cols
        ].to_string(index=False)
    )

    # --- Within-cluster standard deviations ---
    # Quantifies internal heterogeneity per cluster.
    print("\n=== Within-cluster standard deviations ===")
    within_std = (
        summary.groupby("cluster")[RATE_COLS + COUNT_COLS + likert_mean_cols]
               .std()
               .round(3)
    )
    print(within_std.to_string())

    # Flag features where cluster 1 SD exceeds the mean SD across all clusters.
    # Cluster label is looked up explicitly (not by position) for robustness.
    if 1 in within_std.index:
        c1_sd                 = within_std.loc[1]
        mean_sd               = within_std.mean()
        heterogeneous_features = c1_sd[c1_sd > mean_sd].index.tolist()
        if heterogeneous_features:
            print(
                f"\n  Cluster 1 above-average heterogeneity in: "
                f"{', '.join(heterogeneous_features)}"
            )

    return cluster_profile


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_results(
    summary: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    loadings_df: pd.DataFrame,
    output_dir: str | Path = TABLES_DIR,
) -> None:
    """
    Export participant summary, cluster profiles, and PCA loadings to CSV.

    The participant summary retains the full ``PID_labeled`` column
    (e.g. ``5e68bd3b..._P8``) for traceability.

    Parameters
    ----------
    summary:
        Full participant summary with cluster assignments.
    cluster_profile:
        Cluster-level aggregate table.
    loadings_df:
        PCA feature loadings (features x components).
    output_dir:
        Directory for output files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Drop plot-only helper column; keep PID_labeled for traceability
    summary.drop(columns=["PID_short"], errors="ignore").to_csv(
        out / "participant_summary.csv", index=False
    )
    cluster_profile.to_csv(out / "cluster_profiles.csv",  index=False)
    loadings_df.to_csv(out / "pca_loadings.csv")

    print(f"\n[export] participant_summary -> {out / 'participant_summary.csv'}")
    print(f"[export] cluster_profiles    -> {out / 'cluster_profiles.csv'}")
    print(f"[export] pca_loadings        -> {out / 'pca_loadings.csv'}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(
    input_path: str | Path = ENRICHED_EVENTS_PATH,
    output_dir: str | Path = TABLES_DIR,
    k: int | None = None,
    variance_threshold: float = PCA_VARIANCE_THRESHOLD,
) -> None:
    """
    Run the full participant-level profiling and clustering pipeline.

    Steps
    -----
    1.  Load events_enriched.csv; map PIDs to P-labels.
    2.  Aggregate to participant level (rates + counts + context + Likert).
    3.  PCA on all features (retains >= variance_threshold variance).
    4.  Auto-select k via silhouette score (or use provided k).
    5.  Ward hierarchical clustering on PCA components.
    6.  Scree plot, PCA biplot, dendrogram, heatmap, radar plots.
    7.  Kruskal-Wallis + within-cluster SD analysis.
    8.  Export CSVs.

    Parameters
    ----------
    input_path:
        Path to events_enriched.csv. Defaults to the file produced by
        analyze_events.py (see config.ENRICHED_EVENTS_PATH). Run
        analyze_events.py first if this file does not exist yet.
    output_dir:
        Directory for all output CSVs.
    k:
        Number of clusters.  If None, auto-selected by silhouette score.
        Override with e.g. ``main(k=3)`` to skip auto-selection.
    variance_threshold:
        Minimum cumulative PCA variance to retain (default 0.80).
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

    # 3. PCA on all features: rates + counts + context + Likert
    likert_mean_cols = [f"mean_{c}" for c in LIKERT_COLS]
    cluster_features = RATE_COLS + COUNT_COLS + ctx_cols + likert_mean_cols

    X_pca, pca, loadings_df = run_pca(
        summary, cluster_features, variance_threshold
    )

    # 4. Select k on PCA-reduced space
    Z_for_k = linkage(X_pca, method="ward")
    if k is None:
        k = select_n_clusters(
            X_pca, Z_for_k,
            k_range=range(2, min(6, len(summary))),
        )
        print(f"\nUsing k={k} (auto-selected). Override with main(k=N).")
    else:
        print(f"\nUsing k={k} (user-specified).")

    # 5. Cluster on PCA components
    summary, Z = run_clustering(summary, X_pca, k)

    # 6. Visualisations — P_label used throughout for readability
    plot_pca_biplot(X_pca, pca, loadings_df, summary, cluster_features)
    plot_dendrogram(Z, summary["P_label"].tolist(), k)
    plot_participant_heatmap(summary, ctx_cols)

    # 7. Cluster profiles + Kruskal-Wallis + within-cluster SDs
    cluster_profile = build_cluster_profiles(summary, ctx_cols)
    # plot_cluster_radars(cluster_profile, ctx_cols)

    # 8. Export
    export_results(summary, cluster_profile, loadings_df, output_dir)


if __name__ == "__main__":
    main()