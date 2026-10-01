"""Phase 1 dataset audit for the NAACC NY passability data.

Produces ``reports/01_dataset_audit.md`` and figures in ``results/figures/``.
Answers the Phase-1 questions from the project plan with real numbers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import IDENTITY_FIELDS, SURVEY_FIELDS, build_modeling_dataset, load_raw

FIG = Path("results/figures")
REP = Path("reports")


def _fig_path(name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    return FIG / name


def run_audit() -> str:
    raw = load_raw()
    model, clean_rep = build_modeling_dataset()

    L: list[str] = []
    L.append("# NAACC New York passability data - Phase 1 audit\n")
    L.append("Source: NYS DEC ArcGIS FeatureServer `NYS_NAACCSurveys_Projected` "
             "(public, no login). See `data/raw/ny_naacc_PROVENANCE.json`.\n")

    # ---- raw dataset questions -------------------------------------------
    raw["score"] = pd.to_numeric(raw["Aqua_Pass_Score"], errors="coerce")
    n_scored = raw["score"].between(0, 1).sum()
    L.append("## Raw export\n")
    L.append(f"| Question | Answer |\n|---|---|")
    L.append(f"| Total survey records | {len(raw):,} |")
    L.append(f"| Unique crossing codes | {raw['Crossing_Code'].nunique():,} |")
    L.append(f"| Unique survey ids | {raw['Survey_Id'].nunique():,} |")
    L.append(f"| Records with a valid 0-1 passability score | {n_scored:,} |")
    L.append(f"| Records with score = -1 (no score - missing data) | {(raw['score'] == -1).sum():,} |")
    L.append(f"| Tidal sites | {(raw['Tidal_Site'].str.strip().str.lower() == 'yes').sum():,} |")
    L.append(f"| Records with coordinates | {raw['GPS_X_Coordinate'].notna().sum():,} |")
    L.append(f"| Distinct HUC8 watersheds | {raw['NHD_HUC8_Watershed'].nunique()} |")
    L.append(f"| Distinct counties | {raw['County'].nunique()} |")
    codes = raw["Crossing_Code"].value_counts()
    L.append(f"| Crossing codes with >1 record | {(codes > 1).sum():,} (max {codes.max()} rows) |")
    L.append("")
    L.append("Crossing codes appear more than once for two reasons: a crossing with multiple "
             "structures (each culvert is its own row) and repeat surveys of the same crossing "
             "over time. Both are resolved in cleaning.\n")

    # ---- cleaning ----------------------------------------------------
    L.append("## Cleaning to the modelling dataset\n")
    L.append(clean_rep.to_markdown())
    L.append("")

    # ---- modelling dataset --------------------------------------------
    m = model
    L.append("## Modelling dataset\n")
    L.append(f"- **{len(m):,} unique surveyed crossings**")
    L.append(f"- {m['huc8_name'].nunique()} HUC8 watersheds "
             f"(min {m.groupby('huc8_name').size().min()}, "
             f"max {m.groupby('huc8_name').size().max()} crossings per watershed)")
    L.append(f"- {m['County'].nunique()} counties")
    L.append(f"- survey years {int(m['year'].min())}-{int(m['year'].max())} "
             f"(median {int(m['year'].median())})")
    L.append(f"- passability: mean {m['passability'].mean():.3f}, median "
             f"{m['passability'].median():.3f}, sd {m['passability'].std():.3f}")
    L.append(f"- at exactly 0.0: {(m['passability'] == 0).sum():,} ({(m['passability']==0).mean():.1%}); "
             f"at exactly 1.0: {(m['passability'] == 1).sum():,} ({(m['passability']==1).mean():.1%})")
    L.append("")

    # crossing type / road type breakdown
    L.append("### Crossing type\n")
    L.append(m["Crossing_Type"].value_counts().to_frame("n").to_markdown())
    L.append("\n### Road type\n")
    L.append(m["Road_Type"].value_counts().to_frame("n").to_markdown())
    L.append("")

    # AOP class vs score
    L.append("### AOP class vs numeric score\n")
    L.append(m.groupby("AOP")["passability"].agg(["count", "mean", "min", "max"]).round(3).to_markdown())
    L.append("")

    # missingness of survey fields (used for score reconstruction)
    L.append("### Missingness of field-survey variables (in the modelling dataset)\n")
    miss = (m[SURVEY_FIELDS].isna().mean().sort_values(ascending=False) * 100).round(1)
    L.append(miss.to_frame("% missing").to_markdown())
    L.append("")

    # repeat-survey analysis on the raw data
    rep_surv = (raw.groupby("Crossing_Code")["Survey_Id"].nunique())
    L.append("### Repeat-survey analysis (raw)\n")
    L.append(f"- crossings surveyed once: {(rep_surv == 1).sum():,}")
    L.append(f"- crossings surveyed 2+ times: {(rep_surv >= 2).sum():,}")
    L.append(f"- max surveys of one crossing: {rep_surv.max()}")
    L.append("")

    # ---- figures ----------------------------------------------------
    _figures(raw, m)
    L.append("## Figures\n")
    for fn, cap in [
        ("score_histogram.png", "Distribution of the NAACC aquatic passability score (modelling set)"),
        ("score_by_huc8.png", "Mean passability and crossing count by HUC8 watershed"),
        ("crossings_map.png", "Surveyed crossings, coloured by passability"),
        ("surveys_by_year.png", "Surveys by year"),
        ("missingness.png", "Missingness of field-survey variables"),
    ]:
        L.append(f"![{cap}](../results/figures/{fn})\n")

    REP.mkdir(parents=True, exist_ok=True)
    (REP / "01_dataset_audit.md").write_text("\n".join(L), encoding="utf-8")
    return "\n".join(L)


def _figures(raw: pd.DataFrame, m: pd.DataFrame) -> None:
    # histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(m["passability"], bins=40, color="#3a7ca5", edgecolor="white")
    ax.set_xlabel("NAACC aquatic passability score"); ax.set_ylabel("crossings")
    ax.set_title(f"n = {len(m):,} unique surveyed crossings (NY)")
    fig.tight_layout(); fig.savefig(_fig_path("score_histogram.png"), dpi=110); plt.close(fig)

    # by huc8
    g = m.groupby("huc8_name")["passability"].agg(["count", "mean"]).sort_values("count", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(g.index[::-1], g["count"][::-1], color="#c0c0c0")
    ax2 = ax.twiny()
    ax2.plot(g["mean"][::-1], g.index[::-1], "o-", color="#c0392b")
    ax.set_xlabel("crossing count (bars)"); ax2.set_xlabel("mean passability (line)", color="#c0392b")
    ax.set_title("Top 30 HUC8 watersheds by crossing count")
    fig.tight_layout(); fig.savefig(_fig_path("score_by_huc8.png"), dpi=110); plt.close(fig)

    # map
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(m["lon"], m["lat"], c=m["passability"], s=4, cmap="RdYlGn", alpha=0.6)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title("NY NAACC surveyed crossings")
    fig.colorbar(sc, label="passability"); ax.set_aspect(1.3)
    fig.tight_layout(); fig.savefig(_fig_path("crossings_map.png"), dpi=110); plt.close(fig)

    # by year
    yr = m["year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(yr.index, yr.values, color="#3a7ca5")
    ax.set_xlabel("survey year"); ax.set_ylabel("crossings"); ax.set_title("Surveys by year (most recent per crossing)")
    fig.tight_layout(); fig.savefig(_fig_path("surveys_by_year.png"), dpi=110); plt.close(fig)

    # missingness
    miss = (m[SURVEY_FIELDS].isna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(miss.index, miss.values, color="#8a6d3b")
    ax.set_xlabel("% missing"); ax.set_title("Field-survey variable missingness (modelling set)")
    fig.tight_layout(); fig.savefig(_fig_path("missingness.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    run_audit()
    print("wrote reports/01_dataset_audit.md")
