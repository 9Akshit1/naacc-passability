"""Load and clean the NAACC New York survey export into one modelling dataset.

Source: NYS DEC ArcGIS FeatureServer ``NYS_NAACCSurveys_Projected`` (public, no
login), downloaded to ``data/raw/ny_naacc_surveys.csv`` — 44,871 survey records.
Provenance in ``data/raw/ny_naacc_PROVENANCE.json``.

The filtering recipe follows the UMass Designing Sustainable Landscapes (DSL)
"Aquatic Barriers" documentation (Plunkett et al. 2022), adapted to the fields
available in this export. Every drop is counted and reported.

One row of the modelling dataset = one **unique surveyed crossing** (most recent
survey, structures collapsed), with:

* ``passability`` — the target: NAACC aquatic passability score in [0, 1]
* identity + location + HUC8 (for spatial cross-validation)
* the raw field-survey variables (kept for the Phase-2 score reconstruction; NOT
  used as GIS-prediction features)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RAW_CSV = Path("data/raw/ny_naacc_surveys.csv")
INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")

# ArcGIS / NAACC missing-value sentinels
_MISSING_TOKENS = {"", "no data", "nodata", "unknown", "n/a", "na", "none given", "-1", "-1.0"}
_NUMERIC_MISSING = -1.0

# Field-survey columns retained for score reconstruction (not GIS features).
SURVEY_FIELDS = [
    "Crossing_Span", "Inlet_Grade", "Outlet_Grade", "Scour_Pool", "Armoring",
    "Internal_Structure", "Structure_Substrate_Matches_Stream", "Substrate_Continuous",
    "Substrate_Type", "Water_Depth_Matches_Stream", "Water_Velocity", "Barrier_Severity",
    "Barrier_Name", "Outlet_Drop_To_Water_Surface", "Outlet_Drop_To_Stream_Bottom",
    "Inlet_Openness", "Outlet_Openness", "Inlet_Height", "Outlet_Height",
    "Crossing_Structure_Length", "Slope_Percent", "Dry_Passage", "Material",
    "Flow_Condition", "Road_Fill_Height",
]

IDENTITY_FIELDS = [
    "Survey_Id", "Crossing_Code", "Date_Observed", "County", "Municipality", "Road",
    "Road_Type", "Stream_Name", "Crossing_Type", "Number_Of_Culverts",
    "NHD_HUC8_Watershed", "AOP", "Evaluation", "Terrestrial_Passage_Score",
]


@dataclass
class CleaningReport:
    steps: list[tuple[str, int, int]] = field(default_factory=list)  # (label, n_before, n_after)
    notes: list[str] = field(default_factory=list)

    def drop(self, label: str, before: int, after: int) -> None:
        self.steps.append((label, before, after))

    def to_markdown(self) -> str:
        lines = ["| Step | Records before | Records after | Dropped |", "|---|---|---|---|"]
        for label, b, a in self.steps:
            lines.append(f"| {label} | {b:,} | {a:,} | {b - a:,} |")
        if self.notes:
            lines.append("")
            lines += [f"- {n}" for n in self.notes]
        return "\n".join(lines)


def _norm(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def _num(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    return v.where(v != _NUMERIC_MISSING)


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_modeling_dataset(path: Path = RAW_CSV) -> tuple[pd.DataFrame, CleaningReport]:
    rep = CleaningReport()
    df = load_raw(path)
    n0 = len(df)
    rep.drop("raw survey records", n0, n0)

    # coordinates from the ArcGIS geometry (WGS84), falling back to GPS columns
    df["lon"] = _num(df.get("_geom_x", pd.Series(index=df.index)))
    df["lat"] = _num(df.get("_geom_y", pd.Series(index=df.index)))
    gx = _num(df.get("GPS_X_Coordinate", pd.Series(index=df.index)))
    gy = _num(df.get("GPS_Y_Coordinate", pd.Series(index=df.index)))
    df["lon"] = df["lon"].fillna(gx)
    df["lat"] = df["lat"].fillna(gy)

    b = len(df)
    df = df[df["lat"].between(40.0, 46.0) & df["lon"].between(-80.5, -71.0)]
    rep.drop("valid NY coordinates", b, len(df))

    # target
    df["passability"] = pd.to_numeric(df["Aqua_Pass_Score"], errors="coerce")
    b = len(df)
    df = df[df["passability"].notna()]
    rep.drop("has a numeric Aqua_Pass_Score", b, len(df))

    b = len(df)
    df = df[df["passability"].between(0.0, 1.0)]
    rep.drop("score in [0, 1] (drops the -1 'no score - missing data' sentinel)", b, len(df))

    # non-tidal only
    b = len(df)
    df = df[_norm(df["Tidal_Site"]).str.lower() != "yes"]
    rep.drop("non-tidal (Tidal_Site != Yes)", b, len(df))

    # must be an assessable crossing structure
    bad_types = {"inaccessible", "partially inaccessible", "no crossing", "no upstream channel",
                 "buried stream", "removed crossing"}
    b = len(df)
    df = df[~_norm(df["Crossing_Type"]).str.lower().isin(bad_types)]
    rep.drop("accessible crossing with a structure", b, len(df))

    # approved records only (NAACC QA flag)
    b = len(df)
    df = df[_norm(df["Approved"]).isin(["1", "1.0"])]
    rep.drop("NAACC-approved records", b, len(df))

    # collapse multi-structure rows: same crossing code + same survey date ->
    # keep the row with the LOWEST passability (the limiting structure), matching
    # how a crossing-level score is the worst structure.
    b = len(df)
    df["_date"] = pd.to_datetime(df["Date_Observed"], errors="coerce")
    df = (df.sort_values("passability")
            .groupby(["Crossing_Code", "_date"], as_index=False, dropna=False)
            .first())
    rep.drop("collapse multi-structure rows (keep limiting structure)", b, len(df))
    rep.notes.append("Multi-structure rows are collapsed to the lowest-passability structure per "
                     "(crossing code, survey date).")

    # repeat surveys of the same crossing: keep the most recent
    b = len(df)
    df = (df.sort_values("_date")
            .groupby("Crossing_Code", as_index=False)
            .last())
    rep.drop("keep most recent survey per crossing", b, len(df))

    # assemble the tidy frame
    keep = ["Crossing_Code", "lat", "lon", "passability", "_date"] + IDENTITY_FIELDS + SURVEY_FIELDS
    keep = [c for c in dict.fromkeys(keep) if c in df.columns]
    out = df[keep].copy()
    out = out.rename(columns={"NHD_HUC8_Watershed": "huc8_name", "_date": "date_observed"})
    out["crossing_id"] = out["Crossing_Code"]
    out["year"] = out["date_observed"].dt.year

    # numeric survey fields -> real NaN for the -1 sentinel
    for c in ["Outlet_Drop_To_Water_Surface", "Outlet_Drop_To_Stream_Bottom", "Inlet_Openness",
              "Outlet_Openness", "Inlet_Height", "Outlet_Height", "Crossing_Structure_Length",
              "Slope_Percent", "Road_Fill_Height"]:
        if c in out.columns:
            out[c] = _num(out[c])

    rep.notes.append(f"Final modelling dataset: {len(out):,} unique surveyed crossings, "
                     f"{out['huc8_name'].nunique()} HUC8 watersheds, "
                     f"survey years {int(out['year'].min())}-{int(out['year'].max())}.")
    rep.notes.append(f"Target (passability) mean {out['passability'].mean():.3f}, "
                     f"median {out['passability'].median():.3f}, "
                     f"fraction at 0.0: {(out['passability'] == 0).mean():.1%}, "
                     f"fraction at 1.0: {(out['passability'] == 1).mean():.1%}.")
    return out.reset_index(drop=True), rep


def save_modeling_dataset(df: pd.DataFrame, rep: CleaningReport) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "modeling_dataset.parquet", index=False)
    df.to_csv(PROCESSED / "modeling_dataset.csv", index=False)
    (INTERIM / "cleaning_report.md").write_text(rep.to_markdown(), encoding="utf-8")


if __name__ == "__main__":
    d, r = build_modeling_dataset()
    save_modeling_dataset(d, r)
    print(r.to_markdown())
