"""Loader for the UMass DSL (Plunkett) multi-state passability dataset.

Source: ``passability_data.zip`` from Ethan Plunkett (UMass DSL), 2026-10-08 -
56,525 NAACC surveys in 13 states that fall on the DSL stream/road networks,
with the predictors his production pipeline uses. Raw files are in
``data/raw/plunkett/`` or wherever ``NAACC_PLUNKETT_DIR`` points (``passability_data.parquet`` +
``passability_predvars.txt``).

What this module adds on top of the raw file:

* ``state``   - 2-letter state from a point-in-polygon join with the Census state
  boundaries (every point falls inside a state polygon; verified).
* ``huc8`` / ``huc6`` / ``huc4`` - USGS WBD watershed ids (HUC8 polygon join), used
  ONLY as cross-validation groups, never as model inputs.
* No rows are dropped silently. ``audit_flags`` lists suspicious rows; callers
  decide, and every filter is reported.

Survey-derived columns (``bridge``, ``crossing_type``, ``terrestrial_passage_score``,
``tsurveyed``, ``roadtile``, ``moved``, ``date_observed``, ``database``, coordinates)
are NOT predictors: most are field observations that would not exist for an
unsurveyed crossing (``bridge``/``crossing_type`` in particular would leak the answer).
Only the columns in Plunkett's ``passability_predvars.txt`` are eligible.
"""
from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _plunkett_dir() -> Path:
    """Where passability_data.parquet and passability_predvars.txt live.

    Set NAACC_PLUNKETT_DIR to point anywhere; otherwise ``data/raw/plunkett`` inside this
    repository (the files are not committed, see README) with the original sibling
    ``../FTP/web/Oct`` folder as a fallback for the author's machine.
    """
    env = os.environ.get("NAACC_PLUNKETT_DIR")
    if env:
        return Path(env)
    local = ROOT / "data" / "raw" / "plunkett"
    legacy = ROOT.parent / "FTP" / "web" / "Oct"
    return legacy if (not local.exists() and legacy.exists()) else local


FTP_DIR = _plunkett_dir()
STATES_SHP = ROOT / "data" / "raw" / "boundaries" / "states" / "cb_2022_us_state_500k.shp"
HUC8_GEOJSON = ROOT / "data" / "raw" / "boundaries" / "huc8_study_area.geojson"
CACHE = ROOT / "data" / "processed" / "plunkett_tagged.parquet"

TARGET = "aquatic_score"


def predictor_columns() -> list[str]:
    """Exactly the predictors Plunkett's pipeline uses (his own file)."""
    return [ln.strip() for ln in (FTP_DIR / "passability_predvars.txt").read_text().splitlines() if ln.strip()]


# Columns that are field observations / bookkeeping, never model inputs.
SURVEY_ONLY_COLUMNS = [
    "crossing_survey_id", "crossing_code", "terrestrial_passage_score", "tsurveyed", "tsurvey_id",
    "crossing_type", "bridge", "date_observed", "database", "roadtile", "moved",
    "lat_nad83", "lon_nad83", "x_coord", "y_coord",
]


def _tag_geography(df: pd.DataFrame) -> pd.DataFrame:
    pts = gpd.GeoDataFrame(df[["crossing_code"]].copy(),
                           geometry=gpd.points_from_xy(df.lon_nad83, df.lat_nad83), crs=4326)
    st = gpd.read_file(STATES_SHP).to_crs(4326)[["STUSPS", "geometry"]]
    j = gpd.sjoin(pts, st, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    df["state"] = j["STUSPS"].reindex(df.index)

    hu = gpd.read_file(HUC8_GEOJSON).to_crs(4326)[["huc8", "geometry"]]
    j = gpd.sjoin(pts, hu, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    df["huc8"] = j["huc8"].reindex(df.index)
    # points on a polygon seam / just offshore: assign the nearest HUC8 (metric CRS)
    miss = df["huc8"].isna()
    if miss.any():
        pm = pts[miss].to_crs(5070)
        hm = hu.to_crs(5070)
        nn = gpd.sjoin_nearest(pm, hm, how="left")
        nn = nn[~nn.index.duplicated()]
        df.loc[miss, "huc8"] = nn["huc8"].reindex(pm.index).values
    df["huc6"] = df["huc8"].str[:6]
    df["huc4"] = df["huc8"].str[:4]
    return df


def audit_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean columns flagging rows a careful analyst would look at. Nothing is dropped here."""
    f = pd.DataFrame(index=df.index)
    f["removed_crossing"] = df["crossing_type"].eq("Removed Crossing")
    f["tsurveyed_2"] = df["tsurveyed"].eq(2)  # meaning undocumented in the file; reported only
    f["any_predictor_missing"] = df[predictor_columns()].isna().any(axis=1)
    f["d8accum_le_0"] = df["d8accum"] <= 0
    return f


def load(use_cache: bool = True) -> pd.DataFrame:
    if use_cache and CACHE.exists():
        return pd.read_parquet(CACHE)
    df = pd.read_parquet(FTP_DIR / "passability_data.parquet").reset_index(drop=True)
    df = _tag_geography(df)
    assert df["state"].notna().all(), "every crossing should fall in a state polygon"
    assert df["huc8"].notna().all()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    return df


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms of Plunkett's predictors only (no new information)."""
    out = df.copy()
    out["log_d8accum"] = np.log10(out["d8accum"].clip(lower=1))
    out["log_adt"] = np.log10(out["adt"].clip(lower=1))
    return out


if __name__ == "__main__":
    d = load(use_cache=False)
    print(d.shape, d.state.nunique(), "states;", d.huc8.nunique(), "HUC8;", d.huc6.nunique(), "HUC6;", d.huc4.nunique(), "HUC4")
    print(d.groupby("state").huc8.nunique().to_string())
    print(audit_flags(d).sum().to_string())
