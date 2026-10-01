"""Reconstruction of the NAACC 2016 aquatic passability scoring algorithm.

Purpose (Phase 2 of the plan): understand exactly what the target variable
represents by recomputing it from the field-survey variables and checking it
against the ``Aqua_Pass_Score`` NAACC publishes.

Source of the formulas and weights:
https://streamcontinuity.org/resources/aquatic-connectivity-scoring-systems-non-tidal-crossings
("Aquatic Connectivity Scoring Systems for Non-tidal Crossings", rev. 2016-06-16)

This is a faithful re-implementation of the *documented* algorithm. Known gaps
(quantified in ``reports/02_score_reconstruction.md``):

* the NAACC application's exact rule for choosing the limiting structure / the
  inlet-vs-outlet openness input is not fully public;
* NAACC's internal defaults for "Unknown" / missing categoricals are only
  partly documented.

So this is validated as an *approximation*, not a bit-exact reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---- component weights (sum = 1.000) -----------------------------------
WEIGHTS = {
    "outlet_drop": 0.161,
    "physical_barriers": 0.135,
    "constriction": 0.090,
    "inlet_grade": 0.088,
    "water_depth": 0.082,
    "water_velocity": 0.080,
    "scour_pool": 0.071,
    "substrate_match": 0.070,
    "substrate_coverage": 0.057,
    "openness": 0.052,
    "height": 0.045,
    "outlet_armoring": 0.037,
    "internal_structures": 0.032,
}

# ---- categorical component score maps ----------------------------------
_CONSTRICTION = {
    "severe": 0.0, "moderate": 0.5,
    "spans only bankfull/active channel": 0.9, "spans only bankfull": 0.9,
    "spans full channel & banks": 1.0, "spans full channel and banks": 1.0,
    "unknown": 0.5, "no data": np.nan,
}
_INLET_GRADE = {
    "at stream grade": 1.0, "inlet drop": 0.0, "perched": 0.0,
    "clogged/collapsed/submerged": 1.0, "clogged": 1.0, "unknown": 1.0,
}
_INTERNAL = {"none": 1.0, "baffles/weirs": 0.0, "supports": 0.8, "other": 1.0, "no data": 1.0}
_ARMORING = {"extensive": 0.0, "not extensive": 0.5, "none": 1.0}
_BARRIER_SEV = {"none": 1.0, "minor": 0.8, "moderate": 0.5, "severe": 0.0}
_SCOUR = {"large": 0.0, "small": 0.8, "none": 1.0, "unknown": 1.0, "no data": np.nan}
_SUB_COV = {"none": 0.0, "25%": 0.3, "50%": 0.5, "75%": 0.7, "100%": 1.0, "unknown": np.nan}
_SUB_MATCH = {"none": 0.0, "not appropriate": 0.25, "contrasting": 0.75, "comparable": 1.0,
              "unknown": np.nan}
_WATER_DEPTH = {"yes": 1.0, "comparable": 1.0, "no-shallower": 0.0, "no-deeper": 0.5,
                "dry": 1.0, "unknown": np.nan}
_WATER_VEL = {"yes": 1.0, "comparable": 1.0, "no-faster": 0.0, "no-slower": 0.5,
              "dry": 1.0, "unknown": np.nan}


def _cat(series: pd.Series, mapping: dict[str, float], default: float = np.nan) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    return s.map(lambda v: mapping.get(v, default) if isinstance(v, str) else np.nan)


# ---- continuous component score functions -----------------------------


def openness_score(openness_ft: pd.Series) -> pd.Series:
    """So = (1 - e^(-k x (1-d)))^(1/(1-d)),  a=1, k=15, d=0.62 ; x = openness ratio (ft)."""
    x = pd.to_numeric(openness_ft, errors="coerce").clip(lower=0)
    k, d = 15.0, 0.62
    base = 1.0 - np.exp(-k * x * (1.0 - d))
    return np.power(np.clip(base, 0, None), 1.0 / (1.0 - d)).clip(0, 1)


def height_score(height_ft: pd.Series) -> pd.Series:
    """Sh = min(a x^2 / (b^2 + x^2), 1),  a=1.1, b=2.2 (ft)."""
    x = pd.to_numeric(height_ft, errors="coerce").clip(lower=0)
    a, b = 1.1, 2.2
    return np.minimum(a * x**2 / (b**2 + x**2), 1.0)


def outlet_drop_score(drop_ft: pd.Series) -> pd.Series:
    """Sod = 1 - a x^2 / (b^2 + x^2),  a=1.029412, b=0.51449575 (ft)."""
    x = pd.to_numeric(drop_ft, errors="coerce").clip(lower=0)
    a, b = 1.029412, 0.51449575
    return (1.0 - a * x**2 / (b**2 + x**2)).clip(0, 1)


@dataclass
class ScoreResult:
    composite: pd.Series
    final: pd.Series
    components: pd.DataFrame


def compute_passability(df: pd.DataFrame) -> ScoreResult:
    """Recompute the NAACC passability score from the NY dataset's field columns."""
    comp = pd.DataFrame(index=df.index)

    comp["openness"] = openness_score(df.get("Outlet_Openness", df.get("Inlet_Openness")))
    comp["height"] = height_score(df.get("Outlet_Height", df.get("Inlet_Height")))

    # Outlet drop: prefer drop-to-water-surface; fall back to drop-to-stream-bottom
    # only when the water-surface value is genuinely absent.
    drop = pd.to_numeric(df["Outlet_Drop_To_Water_Surface"], errors="coerce")
    if "Outlet_Drop_To_Stream_Bottom" in df.columns:
        db = pd.to_numeric(df["Outlet_Drop_To_Stream_Bottom"], errors="coerce")
        drop = drop.where(drop.notna(), db)
    comp["outlet_drop"] = outlet_drop_score(drop)

    comp["constriction"] = _cat(df["Crossing_Span"], _CONSTRICTION)
    comp["inlet_grade"] = _cat(df["Inlet_Grade"], _INLET_GRADE, default=1.0)
    comp["internal_structures"] = _cat(df["Internal_Structure"], _INTERNAL, default=1.0)
    comp["outlet_armoring"] = _cat(df["Armoring"], _ARMORING, default=1.0)
    comp["physical_barriers"] = _cat(df["Barrier_Severity"], _BARRIER_SEV, default=1.0)
    comp["scour_pool"] = _cat(df["Scour_Pool"], _SCOUR, default=1.0)
    comp["substrate_coverage"] = _cat(df["Substrate_Continuous"], _SUB_COV, default=np.nan)
    comp["substrate_match"] = _cat(df["Structure_Substrate_Matches_Stream"], _SUB_MATCH, default=np.nan)
    comp["water_depth"] = _cat(df["Water_Depth_Matches_Stream"], _WATER_DEPTH, default=np.nan)
    comp["water_velocity"] = _cat(df["Water_Velocity"], _WATER_VEL, default=np.nan)

    # Weighted composite. For components with no data, redistribute their weight
    # proportionally across the components that do (so a partly-filled survey is
    # not silently penalised).
    w = pd.Series(WEIGHTS)
    filled = comp.notna()
    eff_w = filled.mul(w, axis=1)
    eff_w = eff_w.div(eff_w.sum(axis=1), axis=0)
    composite = (comp.fillna(0.0) * eff_w).sum(axis=1)

    final = np.minimum(composite, comp["outlet_drop"].fillna(1.0)).clip(0, 1)
    return ScoreResult(composite=composite, final=final, components=comp)


def validate_against_published(df: pd.DataFrame) -> dict:
    """Compare the reconstruction to the NAACC-published ``passability`` column."""
    res = compute_passability(df)
    y = pd.to_numeric(df["passability"], errors="coerce")
    pred = res.final
    mask = y.notna() & pred.notna()
    y, pred = y[mask], pred[mask]
    err = pred - y
    return {
        "n": int(mask.sum()),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),
        "pearson_r": float(np.corrcoef(y, pred)[0, 1]),
        "spearman_rho": float(pd.Series(y).corr(pd.Series(pred), method="spearman")),
        "within_0.05": float((err.abs() <= 0.05).mean()),
        "within_0.10": float((err.abs() <= 0.10).mean()),
        "within_0.20": float((err.abs() <= 0.20).mean()),
    }
