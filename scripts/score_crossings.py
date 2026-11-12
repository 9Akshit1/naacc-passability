"""Predict the NAACC aquatic passability score for crossings from the map inputs, with the final model from this repository.

    python -m scripts.score_crossings --input crossings.parquet --out predictions.csv

The input is a parquet or csv file with one row per crossing and these columns:
  crossing_code           any id; it is copied to the output
  state                   two-letter state code (CT, DE, MA, MD, ME, NH, NJ, NY, PA, RI, VA, VT, WV)
  lat_nad83, lon_nad83    the crossing location
  every column listed in passability_predvars.txt (the same predictors as in passability_data.parquet)

The bridge inventory columns and the OpenStreetMap columns are added from the downloaded files (see the setup steps in the
README). The bridge inventory match pairs each crossing with the nearest bridge only when each is the other's nearest
neighbour, so it should be run on the full list of crossings and not on a handful of rows.

The model is src/plunkett_final.FinalEnsemble, fitted on all 56,525 surveyed crossings with the same inputs as the model called
T3 in README.md. The fitted model is kept in results/models/final_t3_all_surveys.joblib and reused.

The output has the prediction (0 to 1) and the 90% interval used in this repository: the prediction plus or minus the 90th
percentile of the absolute error of that kind of prediction on crossings in basins the model was not trained on.

    python -m scripts.score_crossings --check

rebuilds the inputs of the surveyed crossings from the raw columns alone and confirms that they equal the stored feature table.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import nbi as nbi_mod, osm_structures as osm, plunkett_data as pdm, plunkett_features as pf
from src.plunkett_final import FinalEnsemble

FEATURES = "data/processed/plunkett_features_osm.parquet"
MODEL = Path("results/models/final_t3_all_surveys.joblib")
OOF = "results/plunkett/oof_T3_plunkett+nbi+osm_ensemble.parquet"


def feature_list(d):
    g = pf.feature_groups(d)
    g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]
    return g["PLUNKETT"] + g["NBI"] + g["OSM"]


def build_inputs(raw):
    """Add the bridge inventory and OpenStreetMap columns to a table of crossings that has the predictors."""
    need = ["crossing_code", "state", "lat_nad83", "lon_nad83"] + pdm.predictor_columns()
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise SystemExit("These columns are missing from the input: " + ", ".join(missing))
    d = pdm.add_engineered(raw.reset_index(drop=True))
    states = sorted(d.state.unique())
    m = nbi_mod.match(d, nbi_mod.load_nbi(states), radius_m=pf.NBI_RADIUS_M)
    d = pd.concat([d, m.drop(columns=["_nearest_dist_m", "_mutual"])], axis=1)
    return pd.concat([d, osm.match_crossings(d, states)], axis=1)


def fitted_model():
    stored = pd.read_parquet(FEATURES)
    feats = feature_list(stored)
    if MODEL.exists():
        return joblib.load(MODEL), feats
    print(f"fitting on {len(stored)} surveyed crossings (a few minutes)", flush=True)
    model = FinalEnsemble().fit(stored[feats], stored.aquatic_score)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL)
    return model, feats


def interval_half_width():
    stored = pd.read_parquet(FEATURES)
    oof = pd.read_parquet(OOF)["oof"].reindex(stored.index)
    err = (stored.aquatic_score - oof).abs().dropna()
    return float(np.quantile(err, 0.9))


def check():
    stored = pd.read_parquet(FEATURES)
    feats = feature_list(stored)
    raw_cols = ["crossing_code", "state", "lat_nad83", "lon_nad83"] + pdm.predictor_columns()
    rebuilt = build_inputs(stored[raw_cols])
    bad = []
    for c in feats:
        a, b = stored[c].to_numpy(float), rebuilt[c].to_numpy(float)
        same = np.isclose(a, b, equal_nan=True)
        if not same.all():
            bad.append((c, int((~same).sum())))
    print(f"{len(feats)} model inputs compared on {len(stored)} crossings; columns that differ: {bad if bad else 'none'}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        check()
    if not a.input:
        ap.error("--input is required")
    raw = pd.read_csv(a.input, low_memory=False) if a.input.lower().endswith(".csv") else pd.read_parquet(a.input)
    d = build_inputs(raw)
    model, feats = fitted_model()
    p = model.predict(d[feats])
    h = interval_half_width()
    out = pd.DataFrame({"crossing_code": d.crossing_code, "state": d.state, "predicted_score": p.round(4),
                        "low_90": np.clip(p - h, 0, 1).round(4), "high_90": np.clip(p + h, 0, 1).round(4)})
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out}; interval half-width {h:.3f}")
