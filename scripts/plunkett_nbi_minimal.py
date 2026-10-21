"""How much of the NBI gain comes from a small set of fields?

Development set only (locked basins untouched), HUC8-group CV, LightGBM with the
default settings used in plunkett_dev_comparison.py, 3 differently seeded splits.

    python -m scripts.plunkett_nbi_minimal
"""
import json

import lightgbm as lgb
import pandas as pd

from src import plunkett_cv as cv, plunkett_features as pf

d = pd.read_parquet("data/processed/plunkett_features_nbi.parquet")
dev = d[~cv.locked_test_mask(d)].copy()
g = pf.feature_groups(d)
P, N = g["PLUNKETT"], g["NBI"]

size = [c for c in N if c.startswith(("nbi_length", "nbi_max_span", "nbi_main_spans", "nbi_deck", "nbi_roadway",
                                      "nbi_vert", "nbi_lat_clr", "nbi_impr"))]
sets = {
    "predictors only": P,
    "+ matched flag only": P + ["nbi_matched"],
    "+ length and max span (3 columns)": P + ["nbi_matched", "nbi_length_m", "nbi_max_span_m"],
    "+ the 8 size fields and matched flag": P + ["nbi_matched"] + size,
    "+ every NBI field we joined": P + N,
}


def make():
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=10)


out = {}
for name, cols in sets.items():
    runs = cv.cross_val_oof(dev, cols, make, "huc8", seeds=(0, 1, 2))
    s = cv.summarize(runs)
    out[name] = {"n_columns": len(cols), **s}
    print(f"{name:42s} cols={len(cols):3d}  R2 {s['r2_mean']:.4f} (sd {s['r2_sd']:.4f})", flush=True)
json.dump(out, open("results/plunkett/nbi_minimal.json", "w"), indent=1)
