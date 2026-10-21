"""Does the mutual-nearest rule matter? Same model, NBI joined two ways (development set only).

In deployment the candidate crossings are every crossing on the network, not just surveyed ones, so the mutual
test (which depends on which other crossings exist) behaves differently there. The one-way rule does not.

    python -m scripts.plunkett_nbi_oneway
"""
import json

import lightgbm as lgb
import pandas as pd

from src import nbi, plunkett_cv as cv, plunkett_features as pf

d = pd.read_parquet("data/processed/plunkett_features_nbi.parquet")
lockmask = cv.locked_test_mask(d)
groups = pf.feature_groups(d)
P, N = groups["PLUNKETT"], groups["NBI"]
structures = nbi.load_nbi(sorted(d.state.unique()))


def make():
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=10)


out = {}
for name, mutual_only in [("mutual nearest (used for all reported numbers)", True), ("one-way nearest", False)]:
    m = nbi.match(d, structures, 60.0, mutual_only=mutual_only)
    x = d.drop(columns=[c for c in d.columns if c.startswith("nbi_")]).join(m.drop(columns=["_nearest_dist_m", "_mutual"]))
    dev = x[~lockmask]
    s = cv.summarize(cv.cross_val_oof(dev, P + N, make, "huc8", seeds=(0, 1, 2)))
    out[name] = {"matched_share": float(m["nbi_matched"].mean()), **s}
    print(f"{name:46s} matched {m['nbi_matched'].mean():.4f}  R2 {s['r2_mean']:.4f} (sd {s['r2_sd']:.4f})", flush=True)
json.dump(out, open("results/plunkett/nbi_oneway.json", "w"), indent=1)
