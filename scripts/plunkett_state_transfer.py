"""Does a multi-state model do worse than single-state models? This is a question from Plunkett's earlier experience.

For each state S with at least 1,000 development crossings, the score is measured on S's crossings only, for three models:
  pooled      trained on all states, with S's watersheds held out (HUC8 group CV over everything)
  state-only  trained on S alone, with S's watersheds held out (HUC8 group CV within S)
  unseen      trained on every other state, never seeing S (leave-one-state-out)
The features and the fixed LightGBM settings are the same in all three. Only the development set and 2 seeds are used.
"""
import json
import numpy as np, pandas as pd, lightgbm as lgb
from src import plunkett_cv as cv, plunkett_features as pf

d = pd.read_parquet("data/processed/plunkett_features_nbi.parquet")
dev = d[~cv.locked_test_mask(d)].copy()
g = pf.feature_groups(d)
rows = []
for feats_name, feats in [("Plunkett", g["PLUNKETT"]), ("Plunkett+NBI", g["PLUNKETT"] + g["NBI"])]:
    mk = lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=10)
    pooled = cv.cross_val_oof(dev, feats, mk, "huc8", seeds=(0, 1))
    unseen = cv.cross_val_oof(dev, feats, mk, "state")
    for S in dev.state.value_counts()[lambda s: s >= 1000].index:
        sub = dev[dev.state == S]
        r = {"features": feats_name, "state": S, "n": len(sub)}
        r["pooled"] = float(np.mean([cv.metrics(sub.aquatic_score, run["oof"][sub.index])["r2"] for run in pooled]))
        r["unseen"] = cv.metrics(sub.aquatic_score, unseen[0]["oof"][sub.index])["r2"]
        so = cv.cross_val_oof(sub, feats, mk, "huc8", seeds=(0, 1), n_splits=min(5, sub.huc8.nunique()))
        r["state_only"] = float(np.mean([x["metrics"]["r2"] for x in so]))
        rows.append(r); print(r, flush=True)
df = pd.DataFrame(rows)
df.to_csv("results/plunkett/state_transfer.csv", index=False)
print(df.round(3).to_string())
print(df.groupby("features")[["pooled", "state_only", "unseen"]].mean().round(3))
