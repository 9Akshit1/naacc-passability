"""Would a STATE-AWARE model (state as a categorical input) do better in states that have training surveys?
Labelling practice differs by state (e.g. 'Bridge Adequate' = fixed score 1.0 is used for 0%-83% of bridges), which GIS
cannot see. This is a separate in-state tier, NOT a replacement for the state-agnostic model (Plunkett's stated
concern). Development set, HUC8 group CV (5 random splits): T3 features +/- state."""
import numpy as np, pandas as pd, lightgbm as lgb
from src import plunkett_cv as cv, plunkett_features as pf
d = pd.read_parquet("data/processed/plunkett_features_osm.parquet"); dev = d[~cv.locked_test_mask(d)].copy()
dev["state_cat"] = dev.state.astype("category").cat.codes.astype(float)
g = pf.feature_groups(d); g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]; base = g["PLUNKETT"] + g["NBI"] + g["OSM"]
mk = lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=10)
mkc = lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=10)
S = (0, 1, 2, 3, 4)
ra = cv.cross_val_oof(dev, base, mk, "huc8", seeds=S); rb = cv.cross_val_oof(dev, base + ["state_cat"], mkc, "huc8", seeds=S, fit_kwargs={"categorical_feature": ["state_cat"]})
a = np.array([r["metrics"]["r2"] for r in ra]); b = np.array([r["metrics"]["r2"] for r in rb])
print(f"T3 state-agnostic R2 {a.mean():.4f}   T3 + state R2 {b.mean():.4f}   paired gain {np.mean(b-a):+.4f} (sd {np.std(b-a, ddof=1):.4f}, {int((b>a).sum())}/5)")
# per-state R2 on pooled OOF of the first split
for lab, rr in [("agnostic", ra[0]), ("state-aware", rb[0])]:
    o = rr["oof"]; print(lab, {s: round(cv.metrics(dev.loc[dev.state == s, "aquatic_score"], o[dev.state == s])["r2"], 3) for s in dev.state.value_counts().index[:8]})
