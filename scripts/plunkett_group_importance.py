"""Grouped permutation importance (development set, HUC8-group folds, LightGBM default, T2 features).
Drop in held-out R2 when one feature GROUP is shuffled within the test fold (10 repeats)."""
import json
import numpy as np, pandas as pd, lightgbm as lgb
from src import plunkett_cv as cv, plunkett_features as pf
d = pd.read_parquet("data/processed/plunkett_features_nbi.parquet"); dev = d[~cv.locked_test_mask(d)].copy()
g = pf.feature_groups(d); P = g["PLUNKETT"]
N = g["NBI"]
pick = lambda *pre: [c for c in N if c.startswith(pre)]
groups = {
 "catchment area (d8accum)": ["d8accum"],
 "gradient + elevation": ["gradient", "dem"],
 "terrain windows (30)": [c for c in P if c.startswith(("elev_sd", "elev_range", "incise1"))],
 "traffic + road class": ["adt", "class"],
 "development (bdev)": [c for c in P if c.startswith("bdev")],
 "county income": ["income"],
 "NBI: matched flag + structure kind/type/culvert": ["nbi_matched", "nbi_kind", "nbi_type", "nbi_is_culvert"],
 "NBI: size (length, span, widths, spans, clearances)": pick("nbi_length", "nbi_max_span", "nbi_main_spans", "nbi_deck", "nbi_roadway", "nbi_vert", "nbi_lat_clr", "nbi_impr"),
 "NBI: age (year built / reconstructed)": pick("nbi_year"),
 "NBI: condition / scour / waterway adequacy": pick("nbi_channel", "nbi_culvert_cond", "nbi_substructure", "nbi_waterway", "nbi_scour"),
 "NBI: skew": ["nbi_skew_deg"],
 "NBI: owner / function class / ADT / lanes": pick("nbi_owner", "nbi_func", "nbi_adt", "nbi_lanes"),
}
used = sum(groups.values(), []); assert sorted(used) == sorted(P + N), set(P + N) ^ set(used)
feats = P + N; rng = np.random.default_rng(0); drops = {k: [] for k in groups}
mk = lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=10)
for tr, te in cv.splits(dev, "huc8", seed=0):
    m = mk().fit(dev.loc[tr, feats], dev.loc[tr, "aquatic_score"]); Xte = dev.loc[te, feats]; y = dev.loc[te, "aquatic_score"]
    base = cv.metrics(y, m.predict(Xte))["r2"]
    for k, cols in groups.items():
        for _ in range(10):
            X2 = Xte.copy(); perm = rng.permutation(len(X2)); X2[cols] = X2[cols].values[perm]
            drops[k].append(base - cv.metrics(y, m.predict(X2))["r2"])
out = {k: {"mean_r2_drop": float(np.mean(v)), "sd": float(np.std(v)), "n_features": len(groups[k])} for k, v in drops.items()}
json.dump(out, open("results/plunkett/group_importance.json", "w"), indent=1)
for k, v in sorted(out.items(), key=lambda kv: -kv[1]["mean_r2_drop"]): print(f"{v['mean_r2_drop']:+.4f} +-{v['sd']:.4f}  ({v['n_features']:2d} feats)  {k}")
