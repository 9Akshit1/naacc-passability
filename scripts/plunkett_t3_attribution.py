"""How much does catchment area (d8accum) drive the final-feature-set model, and what else matters?
Development set only, HUC8 group CV (3 random splits), LightGBM default on T3 features.
(1) TreeSHAP on held-out folds: mean |contribution| per feature group, as a share of the total.
(2) Retrain ablations: without d8accum; d8accum alone; structure evidence (NBI+OSM) alone; etc."""
import json
import numpy as np, pandas as pd, lightgbm as lgb
from src import plunkett_cv as cv, plunkett_features as pf
d = pd.read_parquet("data/processed/plunkett_features_osm.parquet"); dev = d[~cv.locked_test_mask(d)].copy()
g = pf.feature_groups(d); g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]
P, N, O = g["PLUNKETT"], g["NBI"], g["OSM"]
size = [c for c in N if c.startswith(("nbi_length", "nbi_max_span", "nbi_main_spans", "nbi_deck", "nbi_roadway", "nbi_vert", "nbi_lat_clr", "nbi_impr"))]
groups = {"catchment area (d8accum)": ["d8accum"], "gradient + elevation": ["gradient", "dem"],
          "terrain windows": [c for c in P if c.startswith(("elev_sd", "elev_range", "incise1"))], "traffic + road class": ["adt", "class"],
          "development (bdev)": [c for c in P if c.startswith("bdev")], "county income": ["income"],
          "NBI structure size": size, "NBI other fields": [c for c in N if c not in size], "OSM bridge/culvert tags": O}
feats = P + N + O; assert sorted(sum(groups.values(), [])) == sorted(feats)
mk = lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=6)
contrib = {k: [] for k in groups}; share_by_split = []
for seed in (0, 1, 2):
    tot = {k: 0.0 for k in groups}; n = 0
    for tr, te in cv.splits(dev, "huc8", seed=seed):
        m = mk().fit(dev.loc[tr, feats], dev.loc[tr, "aquatic_score"])
        sv = m.predict(dev.loc[te, feats], pred_contrib=True)[:, :-1]          # last column = expected value
        for k, cols in groups.items(): tot[k] += np.abs(sv[:, [feats.index(c) for c in cols]].sum(axis=1)).sum()
        n += len(te)
    s = sum(tot.values()); share_by_split.append({k: v / s for k, v in tot.items()})
shap = {k: float(np.mean([x[k] for x in share_by_split])) for k in groups}
print("TreeSHAP share of total |contribution| (group sums per row, then mean |.|):")
for k, v in sorted(shap.items(), key=lambda kv: -kv[1]): print(f"  {100*v:5.1f}%  {k}")
ab = {"all T3 features": feats, "without d8accum": [c for c in feats if c != "d8accum"],
      "without d8accum AND without NBI/OSM (Plunkett minus d8accum)": [c for c in P if c != "d8accum"],
      "d8accum only": ["d8accum"], "d8accum + NBI + OSM only": ["d8accum"] + N + O, "NBI + OSM only (no Plunkett predictors)": N + O,
      "all but terrain windows": [c for c in feats if c not in groups["terrain windows"]]}
res = {}
for name, f in ab.items():
    s = cv.summarize(cv.cross_val_oof(dev, f, mk, "huc8", seeds=(0, 1, 2))); res[name] = s["r2_mean"]; print(f"  R2={s['r2_mean']:.4f}  {name}", flush=True)
json.dump({"shap_share": shap, "ablation_r2": res}, open("results/plunkett/t3_attribution.json", "w"), indent=1)
