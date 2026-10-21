"""Development-set comparison of feature tiers x models x CV schemes (locked test NOT read).

Tiers: T1 Plunkett predictors | T2 + NBI | T3 + NBI + OSM
Models: lgbm_default | lgbm_tuned (best of scripts/plunkett_tune.py, chosen on seed 100) |
        lgbm_xent_tuned | catboost | ensemble (mean of the tuned members)
Schemes: random | huc8 | huc6 | state       Seeds 0,1,2 (state is deterministic)
Writes results/plunkett/dev_comparison.json and OOF predictions (huc8, seed 0) per model."""
import json, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostRegressor
from src import plunkett_cv as cv, plunkett_features as pf

import os
_osm = "data/processed/plunkett_features_osm.parquet"
d = pd.read_parquet(_osm if os.path.exists(_osm) else "data/processed/plunkett_features_nbi.parquet")
dev = d[~cv.locked_test_mask(d)].copy()
g = pf.feature_groups(d)
g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]   # never fires (0 matches); dropped a priori
tiers = {"T1_plunkett": g["PLUNKETT"], "T2_plunkett+nbi": g["PLUNKETT"] + g["NBI"]}
if g["OSM"]:
    tiers["T3_plunkett+nbi+osm"] = g["PLUNKETT"] + g["NBI"] + g["OSM"]

tuned = sorted((json.loads(l) for l in open("results/plunkett/tune.jsonl")), key=lambda r: -r["r2_mean"])
best_reg = next(r["params"] for r in tuned if r["params"]["objective"] != "cross_entropy")
best_xent = next(r["params"] for r in tuned if r["params"]["objective"] == "cross_entropy")
json.dump({"best_regression": best_reg, "best_cross_entropy": best_xent}, open("results/plunkett/chosen_params.json", "w"), indent=1)

def L(p):  return lambda: lgb.LGBMRegressor(subsample_freq=1, verbose=-1, n_jobs=10, random_state=0, **p)
def default():
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=10, random_state=0)
def cat():
    return CatBoostRegressor(iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=5, verbose=0, thread_count=10, random_seed=0)

class Mean:
    """equal-weight average of fitted members"""
    def __init__(self, makers): self.makers = makers
    def fit(self, X, y): self.ms = [m().fit(X, y) for m in self.makers]; return self
    def predict(self, X): return np.mean([m.predict(X) for m in self.ms], axis=0)

models = {"lgbm_default": default, "lgbm_tuned": L(best_reg), "lgbm_xent_tuned": L(best_xent), "catboost": cat,
          "ensemble": lambda: Mean([L(best_reg), L(best_xent), cat])}
only = sys.argv[1:]   # optional: restrict to some tiers
import os as _o
res = json.load(open("results/plunkett/dev_comparison.json")) if _o.path.exists("results/plunkett/dev_comparison.json") else {}
for tn, feats in tiers.items():
    if only and tn not in only: continue
    for mn, mk in models.items():
        if mn in ("lgbm_xent_tuned", "catboost") and tn != "T2_plunkett+nbi" and tn != "T3_plunkett+nbi+osm": continue   # members only needed for the ensemble
        for scheme in ["huc8", "huc6", "state", "random"]:
            if mn == "ensemble" and scheme == "random": continue
            t = time.time()
            runs = cv.cross_val_oof(dev, feats, mk, scheme, seeds=(0, 1, 2) if scheme != "huc6" else (0, 1))
            s = cv.summarize(runs); res[f"{tn}|{mn}|{scheme}"] = s
            if scheme == "huc8": runs[0]["oof"].to_frame("oof").to_parquet(f"results/plunkett/oof_{tn}_{mn}.parquet")
            print(f"{tn:22s} {mn:16s} {scheme:6s} R2={s['r2_mean']:.4f}+-{s['r2_sd']:.4f} MAE={s['mae_mean']:.4f} rho={s['spearman_mean']:.3f} ({time.time()-t:.0f}s)", flush=True)
            json.dump(res, open("results/plunkett/dev_comparison.json", "w"), indent=1)
