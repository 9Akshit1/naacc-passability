"""Re-run of the main development-set comparisons AFTER fixing the CV seed bug (src/plunkett_cv.random_group_kfold).
Tiers T1/T2/T3; LightGBM default and tuned (5 genuinely different splits), RF Plunkett-style (2 splits, T1 only),
final ensemble (3 splits, HUC8). Per-split R2 are saved so paired tier differences can be computed on identical folds."""
import json, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from src import plunkett_cv as cv, plunkett_features as pf
from src.plunkett_final import FinalEnsemble, LGBM_REG

d = pd.read_parquet("data/processed/plunkett_features_osm.parquet"); dev = d[~cv.locked_test_mask(d)].copy()
g = pf.feature_groups(d); g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]
tiers = {"T1": g["PLUNKETT"], "T2": g["PLUNKETT"] + g["NBI"], "T3": g["PLUNKETT"] + g["NBI"] + g["OSM"]}
default = lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8,
                                    subsample_freq=1, colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=8)
tuned = lambda: lgb.LGBMRegressor(subsample_freq=1, verbose=-1, n_jobs=8, random_state=0, **LGBM_REG)
rf = lambda: RandomForestRegressor(n_estimators=200, max_features=0.33, min_samples_leaf=3, n_jobs=8, random_state=0)
out = {}
def run(key, feats, mk, scheme, seeds):
    t = time.time(); runs = cv.cross_val_oof(dev, feats, mk, scheme, seeds=seeds)
    r2 = [r["metrics"]["r2"] for r in runs]; out[key] = {"r2_per_split": r2, "mean": float(np.mean(r2)), "sd": float(np.std(r2, ddof=1)) if len(r2) > 1 else None}
    json.dump(out, open("results/plunkett/dev_fixed_cv.json", "w"), indent=1)
    print(f"{key:34s} R2={np.mean(r2):.4f} sd={out[key]['sd']}  splits={len(r2)} ({time.time()-t:.0f}s)", flush=True)
S5 = (0, 1, 2, 3, 4)
for scheme in ["huc8", "huc6"]:
    for t, f in tiers.items():
        run(f"{t}|lgbm_default|{scheme}", f, default, scheme, S5)
        run(f"{t}|lgbm_tuned|{scheme}", f, tuned, scheme, S5)
run("T1|rf_plunkett_style|huc8", tiers["T1"], rf, "huc8", (0, 1))
for t, f in tiers.items():
    run(f"{t}|ensemble|huc8", f, FinalEnsemble, "huc8", (0, 1, 2))
