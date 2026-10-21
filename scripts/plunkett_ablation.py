"""Feature-group ablations on Plunkett's own predictors (development set only, HUC8-group CV).

Answers Plunkett's question: does catchment area (d8accum) matter, and is it a proxy for
'bridge?' - plus which other predictor groups carry the signal. LightGBM, fixed settings
(no tuning on this script), 3 differently-seeded HUC8 group assignments.
"""
import json, sys, time
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src import plunkett_cv as cv, plunkett_data as pdm

d = pdm.add_engineered(pdm.load())
dev = d[~cv.locked_test_mask(d)].copy()
P = pdm.predictor_columns()
groups = {
    "catchment": ["d8accum"],
    "gradient_dem": ["gradient", "dem"],
    "terrain_windows": [c for c in P if c.startswith(("elev_sd", "elev_range", "incise1"))],
    "traffic_road": ["adt", "class"],
    "development": [c for c in P if c.startswith("bdev")],
    "income": ["income"],
}
assert sorted(sum(groups.values(), [])) == sorted(P), set(P) ^ set(sum(groups.values(), []))

def lg():
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=8)

def rf():  # Plunkett-style: plain random forest
    return RandomForestRegressor(n_estimators=200, max_features=0.33, min_samples_leaf=3, n_jobs=8, random_state=0)

out = open("results/plunkett/ablation.jsonl", "a")
def run(name, feats, mk=lg, scheme="huc8", seeds=(0, 1, 2)):
    t = time.time()
    s = cv.summarize(cv.cross_val_oof(dev, feats, mk, scheme, seeds=seeds))
    s.update(name=name, scheme=scheme, n_features=len(feats))
    out.write(json.dumps(s) + "\n"); out.flush()
    print(f"{name:34s} {scheme:6s} R2={s['r2_mean']:.4f}±{s['r2_sd']:.4f} MAE={s['mae_mean']:.4f} rho={s['spearman_mean']:.3f} ({time.time()-t:.0f}s)", flush=True)

run("lgbm: all Plunkett predictors", P)
for g, cols in groups.items():
    run(f"lgbm: drop {g}", [c for c in P if c not in cols])
for g, cols in groups.items():
    run(f"lgbm: ONLY {g}", cols)
run("lgbm: only d8accum, log (same as raw for trees)", ["log_d8accum"])
run("rf (Plunkett-style): all predictors", P, mk=rf, seeds=(0, 1))
