"""Random hyper-parameter search for LightGBM on the development set only.

Scored by HUC8-group CV with seed 100 (the reporting runs use seeds 0-2, so tuning and
reporting use different watershed-to-fold assignments). The locked test set is never read.
Feature set: Plunkett + NBI (+ OSM if --osm). Objective is part of the search space."""
import json, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from src import plunkett_cv as cv, plunkett_features as pf

use_osm = "--osm" in sys.argv
path = "data/processed/plunkett_features_osm.parquet" if use_osm else "data/processed/plunkett_features_nbi.parquet"
d = pd.read_parquet(path)
dev = d[~cv.locked_test_mask(d)].copy()
g = pf.feature_groups(d)
feats = g["PLUNKETT"] + g["NBI"] + (g["OSM"] if use_osm else [])
rng = np.random.default_rng(7)
space = dict(num_leaves=[7, 15, 31, 63, 127], min_child_samples=[20, 50, 100, 200, 400],
             n_estimators=[300, 600, 1000], learning_rate=[0.02, 0.03, 0.05],
             colsample_bytree=[0.4, 0.6, 0.8], subsample=[0.6, 0.8, 1.0],
             reg_lambda=[0, 2, 10, 30], objective=["regression", "cross_entropy", "huber"],
             extra_trees=[False, True], max_bin=[63, 255])
out = open(f"results/plunkett/tune{'_osm' if use_osm else ''}.jsonl", "a")
best = -9
for it in range(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 36):
    p = {k: v[rng.integers(len(v))] for k, v in space.items()}
    p = {k: (v.item() if hasattr(v, "item") else v) for k, v in p.items()}
    mk = lambda p=p: lgb.LGBMRegressor(subsample_freq=1, verbose=-1, n_jobs=10, random_state=0, **p)
    t = time.time()
    r = cv.cross_val_oof(dev, feats, mk, "huc8", seeds=(100,))
    s = cv.summarize(r); s["params"] = p
    out.write(json.dumps(s) + "\n"); out.flush()
    best = max(best, s["r2_mean"])
    print(f"{it:2d} R2={s['r2_mean']:.4f} (best {best:.4f}) MAE={s['mae_mean']:.4f} {p} ({time.time()-t:.0f}s)", flush=True)
