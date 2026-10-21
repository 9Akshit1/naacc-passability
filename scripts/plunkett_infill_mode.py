"""Separate question: if a crossing sits among surveyed crossings, do neighbouring survey scores help?

This is NOT the headline protocol. The headline numbers hold out whole watersheds, which answers
"how well does the model work in a place it has no surveys". Here crossings are held out one fifth at a
time at random, and the model may use the scores of nearby surveyed crossings (training fold only), which
answers "how well can a gap be filled inside an already-surveyed area". Surveys are clustered by
programme and county, so random holdout is optimistic about real gaps; read the result as an upper bound.

Development set only. T2 features (his predictors + NBI), LightGBM default settings.

    python -m scripts.plunkett_infill_mode
"""
import json

import lightgbm as lgb
import pandas as pd

from src import plunkett_cv as cv, plunkett_features as pf

d = pd.read_parquet("data/processed/plunkett_features_nbi.parquet")
dev = d[~cv.locked_test_mask(d)].copy()
g = pf.feature_groups(d)
base = g["PLUNKETT"] + g["NBI"]


def make():
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=10)


def prep(tr, te):
    """Neighbour summaries from training-fold labels only; a row never counts itself."""
    nb_tr = pf.neighbour_features(tr, tr, exclude_same_group=None)
    ref = tr.copy()
    nb_te = pf.neighbour_features(ref, te, exclude_same_group=None)
    return tr.join(nb_tr), te.join(nb_te)


out = {}
for name, cols, hook in [("random split, no neighbour scores", base, None),
                         ("random split, with neighbour scores", base + pf.NEIGHBOUR_COLS, prep)]:
    runs = cv.cross_val_oof(dev, cols, make, "random", seeds=(0, 1, 2), prep=hook)
    s = cv.summarize(runs)
    out[name] = s
    print(f"{name:40s} R2 {s['r2_mean']:.4f} (sd {s['r2_sd']:.4f})  MAE {s['mae_mean']:.4f}", flush=True)
json.dump(out, open("results/plunkett/infill_mode.json", "w"), indent=1)
