"""If a crossing sits among surveyed crossings, do the scores of its neighbours help?

This is not the main test. The main numbers hold out whole watersheds, which shows how well the model works in a place with no
surveys. Here one fifth of the crossings are held out at random each time, and the model may use the scores of nearby surveyed
crossings from the training part only. That shows how well a gap could be filled inside an area that is already surveyed.
Surveys are clustered by program and county, so a random holdout is optimistic for real gaps and the result is an upper bound.

It uses the development set, T2 features (Plunkett's predictors and the bridge inventory) and LightGBM default settings.

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
