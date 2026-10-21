"""Is d8accum a proxy for 'bridge?'  (Plunkett's alternative explanation.)
Dev set only, HUC8-group CV. `bridge` is a survey field used here ONLY as an analysis label."""
import numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from src import plunkett_cv as cv, plunkett_data as pdm
d = pdm.add_engineered(pdm.load()); dev = d[~cv.locked_test_mask(d)].copy()
P = pdm.predictor_columns()
dev["bridge_i"] = dev["bridge"].astype(int)
print("bridge share in dev:", dev.bridge_i.mean().round(3))
print("mean score | bridge:", dev[dev.bridge].aquatic_score.mean().round(3), " | non-bridge:", dev[~dev.bridge].aquatic_score.mean().round(3))
mk = lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=6)
for name, feats in [("d8accum only", ["d8accum"]), ("all predictors", P)]:
    oof = np.full(len(dev), np.nan); idx = {i: k for k, i in enumerate(dev.index)}
    for tr, te in cv.splits(dev, "huc8", seed=0):
        m = mk().fit(dev.loc[tr, feats], dev.loc[tr, "bridge_i"])
        oof[[idx[i] for i in te]] = m.predict_proba(dev.loc[te, feats])[:, 1]
    print(f"P(bridge) from {name:15s}: AUC = {roc_auc_score(dev.bridge_i, oof):.3f}")
# within-class: does d8accum still predict the score?
mkr = lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=30, verbose=-1, n_jobs=6)
for lab, sub in [("bridges only", dev[dev.bridge]), ("non-bridges only", dev[~dev.bridge])]:
    for name, feats in [("d8accum only", ["d8accum"]), ("all predictors", P)]:
        s = cv.summarize(cv.cross_val_oof(sub, feats, mkr, "huc8", seeds=(0, 1)))
        print(f"{lab:17s} {name:15s} within-class R2 = {s['r2_mean']:.3f}  (n={len(sub)}, score sd={sub.aquatic_score.std():.3f})")
