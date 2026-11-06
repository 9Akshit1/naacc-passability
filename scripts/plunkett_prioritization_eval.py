"""How well does the model pick which unsurveyed crossings to send a crew to?

Choosing crossings is a ranking job and not a score-prediction job, so it is measured as one. It uses development rows only,
whole-HUC8 folds, 3 seeds, and T3 inputs (Plunkett's predictors, the bridge inventory and OpenStreetMap structures).

For two cut-offs that we chose ("poor" means a score below 0.5 and "bad" means below 0.8) it compares
  - the regression model's prediction, ranked from the lowest score up, and
  - a classifier trained directly on the yes/no label (LightGBM, otherwise the same settings).
The measures are AUC, average precision (compared with the share of poor crossings, which is the value for no skill), and the share
of all poor crossings found when a crew inspects the top 10% or 20% of the ranking (a random ranking finds 10% and 20%).
The cut-offs are not NAACC's published barrier categories.
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import plunkett_cv as cv, plunkett_features as pf
from src.plunkett_final import LGBM_REG

SEEDS = (0, 1, 2)
CUTS = {"poor (<0.5)": 0.5, "bad (<0.8)": 0.8}


def reg():
    return lgb.LGBMRegressor(subsample_freq=1, verbose=-1, n_jobs=10, random_state=0, **LGBM_REG)


def clf():
    p = {k: v for k, v in LGBM_REG.items() if k not in ("objective",)}
    return lgb.LGBMClassifier(subsample_freq=1, verbose=-1, n_jobs=10, random_state=0, **p)


def caught(y, risk, share):
    k = int(round(share * len(y)))
    top = np.argsort(-risk)[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def oof(dev, feats, make, target, seed):
    out = pd.Series(np.nan, index=dev.index)
    for tr, te in cv.random_group_kfold(dev, "huc8", 5, seed):
        m = make().fit(dev.loc[tr, feats], target.loc[tr])
        out.loc[te] = m.predict(dev.loc[te, feats]) if not hasattr(m, "predict_proba") else m.predict_proba(dev.loc[te, feats])[:, 1]
    return out


if __name__ == "__main__":
    d = pd.read_parquet("data/processed/plunkett_features_osm.parquet")
    d, _ = pf.add_derived(d)
    dev = d[~cv.locked_test_mask(d)].copy().reset_index(drop=True)
    g = pf.feature_groups(dev)
    g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]
    feats = g["PLUNKETT"] + g["NBI"] + g["OSM"]
    y = dev["aquatic_score"]
    rows = []
    for seed in SEEDS:
        t = time.time()
        pred = oof(dev, feats, reg, y, seed)
        for name, cut in CUTS.items():
            label = (y < cut).astype(int)
            direct = oof(dev, feats, clf, label, seed)
            for how, risk in (("regression, ranked", 1 - pred.values), ("classifier trained on the label", direct.values)):
                rows.append(dict(seed=seed, cut=name, model=how, share_positive=label.mean(),
                                 auc=roc_auc_score(label, risk), ap=average_precision_score(label, risk),
                                 caught_top10=caught(label.values, risk, 0.10), caught_top20=caught(label.values, risk, 0.20)))
        print(f"seed {seed} done ({time.time() - t:.0f}s)", flush=True)
    r = pd.DataFrame(rows).groupby(["cut", "model"]).mean(numeric_only=True).drop(columns="seed")
    print(r.round(3).to_string(), flush=True)
    r.round(4).to_csv("results/plunkett/prioritization_eval_2026-10-30.csv")
    print("done", flush=True)
