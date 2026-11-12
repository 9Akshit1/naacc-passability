"""Write one row per surveyed crossing with the NAACC score and two predictions of it, each made without using that crossing's basin.

    python -m scripts.build_prediction_table

Columns of results/plunkett/predictions_surveyed_crossings.csv:
  crossing_code, state, huc8    from Plunkett's file and the watershed join
  naacc_score                   the surveyed aquatic passability score
  pred_boosted_ensemble         the final model of this repository with Plunkett's predictors, the bridge inventory and
                                OpenStreetMap (the model called T3 in README.md)
  pred_random_forest            a random forest on Plunkett's predictors alone, with the settings used for the comparison in README.md
  basis                         how the two predictions were made:
                                  cross_validation  the 47,530 development crossings, predicted by 5-fold cross-validation with
                                                    whole HUC8 watersheds held out
                                  set_aside_basins  the 8,995 crossings in the 7 set-aside river basins, predicted by models
                                                    trained on the development crossings

The same predictions were used for the R-squared values in README.md, so the table can be checked against them: the R-squared of
each prediction column against naacc_score is printed by this script for each basis. The random forest on the set-aside basins
is refitted here with the same settings and seed as in scripts/plunkett_locked_eval.py, and the script stops if its R-squared
differs from the recorded one.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src import plunkett_cv as cv, plunkett_features as pf

OUT = "results/plunkett/predictions_surveyed_crossings.csv"


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def rf():
    return RandomForestRegressor(n_estimators=300, max_features=0.33, min_samples_leaf=3, n_jobs=10, random_state=0)


d = pd.read_parquet("data/processed/plunkett_features_osm.parquet")
lock = cv.locked_test_mask(d)
dev, test = d[~lock], d[lock]
plk = pf.feature_groups(d)["PLUNKETT"]

ours_dev = pd.read_parquet("results/plunkett/oof_T3_plunkett+nbi+osm_ensemble.parquet")["oof"].reindex(dev.index)
ours_test = pd.read_csv("results/plunkett/locked_predictions_T3.csv").set_index("crossing_code").pred.reindex(test.crossing_code).values

rf_dev = pd.Series(np.nan, index=dev.index)
for tr, te in cv.splits(dev, "huc8", 5, 0):
    rf_dev.loc[te] = rf().fit(dev.loc[tr, plk], dev.loc[tr, "aquatic_score"]).predict(dev.loc[te, plk])
rf_test = rf().fit(dev[plk], dev.aquatic_score).predict(test[plk])

recorded = json.load(open("results/plunkett/locked_results.json"))["T1_rf_plunkett_style"]["r2"]
if abs(r2(test.aquatic_score, rf_test) - recorded) > 1e-6:
    raise SystemExit(f"random forest on the set-aside basins gives {r2(test.aquatic_score, rf_test):.6f}, recorded {recorded:.6f}")

cols = ["crossing_code", "state", "huc8"]
a = dev[cols].assign(naacc_score=dev.aquatic_score, pred_boosted_ensemble=ours_dev.values, pred_random_forest=rf_dev.values, basis="cross_validation")
b = test[cols].assign(naacc_score=test.aquatic_score, pred_boosted_ensemble=ours_test, pred_random_forest=rf_test, basis="set_aside_basins")
t = pd.concat([a, b]).sort_index()
t[["pred_boosted_ensemble", "pred_random_forest", "naacc_score"]] = t[["pred_boosted_ensemble", "pred_random_forest", "naacc_score"]].round(4)
t.to_csv(OUT, index=False)
print(f"wrote {len(t)} rows to {OUT}; missing predictions: {int(t[['pred_boosted_ensemble', 'pred_random_forest']].isna().sum().sum())}")
for basis, g in t.groupby("basis"):
    print(f"{basis}: n={len(g)}  R2 boosted ensemble {r2(g.naacc_score, g.pred_boosted_ensemble):.4f}  random forest {r2(g.naacc_score, g.pred_random_forest):.4f}")
