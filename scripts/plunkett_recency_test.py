"""Does the age of a survey matter? The surveys in the file run from 2003 to 2026, and the protocol and the people surveying have changed.

It uses development rows only, whole-HUC8 folds, seeds 0 to 2 and the fixed LightGBM regression model.
  A. the reference
  B. survey year added as an input (not usable as it is for unsurveyed crossings, but it shows whether there is a drift to model)
  C. recency weights, so that newer surveys count more, with year not an input (usable for new crossings)
  D. training on 2015 and later only, tested on all held-out rows and also on held-out 2015 and later rows only
It also prints the R2 of the reference model for held-out rows by survey period, to see whether newer surveys are easier or harder.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src import plunkett_cv as cv, plunkett_features as pf
from src.plunkett_final import LGBM_REG

SEEDS = (0, 1, 2)


def reg(seed):
    return lgb.LGBMRegressor(subsample_freq=1, verbose=-1, n_jobs=6, random_state=seed, **LGBM_REG)


if __name__ == "__main__":
    d = pd.read_parquet("data/processed/plunkett_features_osm.parquet")
    d, _ = pf.add_derived(d)
    dev = d[~cv.locked_test_mask(d)].copy().reset_index(drop=True)
    g = pf.feature_groups(dev)
    g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]
    feats = g["PLUNKETT"] + g["NBI"] + g["OSM"]
    year = pd.to_datetime(dev["date_observed"], errors="coerce").dt.year
    print("survey year coverage:", year.notna().mean().round(3), year.describe()[["min", "50%", "max"]].to_dict())
    print(year.value_counts().sort_index().to_dict(), flush=True)
    dev["year"] = year.fillna(year.median())
    y = dev["aquatic_score"]
    recent = (dev["year"] >= 2015).values
    out = {k: [] for k in ("A reference", "B + year input", "C recency weights", "D train 2015+ only (all held-out)", "D train 2015+ only (held-out 2015+)", "A reference (held-out 2015+)")}
    by_period = {k: [] for k in ("<2010", "2010-14", "2015-19", "2020+")}
    for seed in SEEDS:
        pred = {k: pd.Series(np.nan, index=dev.index) for k in ("A", "B", "C", "D")}
        for tr, te in cv.random_group_kfold(dev, "huc8", 5, seed):
            Xtr, Xte = dev.loc[tr, feats], dev.loc[te, feats]
            pred["A"].loc[te] = reg(seed).fit(Xtr, y.loc[tr]).predict(Xte)
            pred["B"].loc[te] = reg(seed).fit(dev.loc[tr, feats + ["year"]], y.loc[tr]).predict(dev.loc[te, feats + ["year"]])
            w = np.exp((dev.loc[tr, "year"].values - 2026) / 8.0)
            pred["C"].loc[te] = reg(seed).fit(Xtr, y.loc[tr], sample_weight=w).predict(Xte)
            keep = tr[recent[tr]]
            pred["D"].loc[te] = reg(seed).fit(dev.loc[keep, feats], y.loc[keep]).predict(Xte)
        out["A reference"].append(r2_score(y, pred["A"]))
        out["B + year input"].append(r2_score(y, pred["B"]))
        out["C recency weights"].append(r2_score(y, pred["C"]))
        out["D train 2015+ only (all held-out)"].append(r2_score(y, pred["D"]))
        out["D train 2015+ only (held-out 2015+)"].append(r2_score(y[recent], pred["D"][recent]))
        out["A reference (held-out 2015+)"].append(r2_score(y[recent], pred["A"][recent]))
        for k, m in (("<2010", dev.year < 2010), ("2010-14", (dev.year >= 2010) & (dev.year < 2015)),
                     ("2015-19", (dev.year >= 2015) & (dev.year < 2020)), ("2020+", dev.year >= 2020)):
            by_period[k].append(r2_score(y[m], pred["A"][m]))
        print(f"seed {seed}: " + "  ".join(f"{k}={v[-1]:.4f}" for k, v in out.items()), flush=True)
    print()
    for k, v in out.items():
        print(f"{k:40s} R2 = {np.mean(v):.4f}  seeds {[round(x, 4) for x in v]}")
    print("\nreference model R2 by survey period (held-out rows):", {k: round(float(np.mean(v)), 3) for k, v in by_period.items()},
          {k: int(m) for k, m in (("<2010", (dev.year < 2010).sum()), ("2010-14", ((dev.year >= 2010) & (dev.year < 2015)).sum()),
                                  ("2015-19", ((dev.year >= 2015) & (dev.year < 2020)).sum()), ("2020+", (dev.year >= 2020).sum()))})
    print("done")
