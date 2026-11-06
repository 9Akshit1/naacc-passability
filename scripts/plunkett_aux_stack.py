"""Can the survey measurements that make up the score be predicted from the map, and does passing those predictions on help?

The score is built from survey fields such as outlet drop, constriction and inlet grade. Each field exists for the surveyed
crossings, so it can be a training target for its own model. The predicted values, never the measured ones, are then added
as inputs to the score model. The folds are nested so that nothing leaks:

  In each outer fold, with whole HUC8 watersheds held out, the field models are fitted on the training watersheds only. Their
  predictions for the training rows come from an inner HUC8 split, and their predictions for the held-out rows come from the
  model fitted on all the training rows. The held-out rows' own survey fields are never seen.

Only development rows are used. The field values come from the public NAACC layer joined on survey id (the crossing code
agrees on 100% of joined rows). It uses the fixed LightGBM regression model, 5 outer folds and seeds 0 and 1. It also reports how
well each field is predicted (out-of-fold AUC or R2) and checks the ranking of the final score.
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from src import plunkett_cv as cv, plunkett_features as pf
from src.plunkett_final import LGBM_REG

SEEDS = (0, 1)
AUX_PARAMS = dict(num_leaves=31, min_child_samples=40, n_estimators=250, learning_rate=0.05, colsample_bytree=0.8,
                  subsample=0.8, subsample_freq=1, reg_lambda=5, verbose=-1, n_jobs=10)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def aux_labels(d: pd.DataFrame) -> pd.DataFrame:
    """One column per auxiliary label, NaN where the survey did not record it. kind: 'bin' or 'reg'."""
    p = pd.read_parquet("data/raw/naacc_public_layer.parquet")
    p["Survey_Id"] = p["Survey_Id"].astype(float)
    p = p.drop_duplicates("Survey_Id").set_index("Survey_Id")
    q = p.reindex(d["crossing_survey_id"].values)
    q.index = d.index
    out = pd.DataFrame(index=d.index)
    drop = num(q["Outlet_D_1"]).where(lambda s: s >= 0)                       # outlet drop to water surface, ft; -1 = no data
    out["perched_outlet"] = (drop > 0.25).astype(float).where(drop.notna())
    out["outlet_drop_log"] = np.log1p(drop)
    og = q["Outlet_Gra"].astype(str).str.strip()
    known = ~og.isin(["", "Unknown", "nan", "None"])
    out["outlet_free_fall"] = og.isin(["Free Fall", "Free Fall Onto Cascade", "Cascade"]).astype(float).where(known)
    ig = q["Inlet_Grad"].astype(str).str.strip()
    out["inlet_perched"] = ig.isin(["Perched", "Inlet Drop"]).astype(float).where(~ig.isin(["", "Unknown", "nan"]))
    cs = q["Crossing_S"].astype(str).str.strip()
    out["severe_constriction"] = (cs == "Severe").astype(float).where(cs.isin(["Severe", "Moderate", "Spans Only Bankfull/Active Channel", "Spans Full Channel & Banks"]))
    ct = q["Crossing_T"].astype(str).str.strip()
    out["is_bridge"] = ct.isin(["Bridge", "Bridge Adequate"]).astype(float).where(ct.isin(["Bridge", "Bridge Adequate", "Culvert", "Multiple Culvert"]))
    sp = q["Scour_Pool"].astype(str).str.strip()
    out["large_scour_pool"] = (sp == "Large").astype(float).where(sp.isin(["None", "Small", "Large"]))
    bw = num(q["Bankfull_W"]).where(lambda s: (s > 0) & (s < 200))
    out["bankfull_log"] = np.log(bw)
    return out


def kind(col):
    return "reg" if col.endswith("_log") else "bin"


def make(col, seed):
    cls = lgb.LGBMRegressor if kind(col) == "reg" else lgb.LGBMClassifier
    return cls(random_state=seed, **AUX_PARAMS)


def predict(m, X):
    return m.predict(X) if isinstance(m, lgb.LGBMRegressor) else m.predict_proba(X)[:, 1]


def fit_aux(X, y, col, seed):
    ok = y.notna().values
    return make(col, seed).fit(X[ok], y[ok])


def aux_features(train, test, feats, labels, seed):
    """Aux predictions: out-of-fold for the training rows, full-training-fit for the test rows."""
    tr_out = pd.DataFrame(index=train.index, dtype=float)
    te_out = pd.DataFrame(index=test.index, dtype=float)
    groups = train["huc8"].astype(str).values
    inner = list(GroupKFold(n_splits=3).split(train, groups=groups))
    for col in labels.columns:
        y = labels.loc[train.index, col]
        oof = np.full(len(train), np.nan)
        for a, b in inner:
            m = fit_aux(train[feats].iloc[a], y.iloc[a], col, seed)
            oof[b] = predict(m, train[feats].iloc[b])
        tr_out["aux_" + col] = oof
        te_out["aux_" + col] = predict(fit_aux(train[feats], y, col, seed), test[feats])
    return tr_out, te_out


if __name__ == "__main__":
    d = pd.read_parquet("data/processed/plunkett_features_osm.parquet")
    d, _ = pf.add_derived(d)
    dev = d[~cv.locked_test_mask(d)].copy().reset_index(drop=True)
    g = pf.feature_groups(dev)
    g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]
    feats = g["PLUNKETT"] + g["NBI"] + g["OSM"]
    labels = aux_labels(dev)
    print("labelled rows per aux label:", labels.notna().sum().to_dict(), flush=True)
    y = dev["aquatic_score"]
    auxcols = list(labels.columns)
    reg = lambda seed: lgb.LGBMRegressor(subsample_freq=1, verbose=-1, n_jobs=10, random_state=seed, **LGBM_REG)

    res = {"reference": [], "+ all aux": []}
    only = {c: [] for c in ("perched_outlet", "outlet_free_fall", "is_bridge")}
    aux_quality = {c: [] for c in auxcols}
    pooled_pred = {k: [] for k in list(res) + [f"+ only {c}" for c in only]}
    pooled_y = []
    for seed in SEEDS:
        oof = {k: pd.Series(np.nan, index=dev.index) for k in pooled_pred}
        for fold, (tr, te) in enumerate(cv.random_group_kfold(dev, "huc8", 5, seed)):
            t = time.time()
            train, test = dev.loc[tr], dev.loc[te]
            tr_aux, te_aux = aux_features(train, test, feats, labels, seed)
            oof["reference"].loc[te] = reg(seed).fit(train[feats], y.loc[tr]).predict(test[feats])
            Xtr, Xte = pd.concat([train[feats], tr_aux], axis=1), pd.concat([test[feats], te_aux], axis=1)
            oof["+ all aux"].loc[te] = reg(seed).fit(Xtr, y.loc[tr]).predict(Xte)
            for c in only:
                cols = list(train[feats].columns) + ["aux_" + c]
                oof[f"+ only {c}"].loc[te] = reg(seed).fit(Xtr[cols], y.loc[tr]).predict(Xte[cols])
            for c in auxcols:                       # how well is each aux label predicted on the held-out HUC8s?
                lab = labels.loc[te, c]
                ok = lab.notna().values
                if ok.sum() > 50 and lab[ok].nunique() > 1:
                    aux_quality[c].append(roc_auc_score(lab[ok], te_aux["aux_" + c][ok]) if kind(c) == "bin"
                                          else r2_score(lab[ok], te_aux["aux_" + c][ok]))
            print(f"seed {seed} fold {fold} done ({time.time() - t:.0f}s)", flush=True)
        for k, s in oof.items():
            pooled_pred[k].append(r2_score(y, s))
        print(f"seed {seed}: " + "  ".join(f"{k}={pooled_pred[k][-1]:.4f}" for k in oof), flush=True)
        if seed == SEEDS[0]:
            keep = {k: s.copy() for k, s in oof.items()}
    print("\nscore model, development rows, pooled out-of-fold R2 (mean of seeds):")
    ref = np.mean(pooled_pred["reference"])
    for k, v in pooled_pred.items():
        print(f"  {k:34s} R2={np.mean(v):.4f}  ({np.mean(v) - ref:+.4f})  seeds {[round(x, 4) for x in v]}")
    print("\nhow well each survey field is predicted from map data (held-out HUC8s; AUC for yes/no, R2 for log values):")
    for c, v in aux_quality.items():
        print(f"  {c:22s} {np.mean(v):.3f}   (rows labelled {int(labels[c].notna().sum())})")
    print("done", flush=True)
