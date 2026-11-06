"""Development-set tests of ideas that earlier rounds did not cover.

The baseline is the fixed T3 LightGBM regression model (Plunkett, bridge inventory and OpenStreetMap), HUC8 group CV and 3 seeds.
The test basins are not read. Every row used is a development row.

  A  robust loss functions (Huber, Fair, L1). The earlier search had tried only cross-entropy.
  B  neighbourhood context from the other surveyed crossings, using no scores: local crossing density, the crossing's own
     catchment area compared with its neighbours', and an elevation slope between nearby crossings on a stream of similar size
     (a rough stand-in for the change in stream gradient at a crossing)
  C  dropping features by gain importance, chosen inside each training fold
The neighbourhood features use feature values only and never scores, so held-out rows may serve as neighbours of each other.
Survey density depends on where programs surveyed, so B is a test and not a candidate for use until it is checked against an
inventory of unsurveyed crossings.
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src import plunkett_cv as cv, plunkett_features as pf
from src.plunkett_final import LGBM_REG

SEEDS = (0, 1, 2)


def lgbm(**over):
    p = dict(LGBM_REG)
    p.update(over)
    return lambda: lgb.LGBMRegressor(subsample_freq=1, verbose=-1, n_jobs=10, random_state=0, **p)


def neighbourhood(d: pd.DataFrame) -> pd.DataFrame:
    xy = np.c_[d["_x"], d["_y"]]
    tree = cKDTree(xy)
    log_area = np.log10(d["d8accum"].clip(lower=1e-6)).values
    grad, dem = d["gradient"].values, d["dem"].values
    large = ((d["nbi_matched"].fillna(0) > 0) | (d["osm_bridge_flag"].fillna(0) > 0)).astype(float).values
    n1 = np.array([len(i) - 1 for i in tree.query_ball_point(xy, 1000)])
    near5 = tree.query_ball_point(xy, 5000)
    out = np.full((len(d), 6), np.nan)
    out[:, 0], out[:, 1] = n1, [len(i) - 1 for i in near5]
    for i, idx in enumerate(near5):
        idx = np.array([j for j in idx if j != i])
        if len(idx) == 0:
            continue
        dist = np.hypot(*(xy[idx] - xy[i]).T)
        out[i, 2] = log_area[i] - np.nanmean(log_area[idx])
        out[i, 5] = large[idx].mean()
        close = dist <= 1500
        if close.any():
            out[i, 3] = grad[i] - np.nanmean(grad[idx[close]])
            similar = close & (np.abs(log_area[idx] - log_area[i]) <= 0.5) & (dist > 30)
            if similar.any():
                out[i, 4] = np.nanmedian(np.abs(dem[idx[similar]] - dem[i]) / dist[similar])
    return pd.DataFrame(out, index=d.index,
                        columns=["nb_n_1km", "nb_n_5km", "nb_dlogarea_5km", "nb_dgrad_1.5km", "nb_slope_same_stream", "nb_large_share_5km"])


class Pruned:
    """Fit, keep the top-k features by gain in THIS training fold, refit."""
    def __init__(self, make, k):
        self.make, self.k = make, k

    def fit(self, X, y):
        first = self.make().fit(X, y)
        self.keep_ = list(pd.Series(first.booster_.feature_importance("gain"), index=X.columns).nlargest(self.k).index)
        self.m_ = self.make().fit(X[self.keep_], y)
        return self

    def predict(self, X):
        return self.m_.predict(X[self.keep_])


def run(name, df, feats, make, scheme="huc8", seeds=SEEDS):
    t = time.time()
    s = cv.summarize(cv.cross_val_oof(df, feats, make, scheme, seeds=seeds))
    print(f"{name:52s} {scheme:6s} R2={s['r2_mean']:.4f}+-{s['r2_sd']:.4f}  MAE={s['mae_mean']:.4f}  ({time.time() - t:.0f}s)", flush=True)
    return s


if __name__ == "__main__":
    d = pd.read_parquet("data/processed/plunkett_features_osm.parquet")
    d, _ = pf.add_derived(d)
    dev = d[~cv.locked_test_mask(d)].copy().reset_index(drop=True)
    g = pf.feature_groups(dev)
    g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]
    base = g["PLUNKETT"] + g["NBI"] + g["OSM"]
    print(f"development rows: {len(dev)}, features: {len(base)}", flush=True)

    ref = run("reference: frozen LightGBM regression", dev, base, lgbm())
    for name, over in [("Huber alpha 0.5", dict(objective="huber", alpha=0.5)),
                       ("Huber alpha 0.2", dict(objective="huber", alpha=0.2)),
                       ("Fair loss", dict(objective="fair", fair_c=0.3)),
                       ("L1", dict(objective="l1"))]:
        run("A  " + name, dev, base, lgbm(**over))

    nb = neighbourhood(dev)
    dev = pd.concat([dev, nb], axis=1)
    print("neighbourhood features built:", {c: round(float(nb[c].notna().mean()), 3) for c in nb.columns}, flush=True)
    run("B  + neighbourhood context (all six)", dev, base + list(nb.columns), lgbm())
    run("B  + slope/area deltas only (no density)", dev, base + ["nb_dlogarea_5km", "nb_dgrad_1.5km", "nb_slope_same_stream"], lgbm())
    run("B  + density only", dev, base + ["nb_n_1km", "nb_n_5km", "nb_large_share_5km"], lgbm())

    for k in (15, 25):
        run(f"C  top-{k} features by in-fold gain", dev, base, lambda k=k: Pruned(lgbm(), k))
    print("done", flush=True)
