"""One model for every region, a model for each region, or an average of both? It uses development rows only and never reads the test basins.

    python -m scripts.plunkett_regions_eval {A|B|C}

A  map only, leaving out one state at a time. The held-out state stands in for a place with no local training data.
B  the same, on the crossings with a surveyed outlet drop, with and without the yes/no answer to "is the outlet perched?"
   (q_perched is 1 when the outlet drop to the water surface is more than 0.25 ft). Crossings with no recorded drop are left out
   and never filled in.
C  places that do have local data: HUC8 grouped 5-fold CV over all 13 states, comparing one model for everything, a model for
   each state, and the average of the two.

For a held-out state s, with no rows of s used in training, the variants are:
  pooled     all the other states
  regional   the 4 states whose centres are nearest to s
  ensemble   the average of pooled and regional
  simweight  pooled, with each training row weighted by how much it looks like s on the map inputs (a domain classifier trained on
             the inputs only, so no labels of s are used)
It uses the fixed LightGBM model (scripts.plunkett_streamcat_eval.reg) and seed 0.
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from scripts.plunkett_streamcat_eval import reg
from src import plunkett_cv as cv, plunkett_features as pf, streamcat_feats as sf

NJ_ = 4
K_NEAR = 4
SEED = 0


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def load(full=False):
    d = pd.read_parquet("data/processed/plunkett_features_osm.parquet")
    d, _ = pf.add_derived(d)
    g = pf.feature_groups(d)
    g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]
    t3 = g["PLUNKETT"] + g["NBI"] + g["OSM"]
    raster = [c for c in g["PLUNKETT"] if c.startswith(("elev_", "incise1_", "bdev_")) or c in ("gradient", "dem", "d8accum")]
    d = pd.concat([d, sf.build(d).drop(columns="comid")], axis=1)
    pub = [c for c in d.columns if c.startswith(("sc_", "vaa_"))] + ["snap_m", "snap2_m", "order_line", "ftype_line"]
    nodsl = [c for c in g["PLUNKETT"] if c not in raster] + g["NBI"] + g["OSM"] + pub
    o = pd.read_parquet("data/processed/plunkett_outlet_drop.parquet")[["crossing_code", "drop_ws"]]
    d = d.merge(o, on="crossing_code", how="left")
    d["q_perched"] = np.where(d.drop_ws.isna(), np.nan, (d.drop_ws > 0.25).astype(float))
    if full:
        return d, t3, nodsl
    dev = d[~cv.locked_test_mask(d).values].reset_index(drop=True)
    return dev, t3, nodsl


def model():
    m = reg(SEED)
    m.set_params(n_jobs=NJ_)
    return m


def fitpred(x, tr, te, feats, y="aquatic_score", w=None):
    m = model()
    m.fit(x.loc[tr, feats], x.loc[tr, y], sample_weight=None if w is None else w)
    return m.predict(x.loc[te, feats])


def sim_weights(xs, xt):
    """Density-ratio weights for source rows xs toward target rows xt from the inputs alone (cross-fitted, so no row scores itself)."""
    X = pd.concat([xs, xt], ignore_index=True)
    lab = np.r_[np.zeros(len(xs)), np.ones(len(xt))]
    p = np.zeros(len(X))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, lab):
        c = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.7, verbose=-1, n_jobs=NJ_)
        c.fit(X.iloc[a], lab[a])
        p[b] = c.predict_proba(X.iloc[b])[:, 1]
    p = np.clip(p[:len(xs)], 1e-3, 1 - 1e-3)
    w = p / (1 - p) * (len(xs) / len(xt))
    w = np.clip(w, 0.05, 20.0)
    return w / w.mean()


def nearest_states(x):
    c = x.groupby("state")[["lat_nad83", "lon_nad83"]].mean()
    out = {}
    for s in c.index:
        dd = np.hypot(c.lat_nad83 - c.lat_nad83[s], (c.lon_nad83 - c.lon_nad83[s]) * np.cos(np.radians(c.lat_nad83[s]))).drop(s)
        out[s] = list(dd.sort_values().index[:K_NEAR])
    return out


def loso(x, feats, tag, y="aquatic_score"):
    x = x.reset_index(drop=True)
    nn = nearest_states(x)
    rows, oof = [], {k: np.full(len(x), np.nan) for k in ("pooled", "regional", "ensemble", "simweight")}
    for s in sorted(x.state.unique()):
        te = (x.state == s).values
        tr = ~te
        t0 = time.time()
        pp = fitpred(x, tr, te, feats, y)
        pr = fitpred(x, x.state.isin(nn[s]).values, te, feats, y)
        w = sim_weights(x.loc[tr, feats], x.loc[te, feats])
        ps = fitpred(x, tr, te, feats, y, w=w)
        for k, v in (("pooled", pp), ("regional", pr), ("ensemble", (pp + pr) / 2), ("simweight", ps)):
            oof[k][te] = v
        yt = x.loc[te, y].values
        rows.append(dict(state=s, n=int(te.sum()), pooled=r2(yt, pp), regional=r2(yt, pr), ensemble=r2(yt, (pp + pr) / 2), simweight=r2(yt, ps)))
        print(f"  [{tag}] {s} n={te.sum():5d} (regional from {nn[s]})  pooled {rows[-1]['pooled']:.3f}  regional {rows[-1]['regional']:.3f}  "
              f"ensemble {rows[-1]['ensemble']:.3f}  simweight {rows[-1]['simweight']:.3f}  [{time.time() - t0:.0f}s]", flush=True)
    t = pd.DataFrame(rows)
    big = t.n >= 300
    yy = x[y].values
    print(f"  [{tag}] ALL held-out rows pooled together: " + "  ".join(f"{k} {r2(yy, v):.4f}" for k, v in oof.items()), flush=True)
    print(f"  [{tag}] mean over states with n>=300 ({big.sum()} states): " + "  ".join(f"{k} {t.loc[big, k].mean():.4f}" for k in oof), flush=True)
    print(f"  [{tag}] state wins vs pooled (n>=300): " + "  ".join(f"{k} {(t.loc[big, k] > t.loc[big, 'pooled']).sum()}/{big.sum()}" for k in ("regional", "ensemble", "simweight")), flush=True)
    return t, oof


def part_A(dev, t3, nodsl):
    print("PART A: map-only leave-one-state-out, all development rows", len(dev), flush=True)
    for name, f in (("no DSL rasters (portable)", nodsl), ("with DSL rasters", t3)):
        print(f"\n{name}  ({len(f)} inputs)", flush=True)
        loso(dev, f, name.split()[0])


def part_B(dev, t3, nodsl):
    x = dev[dev.q_perched.notna()].reset_index(drop=True)
    print(f"PART B: crossings with a recorded outlet drop {len(x)} (perched share {x.q_perched.mean():.3f}); states: {x.state.value_counts().to_dict()}", flush=True)
    for name, f in (("map only (no DSL rasters)", nodsl), ("map + perched yes/no (no DSL rasters)", nodsl + ["q_perched"]),
                    ("map only (with DSL rasters)", t3), ("map + perched yes/no (with DSL rasters)", t3 + ["q_perched"])):
        print(f"\n{name}", flush=True)
        loso(x, f, name[:12])


def part_C(dev, t3, nodsl):
    st = pd.get_dummies(dev.state, prefix="st").astype(float)
    dev = pd.concat([dev, st], axis=1)
    onehot = list(st.columns)
    for task, x0, feats in (("map-only, no DSL rasters", dev, nodsl),
                            ("map + perched yes/no, no DSL rasters", dev[dev.q_perched.notna()].reset_index(drop=True), nodsl + ["q_perched"])):
        x = x0.reset_index(drop=True)
        big = [s for s, n in x.state.value_counts().items() if n >= 800]
        print(f"\nPART C: {task}; rows {len(x)}; states with per-state models (>=800 rows): {big}", flush=True)
        oof = {k: np.full(len(x), np.nan) for k in ("pooled", "pooled+state", "per-state", "ensemble")}
        for tr, te in cv.random_group_kfold(x, "huc8", 5, SEED):
            tr, te = np.asarray(tr), np.asarray(te)
            oof["pooled"][te] = fitpred(x, tr, te, feats)
            oof["pooled+state"][te] = fitpred(x, tr, te, feats + onehot)
            oof["per-state"][te] = oof["pooled+state"][te]            # states without their own model fall back to the pooled+state model
            for s in big:
                trs = tr[(x.state.values[tr] == s)]
                tes = te[(x.state.values[te] == s)]
                if len(tes):
                    oof["per-state"][tes] = fitpred(x, trs, tes, feats)
            oof["ensemble"][te] = (oof["pooled+state"][te] + oof["per-state"][te]) / 2
            print("   fold done", flush=True)
        yy = x.aquatic_score.values
        print("  overall:  " + "  ".join(f"{k} {r2(yy, v):.4f}" for k, v in oof.items()), flush=True)
        for s in big:
            m = (x.state == s).values
            print(f"  {s} n={m.sum():5d}  " + "  ".join(f"{k} {r2(yy[m], v[m]):.3f}" for k, v in oof.items()), flush=True)


if __name__ == "__main__":
    dev, t3, nodsl = load()
    print(f"dev rows {len(dev)}, inputs: t3 {len(t3)}, no-DSL {len(nodsl)}", flush=True)
    {"A": part_A, "B": part_B, "C": part_C}[sys.argv[1]](dev, t3, nodsl)
    print("done", flush=True)
