"""Do StreamCat and NHDPlus attributes help the score model, and can they replace the DSL catchment inputs?

It uses development rows only, whole-HUC8 folds, seeds 0 to 2 and the fixed LightGBM regression model. The reference inputs are
Plunkett's 39, the bridge inventory and OpenStreetMap structures, as in every earlier test.

Part 1 asks whether there is any new information in the attributes. It adds each attribute group alone, then all of them, then
all of them only where the crossing is within 50 m of the stream line.
Part 2 replaces DSL's catchment area and gradient with public data, to see whether a new location could be scored without his
rasters. It compares the reference, the reference without d8accum, the reference without d8accum plus the StreamCat area and
NHDPlus slope, and the same on the crossings within 50 m of the stream line, which is what a candidate placed on a stream line
would look like.
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src import plunkett_cv as cv, plunkett_features as pf, streamcat_feats as sf
from src.plunkett_final import LGBM_REG

SEEDS = (0, 1, 2)
SNAP = ["snap_m", "snap2_m"]


def reg(seed):
    return lgb.LGBMRegressor(subsample_freq=1, verbose=-1, n_jobs=10, random_state=seed, **LGBM_REG)


def r2(dev, feats, y, mask=None):
    scores = []
    for seed in SEEDS:
        oof = pd.Series(np.nan, index=dev.index)
        for tr, te in cv.random_group_kfold(dev, "huc8", 5, seed):
            oof.loc[te] = reg(seed).fit(dev.loc[tr, feats], y.loc[tr]).predict(dev.loc[te, feats])
        m = np.ones(len(dev), bool) if mask is None else mask
        scores.append(r2_score(y[m], oof[m]))
    return float(np.mean(scores)), float(np.std(scores))


if __name__ == "__main__":
    d = pd.read_parquet("data/processed/plunkett_features_osm.parquet")
    d, _ = pf.add_derived(d)
    g = pf.feature_groups(d)                      # before the new columns are joined, or they would count as baseline
    g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]
    base = g["PLUNKETT"] + g["NBI"] + g["OSM"]
    ex = sf.build(d)
    linked = ex["comid"].notna()
    print(f"linked to a stream: {linked.mean():.3f}; within 50 m: {(ex.snap_m <= 50).mean():.3f}", flush=True)
    d = pd.concat([d, ex.drop(columns="comid")], axis=1)
    lock = cv.locked_test_mask(d)
    dev = d[~lock].copy().reset_index(drop=True)
    assert not any(c.startswith(("sc_", "vaa_")) or c in ("snap_m", "snap2_m", "order_line", "ftype_line") for c in base)
    y = dev["aquatic_score"]
    near = (dev["snap_m"] <= 50).values
    print(f"development rows {len(dev)}, linked {dev.snap_m.notna().mean():.3f}, within 50 m {near.mean():.3f}", flush=True)

    sc_cols = [c for c in dev.columns if c.startswith("sc_")]

    def cols_for(group):
        names = set(sf.GROUPS[group])
        return [c for c in sc_cols if c[3:-2] in names or c[3:-3] in names]

    vaa_cols = [c for c in dev.columns if c.startswith("vaa_")]
    area_cols = ["sc_wsarea", "sc_catarea"] if "sc_wsarea" in dev else [c for c in sc_cols if "area" in c]
    area_cols = [c for c in sc_cols if c in ("sc_wsareaws", "sc_catareacat", "sc_wsarea", "sc_catarea")] or area_cols
    print("area columns:", area_cols, "| sc columns", len(sc_cols), "| vaa columns", len(vaa_cols), flush=True)

    masked = dev.copy()
    for c in sc_cols + vaa_cols:
        masked.loc[~near, c] = np.nan            # trust the link only close to the stream line
    results = []

    def run(name, frame, feats, mask=None):
        t = time.time()
        m, s = r2(frame, feats, y, mask)
        results.append((name, m, s))
        print(f"{name:62s} R2={m:.4f} (sd {s:.4f})  [{time.time() - t:.0f}s]", flush=True)
        return m

    print("\nPART 1: added information", flush=True)
    ref = run("reference (T3)", dev, base)
    run("reference + snap distance only", dev, base + SNAP)
    for grp in sf.GROUPS:
        run(f"+ StreamCat {grp} (ws+cat) + snap", dev, base + cols_for(grp) + SNAP)
    run("+ NHDPlus VAA (slope, order, flow, neighbours) + snap", dev, base + vaa_cols + SNAP)
    run("+ ALL StreamCat + VAA + snap", dev, base + sc_cols + vaa_cols + SNAP)
    run("+ ALL StreamCat + VAA, values only within 50 m, + snap", masked, base + sc_cols + vaa_cols + SNAP)
    run("+ ALL, evaluated only on rows within 50 m (reference below)", dev, base + sc_cols + vaa_cols + SNAP, near)
    run("reference, evaluated only on rows within 50 m", dev, base, near)

    print("\nPART 2: public data in place of the DSL catchment area and gradient", flush=True)
    dsl_area = [c for c in base if c in ("d8accum", "log_d8accum")]
    dsl_grad = [c for c in base if c == "gradient"]
    no_area = [c for c in base if c not in dsl_area]
    pub = area_cols + ["vaa_log_totda", "vaa_slope", "vaa_order", "vaa_log_flow", "snap_m", "snap2_m"]
    pub = [c for c in pub if c in dev]
    for name, feats, mask in [
        ("all rows: reference", base, None),
        ("all rows: without d8accum", no_area, None),
        ("all rows: without d8accum, + StreamCat/NHDPlus area, slope, flow", no_area + pub, None),
        ("within 50 m: reference", base, near),
        ("within 50 m: without d8accum", no_area, near),
        ("within 50 m: without d8accum, + StreamCat/NHDPlus area, slope, flow", no_area + pub, near),
        ("within 50 m: without d8accum, gradient; + StreamCat/NHDPlus", [c for c in no_area if c not in dsl_grad] + pub, near),
    ]:
        run(name, dev, feats, mask)
    pd.DataFrame(results, columns=["experiment", "r2", "sd"]).round(4).to_csv("results/plunkett/streamcat_eval_2026-11-04.csv", index=False)
    print("done", flush=True)
