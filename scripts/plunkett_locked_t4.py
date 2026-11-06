"""Score two more models on the test basins, once each. They were chosen after the StreamCat and NHDPlus tests on the development set.

    T4   T3 inputs plus StreamCat (ws and cat), NHDPlus value added attributes and snap distances (development R2 0.3352
         against 0.3252)
    T4P  Plunkett's road and income inputs, the bridge inventory, OpenStreetMap and the same public data, with no DSL raster
         input at all (no gradient, dem, d8accum, elev_sd/range, incise or bdev) (development R2 0.2765)

The model is src/plunkett_final.FinalEnsemble, unchanged, and an entry that is already in locked_results.json is refused. A paired
bootstrap over HUC8 watersheds compares both against the saved T3 predictions.
"""
import json
import os

import numpy as np
import pandas as pd

from src import plunkett_cv as cv, plunkett_features as pf, streamcat_feats as sf
from src.plunkett_final import FinalEnsemble

OUT = "results/plunkett/locked_results.json"
done = json.load(open(OUT)) if os.path.exists(OUT) else {}
d = pd.read_parquet("data/processed/plunkett_features_osm.parquet")
d, _ = pf.add_derived(d)
g = pf.feature_groups(d)
g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]
d = pd.concat([d, sf.build(d).drop(columns="comid")], axis=1)
lock = cv.locked_test_mask(d)
dev, test = d[~lock], d[lock]
pub = [c for c in d.columns if c.startswith(("sc_", "vaa_"))] + ["snap_m", "snap2_m"]
raster = [c for c in g["PLUNKETT"] if c.startswith(("elev_", "incise1_", "bdev_")) or c in ("gradient", "dem", "d8accum")]
tiers = {"T4_t3+streamcat+vaa": g["PLUNKETT"] + g["NBI"] + g["OSM"] + pub,
         "T4P_no_dsl_rasters": [c for c in g["PLUNKETT"] if c not in raster] + g["NBI"] + g["OSM"] + pub}
y = test["aquatic_score"]
t3 = pd.read_csv("results/plunkett/locked_predictions_T3.csv").set_index("crossing_code")["pred"].reindex(test["crossing_code"]).values
rng = np.random.default_rng(0)
ug = np.unique(test.huc8.values)
idx = {u: np.where(test.huc8.values == u)[0] for u in ug}
boot = [np.concatenate([idx[u] for u in rng.choice(ug, len(ug))]) for _ in range(2000)]


def r2(yy, pp):
    return 1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()


for name, feats in tiers.items():
    if name in done:
        print("REFUSED:", name, "already scored"); continue
    p = FinalEnsemble().fit(dev[feats], dev.aquatic_score).predict(test[feats])
    m = cv.metrics(y, p)
    yv = y.values
    dl = np.array([r2(yv[s], p[s]) - r2(yv[s], t3[s]) for s in boot])
    m["delta_r2_vs_T3"] = {"point": float(r2(yv, p) - r2(yv, t3)), "ci95": [float(np.quantile(dl, .025)), float(np.quantile(dl, .975))]}
    m["n_features"] = len(feats)
    m["within_50m_share"] = float((test.snap_m <= 50).mean())
    near = (test.snap_m <= 50).values
    m["r2_within_50m"] = float(r2(yv[near], p[near])); m["r2_T3_within_50m"] = float(r2(yv[near], t3[near]))
    done[name] = m
    test.assign(pred=p)[["crossing_code", "state", "huc8", "aquatic_score", "pred"]].to_csv(
        f"results/plunkett/locked_predictions_{name.split('_')[0]}.csv", index=False)
    print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}), flush=True)
json.dump(done, open(OUT, "w"), indent=1)
