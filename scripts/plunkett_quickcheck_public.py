"""What is a one-question volunteer check worth at a new location, where the model has no DSL rasters?

The set-up is the same as plunkett_quickcheck_tier.py: New York development crossings with survey fields, HUC8 group CV and 3 splits,
without the test basins. The map inputs are the ones a new location could have: road ADT, class and income, the bridge inventory,
OpenStreetMap, StreamCat, NHDPlus and the snap distance (no gradient, dem, d8accum, elev_*, incise_* or bdev_*).
q_perched is 1 when the outlet drop is more than 0.25 ft. The error test flips answers at random with probability e in both
training and test data, which is an assumption about volunteer error and not a measurement.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import plunkett_cv as cv, plunkett_features as pf, streamcat_feats as sf

if __name__ == "__main__":
    d = pd.read_parquet("data/processed/plunkett_features_osm.parquet")
    d, _ = pf.add_derived(d)
    g = pf.feature_groups(d)
    g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]
    t3 = g["PLUNKETT"] + g["NBI"] + g["OSM"]
    raster = [c for c in g["PLUNKETT"] if c.startswith(("elev_", "incise1_", "bdev_")) or c in ("gradient", "dem", "d8accum")]
    d = pd.concat([d, sf.build(d).drop(columns="comid")], axis=1)
    pub = [c for c in d.columns if c.startswith(("sc_", "vaa_"))] + ["snap_m", "snap2_m", "order_line", "ftype_line"]
    nodsl = [c for c in g["PLUNKETT"] if c not in raster] + g["NBI"] + g["OSM"] + pub
    lock = cv.locked_test_mask(d)
    old = pd.read_parquet("data/processed/subset_predictors.parquet")
    drop = pd.to_numeric(old.Outlet_Drop_To_Water_Surface, errors="coerce")
    o = pd.DataFrame({"Crossing_Code": old.Crossing_Code, "q_perched": np.where(drop.isna(), np.nan, (drop > 0.25).astype(float))})
    x = d.loc[~lock].merge(o, left_on="crossing_code", right_on="Crossing_Code").reset_index(drop=True)
    x = x[x.q_perched.notna()].reset_index(drop=True)
    print(f"NY development crossings with a perched answer: {len(x)}, perched share {x.q_perched.mean():.3f}", flush=True)
    mk = lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=8)

    def r2(frame, f):
        s = cv.summarize(cv.cross_val_oof(frame, f, mk, "huc8", seeds=(0, 1, 2)))
        return s["r2_mean"], s["r2_sd"]

    for name, f in [("T3 map inputs (with DSL rasters)", t3), ("no DSL rasters, public data", nodsl), ("StreamCat + NHDPlus + snap only", pub)]:
        a, s = r2(x, f); b, s2 = r2(x, f + ["q_perched"])
        print(f"{name:36s} map only R2={a:.4f} (sd {s:.4f})   + perched yes/no R2={b:.4f} (sd {s2:.4f})", flush=True)
    print("\nno DSL rasters + perched yes/no, with random answer errors (assumption):", flush=True)
    rng = np.random.default_rng(0)
    for e in (0.0, 0.1, 0.2, 0.3):
        y = x.copy()
        flip = rng.random(len(y)) < e
        y["q_perched"] = np.where(flip, 1 - y.q_perched, y.q_perched)
        a, s = r2(y, nodsl + ["q_perched"])
        print(f"  error rate {e:.0%}: R2={a:.4f} (sd {s:.4f})", flush=True)
    print("done", flush=True)
