"""How far can a model get with no DSL raster inputs at all? A new candidate location would not have them.

It uses development rows only, whole-HUC8 folds, seeds 0 to 2 and the fixed LightGBM regression model. It drops every Plunkett
predictor that comes from DSL's elevation and flow rasters (gradient, dem, d8accum, elev_sd_*, elev_range_*, incise1_* and bdev_*),
keeps road ADT, class and income, and replaces the rest with StreamCat, NHDPlus value added attributes and snap distance. The
result is compared with the full T3 model and with T3 plus the public data (the T4 candidate).
"""
import time

import numpy as np
import pandas as pd

from scripts.plunkett_streamcat_eval import SNAP, r2
from src import plunkett_cv as cv, plunkett_features as pf, streamcat_feats as sf

if __name__ == "__main__":
    d = pd.read_parquet("data/processed/plunkett_features_osm.parquet")
    d, _ = pf.add_derived(d)
    g = pf.feature_groups(d)
    g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]
    base = g["PLUNKETT"] + g["NBI"] + g["OSM"]
    d = pd.concat([d, sf.build(d).drop(columns="comid")], axis=1)
    dev = d[~cv.locked_test_mask(d)].copy().reset_index(drop=True)
    y = dev["aquatic_score"]
    near = (dev["snap_m"] <= 50).values
    raster = [c for c in g["PLUNKETT"] if c.startswith(("elev_", "incise1_", "bdev_")) or c in ("gradient", "dem", "d8accum")]
    keep_pl = [c for c in g["PLUNKETT"] if c not in raster]
    print("kept Plunkett columns:", keep_pl, flush=True)
    pub = [c for c in dev.columns if c.startswith(("sc_", "vaa_"))] + SNAP
    for name, feats, mask in [
        ("all rows: T3 reference", base, None),
        ("all rows: no DSL rasters, public data instead", keep_pl + g["NBI"] + g["OSM"] + pub, None),
        ("all rows: T3 + public data (T4)", base + pub, None),
        ("within 50 m: T3 reference", base, near),
        ("within 50 m: no DSL rasters, public data instead", keep_pl + g["NBI"] + g["OSM"] + pub, near),
        ("within 50 m: T3 + public data (T4)", base + pub, near),
    ]:
        t = time.time()
        m, s = r2(dev, feats, y, mask)
        print(f"{name:55s} R2={m:.4f} (sd {s:.4f}) [{time.time() - t:.0f}s]", flush=True)
    print("done", flush=True)
