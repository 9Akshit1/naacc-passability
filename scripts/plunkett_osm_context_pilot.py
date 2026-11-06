"""Does road and waterway context from OpenStreetMap help? This pilot uses the two states whose extracts are on disk (PA and RI).

It uses development rows only and the same set-up as the other development tests: the fixed LightGBM regression model, HUC8 group
folds and 3 seeds, within these two states. The reference is Plunkett's predictors, the bridge inventory and OpenStreetMap
structures. If the gain is not clearly above the spread between seeds (about 0.003), it is not worth downloading the other eleven
extracts.
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import osm_context, plunkett_cv as cv, plunkett_features as pf
from src.plunkett_final import LGBM_REG
from pathlib import Path

PBF = {"PA": "pennsylvania", "RI": "rhode-island"}
OUT = Path("data/processed/osm_context_pa_ri.parquet")


def lgbm():
    p = dict(LGBM_REG)
    return lambda: lgb.LGBMRegressor(subsample_freq=1, verbose=-1, n_jobs=10, random_state=0, **p)


if __name__ == "__main__":
    d = pd.read_parquet("data/processed/plunkett_features_osm.parquet")
    d, _ = pf.add_derived(d)
    lock = cv.locked_test_mask(d)
    g = pf.feature_groups(d)                      # before the new columns are joined, or they would count as baseline
    g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]
    base = g["PLUNKETT"] + g["NBI"] + g["OSM"]
    if OUT.exists():
        ctx = pd.read_parquet(OUT)
    else:
        parts = []
        for st, name in PBF.items():
            t = time.time()
            sub = d[d.state == st]
            parts.append(osm_context.context_for(sub, Path(f"data/raw/osm/pbf/{name}.osm.pbf")))
            print(st, len(sub), f"{time.time() - t:.0f}s", flush=True)
        ctx = pd.concat(parts)
        ctx.to_parquet(OUT)
    print(ctx.notna().mean().round(3).to_string(), flush=True)

    d = d.join(ctx, how="inner")
    dev = d[~lock.reindex(d.index)].copy().reset_index(drop=True)
    new = list(ctx.columns)
    assert not set(new) & set(base)
    print(f"PA+RI development rows: {len(dev)}", flush=True)
    for name, feats in [("reference", base), ("+ road and waterway context", base + new),
                        ("+ waterway only", base + [c for c in new if c.startswith("osm_water")]),
                        ("+ road only", base + [c for c in new if c.startswith("osm_road")]),
                        ("+ skew only", base + ["osm_skew_deg"])]:
        s = cv.summarize(cv.cross_val_oof(dev, feats, lgbm(), "huc8", seeds=(0, 1, 2)))
        print(f"{name:34s} R2={s['r2_mean']:.4f}+-{s['r2_sd']:.4f} MAE={s['mae_mean']:.4f}", flush=True)
    print("done", flush=True)
