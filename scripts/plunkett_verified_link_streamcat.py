"""Do StreamCat / NHDPlus attributes help once the COMID link is verified against d8accum?
NY dev crossings present in the old pipeline. 'Verified link' = old NLDI area within a factor 2 of the d8accum area
(30 m cells assumed; the verification only needs rough agreement). T3 features +/- old-pipeline extras (sc_*,
stream_slope, stream_order). HUC8 CV, 3 random splits, LightGBM default. Locked set untouched."""
import numpy as np, pandas as pd, lightgbm as lgb
from src import plunkett_cv as cv, plunkett_features as pf
d = pd.read_parquet("data/processed/plunkett_features_osm.parquet"); lock = cv.locked_test_mask(d)
old = pd.read_parquet("data/processed/subset_predictors.parquet")
extra = [c for c in old.columns if c.startswith("sc_") and not c.endswith("_local") and c != "sc_watershed_area_km2"] + ["stream_slope", "stream_order"]
extra = [c for c in extra if c in old.columns]; print("extras:", extra)
g = pf.feature_groups(d); g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]; base = g["PLUNKETT"] + g["NBI"] + g["OSM"]
x = d.loc[~lock].merge(old[["Crossing_Code", "drainage_area_km2"] + extra], left_on="crossing_code", right_on="Crossing_Code", how="inner").reset_index(drop=True)
x["ratio"] = np.log10(x.drainage_area_km2 / (x.d8accum * 900 / 1e6)); x["verified"] = x.ratio.abs() < np.log10(2)
print(f"NY dev crossings: {len(x)}; verified link: {x.verified.sum()} ({x.verified.mean():.1%})")
mk = lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=4)
for lab, sub in [("ALL linked crossings", x), ("VERIFIED-link crossings only", x[x.verified].reset_index(drop=True))]:
    for name, f in [("T3 base", base), ("T3 + StreamCat/NHDPlus extras", base + extra)]:
        r = cv.summarize(cv.cross_val_oof(sub, f, mk, "huc8", seeds=(0, 1, 2)))
        print(f"{lab:30s} n={len(sub):6d}  {name:32s} R2={r['r2_mean']:.4f} (sd {r['r2_sd']:.4f})", flush=True)
