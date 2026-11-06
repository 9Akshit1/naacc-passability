"""Plunkett's question: if the earlier New York pipeline had used d8accum for catchment area, would it improve?

It uses the same crossings (New York surveys that are in both datasets), the same old features and model (the project's
HistGradientBoosting pipeline) and the same HUC8 group folds and seeds. Only the catchment columns change: A is the old NLDI area,
B is the area from d8accum (30 m cells converted to km2), and C is both. Only development crossings are used.
"""
import numpy as np, pandas as pd
from src import models as M, plunkett_cv as cv, plunkett_data as pdm
d = pdm.load(); lock = cv.locked_test_mask(d)
old = pd.read_parquet("data/processed/subset_predictors.parquet")
x = old.merge(d[["crossing_code", "d8accum", "huc8", "state"]].assign(_lock=lock.values), left_on="Crossing_Code", right_on="crossing_code")
x = x[~x._lock].copy().reset_index(drop=True)
x["passability_target"] = x["passability"]; x["aquatic_score"] = x["passability"]
x["d8_area_km2"] = x["d8accum"] * 900 / 1e6; x["log_d8_area"] = np.log10(x["d8_area_km2"])
print("crossings:", len(x), "HUC8:", x.huc8.nunique())
base_num = [c for c in M.NUMERIC_FEATURES_DEFAULT if c in x.columns]
cat = [c for c in M.CATEGORICAL_FEATURES_DEFAULT if c in x.columns]
area_old = ["log_drainage_area", "drainage_area_km2"]
sets = {"A old NLDI area": base_num,
        "B d8accum area (replaces old)": [c for c in base_num if c not in area_old] + ["d8_area_km2", "log_d8_area"],
        "C both": base_num + ["d8_area_km2", "log_d8_area"],
        "D no catchment area at all": [c for c in base_num if c not in area_old]}
for name, num in sets.items():
    mk = lambda num=num: M.make_models(num, cat)["hist_gbm"]
    s = cv.summarize(cv.cross_val_oof(x, num + cat, mk, "huc8", seeds=(0, 1, 2)))
    print(f"{name:32s} R2={s['r2_mean']:.4f}+-{s['r2_sd']:.4f} MAE={s['mae_mean']:.4f}", flush=True)
