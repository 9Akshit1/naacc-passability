"""Plunkett's question: if the earlier NY-only pipeline used d8accum for catchment area, does it improve?

Same crossings (NY surveys present in both datasets), same old feature set and model
(project HistGradientBoosting pipeline), same HUC8-group folds/seeds. Only the catchment
columns differ.  A = old NLDI area | B = d8accum-based area (30 m cells -> km2) | C = both.
No locked-set rows are used (development-set crossings only)."""
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
