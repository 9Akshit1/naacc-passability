"""Quick-check tier (NOT a GIS-only model): GIS features + a few coarse volunteer-observable answers.
NY development crossings with survey fields; HUC8 grouped CV, 3 splits. Locked set untouched here.
Coarse inputs are derived from the surveyed values by binning, i.e. what a volunteer could report in seconds."""
import numpy as np, pandas as pd, lightgbm as lgb
from src import plunkett_cv as cv, plunkett_features as pf
d = pd.read_parquet("data/processed/plunkett_features_osm.parquet"); lock = cv.locked_test_mask(d)
old = pd.read_parquet("data/processed/subset_predictors.parquet")
g = pf.feature_groups(d); g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]; base = g["PLUNKETT"] + g["NBI"] + g["OSM"]
num = lambda s: pd.to_numeric(s, errors="coerce")
o = pd.DataFrame({"Crossing_Code": old.Crossing_Code})
drop = num(old.Outlet_Drop_To_Water_Surface)
o["q_perched"] = np.where(drop.isna(), np.nan, (drop > 0.25).astype(float))            # perched outlet yes/no (> 3 in)
o["q_drop3"] = np.where(drop.isna(), np.nan, np.digitize(drop.fillna(0), [0.05, 0.5]))   # none / small / large
o["q_type"] = old.Crossing_Type.astype("category").cat.codes
o["q_span"] = old.Crossing_Span.astype("category").cat.codes
o["q_inlet_grade"] = old.Inlet_Grade.astype("category").cat.codes
x = d.loc[~lock].merge(o, left_on="crossing_code", right_on="Crossing_Code").reset_index(drop=True)
print("NY dev crossings with survey fields:", len(x), " perched share:", round(float(x.q_perched.mean()), 3), " missing:", round(float(x.q_perched.isna().mean()), 3))
mk = lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=8)
def r2(f):
    s = cv.summarize(cv.cross_val_oof(x, f, mk, "huc8", seeds=(0, 1, 2))); return s["r2_mean"], s["r2_sd"]
for name, f in [("GIS only", base), ("GIS + perched yes/no", base + ["q_perched"]), ("GIS + drop none/small/large", base + ["q_drop3"]),
                ("GIS + perched + type", base + ["q_perched", "q_type"]), ("GIS + perched + type + constriction + inlet grade", base + ["q_perched", "q_type", "q_span", "q_inlet_grade"])]:
    a, s = r2(f); print(f"{name:55s} R2={a:.4f} (sd {s:.4f})", flush=True)
