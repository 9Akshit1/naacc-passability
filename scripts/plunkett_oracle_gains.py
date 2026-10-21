"""ORACLE analysis (an upper bound, NOT a deployable model): how much R2 would a perfectly known field property add to the
T3 GIS model?  NY development crossings with survey fields. Tells us which properties are worth estimating remotely.
HUC8 CV, 3 random splits, LightGBM. Field values are used only as an analysis device; locked set untouched."""
import numpy as np, pandas as pd, lightgbm as lgb
from src import plunkett_cv as cv, plunkett_features as pf
d = pd.read_parquet("data/processed/plunkett_features_osm.parquet"); lock = cv.locked_test_mask(d)
old = pd.read_parquet("data/processed/subset_predictors.parquet")
g = pf.feature_groups(d); g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]; base = g["PLUNKETT"] + g["NBI"] + g["OSM"]
num = lambda s: pd.to_numeric(s, errors="coerce")
o = pd.DataFrame({"Crossing_Code": old.Crossing_Code})
o["o_type"] = old.Crossing_Type.astype("category").cat.codes
o["o_n_culverts"] = num(old.Number_Of_Culverts)
o["o_material"] = old.Material.astype("category").cat.codes
o["o_struct_length"] = num(old.Crossing_Structure_Length)
o["o_fill_height"] = num(old.Road_Fill_Height)
o["o_inlet_height"] = num(old.Inlet_Height); o["o_outlet_height"] = num(old.Outlet_Height)
o["o_inlet_open"] = num(old.Inlet_Openness); o["o_outlet_open"] = num(old.Outlet_Openness)
o["o_outlet_drop"] = num(old.Outlet_Drop_To_Water_Surface)
o["o_slope_pct"] = num(old.Slope_Percent)
o["o_crossing_span"] = old.Crossing_Span.astype("category").cat.codes
o["o_inlet_grade"] = old.Inlet_Grade.astype("category").cat.codes
o["o_flow_condition"] = old.Flow_Condition.astype("category").cat.codes
x = d.loc[~lock].merge(o, left_on="crossing_code", right_on="Crossing_Code").reset_index(drop=True)
print("NY dev crossings with survey fields:", len(x))
mk = lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=8)
def r2(f): return cv.summarize(cv.cross_val_oof(x, f, mk, "huc8", seeds=(0, 1, 2)))["r2_mean"]
b = r2(base); print(f"GIS model (T3 features) R2 = {b:.4f}", flush=True)
sets = {"crossing TYPE (bridge/culvert/multi/ford...)": ["o_type"], "number of culverts/cells": ["o_n_culverts"], "structure material": ["o_material"],
        "structure length": ["o_struct_length"], "road fill height": ["o_fill_height"], "structure inlet+outlet HEIGHT": ["o_inlet_height", "o_outlet_height"],
        "structure OPENNESS (inlet+outlet)": ["o_inlet_open", "o_outlet_open"], "outlet DROP": ["o_outlet_drop"], "structure slope %": ["o_slope_pct"],
        "constriction (crossing span class)": ["o_crossing_span"], "inlet grade": ["o_inlet_grade"], "flow condition": ["o_flow_condition"]}
for n, f in sets.items(): print(f"  + oracle {n:46s} R2={r2(base+f):.4f}  (gain {r2(base+f)-b:+.4f})", flush=True)
allsz = ["o_inlet_height", "o_outlet_height", "o_inlet_open", "o_outlet_open", "o_struct_length"]
print(f"  + oracle ALL size fields {allsz}: R2={r2(base+allsz):.4f}")
print(f"  + oracle type + size fields: R2={r2(base+allsz+['o_type','o_n_culverts']):.4f}")
