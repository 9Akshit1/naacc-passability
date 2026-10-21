"""Re-check 'which NAACC score components can GIS predict' with the corrected multi-state features.

The earlier claim (METHODOLOGY / WRITEUP) used the NY pipeline whose catchment area was defective. Here: NY dev-set
crossings present in both datasets; components recomputed by src/naacc_score.py (weights/tables verified against NAACC's
published document 2026-10-08); predicted from Plunkett + NBI + OSM features with HUC8-group CV (5 random splits).
Also tests NAACC's documented multiple-structure rule via the reconstruction error."""
import numpy as np, pandas as pd, lightgbm as lgb
from src import plunkett_cv as cv, plunkett_features as pf, naacc_score as ns

d = pd.read_parquet("data/processed/plunkett_features_osm.parquet"); lock = cv.locked_test_mask(d)
old = pd.read_parquet("data/processed/subset_predictors.parquet")
res = ns.compute_passability(old)
comp = res.components.copy(); comp["composite"] = res.composite; comp["final_recon"] = res.final; comp["published"] = pd.to_numeric(old["passability"], errors="coerce")
comp["Crossing_Code"] = old["Crossing_Code"].values; comp["n_culverts"] = pd.to_numeric(old["Number_Of_Culverts"], errors="coerce").values
g = pf.feature_groups(d); g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]
feats = g["PLUNKETT"] + g["NBI"] + g["OSM"]
x = d.loc[~lock, ["crossing_code", "huc8", "state"] + feats].merge(comp, left_on="crossing_code", right_on="Crossing_Code", how="inner").reset_index(drop=True)
print("NY dev crossings with recomputed components:", len(x))
mk = lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=8)
W = ns.WEIGHTS; rows = []
for c in list(W) + ["composite", "final_recon", "published"]:
    s = x[x[c].notna()].reset_index(drop=True)
    if s[c].nunique() < 3 or len(s) < 1000: rows.append({"component": c, "n": len(s), "r2": np.nan}); continue
    s = s.assign(aquatic_score=s[c])
    r = cv.cross_val_oof(s, feats, mk, "huc8", seeds=(0, 1, 2))
    rows.append({"component": c, "weight": W.get(c), "n": len(s), "r2": float(np.mean([q["metrics"]["r2"] for q in r])), "share_equal_1": float((s[c] >= 0.999).mean())})
    print(rows[-1], flush=True)
t = pd.DataFrame(rows); t.to_csv("results/plunkett/component_predictability.csv", index=False)
print(t.round(3).to_string())
# multiple-structure rule: reconstruction error vs number of culverts
x["err"] = (x.final_recon - x.published).abs(); x["multi"] = x.n_culverts.fillna(1) > 1
print("reconstruction MAE single-structure vs multiple-structure crossings:", x.groupby("multi").err.agg(["mean", "size"]).round(4).to_dict())
