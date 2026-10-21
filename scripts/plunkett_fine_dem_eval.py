"""Pilot evaluation of fine-scale (1 m-class) ground-profile features. 6,000 random NY development crossings.
(1) distinctness / coverage, (2) passability score with and without fd_* features (paired folds, 5 random splits),
(3) outlet drop (field-measured, ft) as an auxiliary target: R2 and perched-outlet AUC with and without fd_*.
Locked set untouched."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from src import fine_dem, plunkett_cv as cv, plunkett_features as pf
d = pd.read_parquet("data/processed/plunkett_features_osm.parquet")
z = np.load("data/processed/fine_dem_pilot_z.npy"); codes = pd.read_csv("data/processed/fine_dem_pilot_codes.csv").crossing_code
fd = fine_dem.derive(z); fd["crossing_code"] = codes.values
old = pd.read_parquet("data/processed/subset_predictors.parquet")
x = d.merge(fd, on="crossing_code").merge(old[["Crossing_Code", "Outlet_Drop_To_Water_Surface", "Road_Fill_Height"]], left_on="crossing_code", right_on="Crossing_Code").reset_index(drop=True)
x["drop_ft"] = pd.to_numeric(x.Outlet_Drop_To_Water_Surface, errors="coerce"); x["fill_ft"] = pd.to_numeric(x.Road_Fill_Height, errors="coerce")
FD = [c for c in fd.columns if c.startswith("fd_")]
print("crossings:", len(x), " | share with >=40 distinct elevations of 49 (fine/lidar-grade):", round(float((x.fd_distinct >= 40).mean()), 3),
      " | median distinct:", float(x.fd_distinct.median()))
g = pf.feature_groups(d); g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]; base = g["PLUNKETT"] + g["NBI"] + g["OSM"]
mk = lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=10)
S = (0, 1, 2, 3, 4)
def per_split(df, feats, target):
    df = df.assign(aquatic_score=df[target]); return [r["metrics"]["r2"] for r in cv.cross_val_oof(df, feats, mk, "huc8", seeds=S)]
a, b = per_split(x, base, "aquatic_score"), per_split(x, base + FD, "aquatic_score")
dd = np.array(b) - np.array(a)
print(f"\nPASSABILITY SCORE  T3 base R2 = {np.mean(a):.4f}   + fine-DEM features = {np.mean(b):.4f}   paired gain {dd.mean():+.4f} (sd {dd.std(ddof=1):.4f}, {int((dd>0).sum())}/5 splits better)")
y = x[x.drop_ft.notna()].reset_index(drop=True); y["drop_log"] = np.log1p(y.drop_ft.clip(lower=0))
print(f"\nOUTLET DROP target: n={len(y)}  share perched (>0.25 ft): {(y.drop_ft>0.25).mean():.3f}")
a2, b2 = per_split(y, base, "drop_log"), per_split(y, base + FD, "drop_log")
print(f"  log(1+drop) R2: base {np.mean(a2):.4f} -> +fine-DEM {np.mean(b2):.4f}  (gain {np.mean(b2)-np.mean(a2):+.4f})")
y["perched"] = (y.drop_ft > 0.25).astype(float)
clf = lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=10)
def auc(feats):
    out = []
    for sd in S:
        oof = pd.Series(np.nan, index=y.index)
        for tr, te in cv.splits(y, "huc8", seed=sd):
            oof.loc[te] = clf().fit(y.loc[tr, feats], y.loc[tr, "perched"]).predict_proba(y.loc[te, feats])[:, 1]
        out.append(roc_auc_score(y.perched, oof))
    return np.mean(out)
print(f"  perched-outlet AUC: base {auc(base):.4f} -> +fine-DEM {auc(base+FD):.4f}")
# which fd features matter most for the drop target
m = mk().fit(y[base + FD], y.drop_log); imp = pd.Series(m.booster_.feature_importance("gain"), index=base + FD)
print("  top fine-DEM features by gain (drop target):", imp[FD].sort_values(ascending=False).head(6).round(0).to_dict())
# direct physical check: embankment height proxy vs surveyed road fill height
v = x[x.fill_ft.between(0, 60)]
from scipy.stats import spearmanr
print(f"\nPHYSICAL CHECK  Spearman(fd_embank_15 [m], surveyed Road_Fill_Height [ft]) = {spearmanr(v.fd_embank_15, v.fill_ft, nan_policy='omit')[0]:.3f}  (n={len(v)})")
