"""Dev-set tests of derived features and sample weighting (T3 baseline = Plunkett + NBI + OSM, LightGBM default).
HUC8-group and leave-one-state-out CV; 3 seeds for HUC8. Locked set not read."""
import numpy as np, pandas as pd, lightgbm as lgb, time
from src import plunkett_cv as cv, plunkett_features as pf
d = pd.read_parquet("data/processed/plunkett_features_osm.parquet")
d, new = pf.add_derived(d); dev = d[~cv.locked_test_mask(d)].copy()
g = pf.feature_groups(d); g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]; base = g["PLUNKETT"] + g["NBI"] + g["OSM"]
mk = lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=10)
sets = {"T3 base": base, "+ log_area, stream power": base + ["log_area_km2", "log_stream_power"],
        "+ terrain-scale features": base + new["SCALE"],
        "+ constriction (struct size / expected width)": base + ["log_area_km2"] + new["CONSTRICTION"],
        "+ scale + constriction": base + ["log_area_km2"] + new["SCALE"] + new["CONSTRICTION"]}
for name, feats in sets.items():
    pass
    for scheme in ["huc8", "state"]:
        t = time.time(); s = cv.summarize(cv.cross_val_oof(dev, feats, mk, scheme, seeds=(0, 1, 2)))
        print(f"{name:46s} {scheme:6s} R2={s['r2_mean']:.4f}+-{s['r2_sd']:.4f} MAE={s['mae_mean']:.4f} ({time.time()-t:.0f}s)", flush=True)
# sample weighting: down-weight over-sampled watersheds, weights from TRAINING fold only
class W:
    def __init__(self, power): self.power = power
    def __call__(self):
        outer = self
        class M:
            def fit(s, X, y):
                n = dev.loc[X.index, "huc8"].map(dev.loc[X.index, "huc8"].value_counts())
                s.m = mk().fit(X, y, sample_weight=(1.0 / n ** outer.power).values); return s
            def predict(s, X): return s.m.predict(X)
        return M()
for pw in [0.5, 1.0]:
    for scheme in ["huc8", "state"]:
        s = cv.summarize(cv.cross_val_oof(dev, base, W(pw), scheme, seeds=(0, 1, 2)))
        print(f"{'T2 base, weights 1/n_huc8^'+str(pw):46s} {scheme:6s} R2={s['r2_mean']:.4f}+-{s['r2_sd']:.4f}", flush=True)
