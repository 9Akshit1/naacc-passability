"""Quick-check tier robustness + single locked look. Volunteer error simulated by flipping the perched answer with prob e
(symmetric, random - an assumption, not a measured volunteer error rate). Locked NY crossings scored ONCE at the end."""
import numpy as np, pandas as pd, lightgbm as lgb
from src import plunkett_cv as cv, plunkett_features as pf
d = pd.read_parquet("data/processed/plunkett_features_osm.parquet"); lock = cv.locked_test_mask(d)
old = pd.read_parquet("data/processed/subset_predictors.parquet")
g = pf.feature_groups(d); g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]; base = g["PLUNKETT"] + g["NBI"] + g["OSM"]
drop = pd.to_numeric(old.Outlet_Drop_To_Water_Surface, errors="coerce")
o = pd.DataFrame({"Crossing_Code": old.Crossing_Code, "q_perched": np.where(drop.isna(), np.nan, (drop > 0.25).astype(float))})
a = d.merge(o, left_on="crossing_code", right_on="Crossing_Code"); a["lock"] = lock.reindex(a.index.map(lambda i: i)).values if False else a["crossing_code"].map(dict(zip(d.crossing_code, lock.values)))
dev = a[~a.lock.astype(bool)].reset_index(drop=True); tst = a[a.lock.astype(bool)].reset_index(drop=True)
mk = lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=8)
def flip(s, e, seed):
    r = np.random.default_rng(seed); s = s.copy(); m = r.random(len(s)) < e; s[m & s.notna().values] = 1 - s[m & s.notna().values]; return s
print("dev", len(dev), "locked NY with fields", len(tst))
F = base + ["q_perched"]
for e in (0.0, 0.1, 0.2, 0.3):
    x = dev.copy(); x["q_perched"] = flip(x.q_perched, e, 1)           # train AND test see equally noisy answers
    s = cv.summarize(cv.cross_val_oof(x, F, mk, "huc8", seeds=(0, 1))); print(f"volunteer error {e:.0%}: dev HUC8 R2 = {s['r2_mean']:.4f}", flush=True)
m = mk().fit(dev[base], dev.aquatic_score); p0 = m.predict(tst[base])
m1 = mk().fit(dev[F], dev.aquatic_score); p1 = m1.predict(tst[F])
print("LOCKED NY (once): GIS only", round(cv.metrics(tst.aquatic_score, p0)["r2"], 4), " GIS + perched yes/no", round(cv.metrics(tst.aquatic_score, p1)["r2"], 4),
      " MAE", round(cv.metrics(tst.aquatic_score, p0)["mae"], 3), "->", round(cv.metrics(tst.aquatic_score, p1)["mae"], 3))
