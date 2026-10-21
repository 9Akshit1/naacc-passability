"""Re-derive the headline numbers from the raw data and the saved locked-test predictions.

Nothing here reuses the modelling or evaluation code, so a bug in that code cannot also hide here.
It needs Plunkett's file (see README) and the feature table from src/plunkett_features.py.

    python -m scripts.verify_headline_numbers          # about 2 minutes; refits the random-forest baseline
    python -m scripts.verify_headline_numbers --quick  # skip the refit

Prints PASS or FAIL per check and exits with status 1 if any check fails.
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor

from src import plunkett_data as pdm

QUICK = "--quick" in sys.argv
failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures.append(name)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


raw = pd.read_parquet(pdm.FTP_DIR / "passability_data.parquet")
tag = pd.read_parquet("data/processed/plunkett_tagged.parquet")
feat = pd.read_parquet("data/processed/plunkett_features_nbi.parquet")
lock = json.load(open("results/plunkett/locked_test_definition.json"))
stored = json.load(open("results/plunkett/locked_results.json"))
predvars = [ln.strip() for ln in open(pdm.FTP_DIR / "passability_predvars.txt") if ln.strip()]

# 1. the input file
check("56,525 surveys, one row per crossing code", len(raw) == 56525 and raw.crossing_code.is_unique)
check("target is within 0..1 and never missing", raw.aquatic_score.between(0, 1).all())
survey_fields = {"bridge", "crossing_type", "terrestrial_passage_score", "tsurveyed", "roadtile", "moved", "date_observed",
                 "database", "x_coord", "y_coord", "lat_nad83", "lon_nad83", "aquatic_score", "crossing_code",
                 "crossing_survey_id", "tsurvey_id"}
check("no survey field among the 39 predictors", len(predvars) == 39 and not (survey_fields & set(predvars)))

# 2. the locked test set
mask = tag.huc6.isin(lock["locked_huc6"])
check("locked set is the 7 recorded HUC6 basins, 8,995 crossings", int(mask.sum()) == lock["n_locked"] == 8995)
check("no HUC8 appears on both sides of the split", not (set(tag.huc8[mask]) & set(tag.huc8[~mask])))
x, y_ = Transformer.from_crs(4326, 5070, always_xy=True).transform(tag.lon_nad83.values, tag.lat_nad83.values)
d, _ = cKDTree(np.c_[x[~mask], y_[~mask]]).query(np.c_[x[mask], y_[mask]])
check("no locked crossing within 50 m of a development crossing", d.min() > 50, f"closest {d.min():.0f} m, median {np.median(d) / 1000:.1f} km")

# 3. saved locked predictions reproduce the stored scores
truth = raw.set_index("crossing_code").aquatic_score
tiers = {"T1": "T1_plunkett", "T2": "T2_plunkett+nbi", "T3": "T3_plunkett+nbi+osm"}
preds = {}
for k, name in tiers.items():
    p = pd.read_csv(f"results/plunkett/locked_predictions_{k}.csv").set_index("crossing_code")
    ok_ids = set(p.index) == set(tag.crossing_code[mask])
    y = truth.reindex(p.index).values
    q = p["pred"].values
    preds[k] = p["pred"]
    check(f"{k}: same crossings as the locked set, targets match the raw file", ok_ids and np.allclose(y, p["aquatic_score"].values))
    check(f"{k}: locked R2 {r2(y, q):.4f} equals stored {stored[name]['r2']:.4f}", abs(r2(y, q) - stored[name]["r2"]) < 5e-4)
    check(f"{k}: locked MAE {np.abs(y - q).mean():.4f} equals stored {stored[name]['mae']:.4f}", abs(np.abs(y - q).mean() - stored[name]["mae"]) < 5e-4)

# 4. baseline refit and paired bootstrap
te = feat[mask]
y = te.aquatic_score.values
if not QUICK:
    dev = feat[~mask]
    rf = RandomForestRegressor(n_estimators=300, max_features=0.33, min_samples_leaf=3, n_jobs=-1, random_state=0)
    p_rf = rf.fit(dev[predvars], dev.aquatic_score).predict(te[predvars])
    check(f"random-forest baseline refit: locked R2 {r2(y, p_rf):.4f} vs stored {stored['T1_rf_plunkett_style']['r2']:.4f}",
          abs(r2(y, p_rf) - stored["T1_rf_plunkett_style"]["r2"]) < 3e-3)
    t3 = preds["T3"].reindex(te.crossing_code).values
    groups = te.huc8.values
    ug = np.unique(groups)
    idx = {u: np.where(groups == u)[0] for u in ug}
    rng = np.random.default_rng(123)
    delta = []
    for _ in range(2000):
        s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        delta.append(r2(y[s], t3[s]) - r2(y[s], p_rf[s]))
    lo, hi = np.percentile(delta, [2.5, 97.5])
    check(f"T3 minus RF: {r2(y, t3) - r2(y, p_rf):+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] over {len(ug)} HUC8s", lo > 0)

# 5. NBI join, rebuilt from the raw FHWA files with separate code
frames = []
for f in sorted(glob.glob("data/raw/nbi2025/*25.txt")):
    t = pd.read_csv(f, dtype=str, low_memory=False, usecols=["LAT_016", "LONG_017", "SERVICE_UND_042B"])
    frames.append(t.apply(lambda c: c.str.strip().str.strip("'")))
n = pd.concat(frames)
la, lo_ = n.LAT_016.fillna("").str.zfill(8), n.LONG_017.fillna("").str.zfill(9)
num = lambda s: pd.to_numeric(s, errors="coerce")
la_min, la_sec = num(la.str[2:4]), num(la.str[4:]) / 100
lo_min, lo_sec = num(lo_.str[3:5]), num(lo_.str[5:]) / 100
ok = (la_min < 60) & (la_sec < 60) & (lo_min < 60) & (lo_sec < 60)
n, la, lo_ = n[ok.values], la[ok.values], lo_[ok.values]
lat = num(la.str[:2]) + num(la.str[2:4]) / 60 + num(la.str[4:]) / 360000
lon = -(num(lo_.str[:3]) + num(lo_.str[3:5]) / 60 + num(lo_.str[5:]) / 360000)
keep = n.SERVICE_UND_042B.isin(list("56789")).values & lat.between(36, 48).values & lon.between(-84, -66).values
tr = Transformer.from_crs(4326, 5070, always_xy=True)
cx, cy = tr.transform(feat.lon_nad83.values, feat.lat_nad83.values)
sx, sy = tr.transform(lon.values[keep], lat.values[keep])
dcn, icn = cKDTree(np.c_[sx, sy]).query(np.c_[cx, cy])
_, inc = cKDTree(np.c_[cx, cy]).query(np.c_[sx, sy])
mine = (inc[icn] == np.arange(len(feat))) & (dcn <= 60)
check(f"NBI waterway structures: {keep.sum():,} (repo uses 68,102)", keep.sum() == 68102)
check(f"NBI matches: rebuilt {mine.sum():,}, stored {int(feat.nbi_matched.sum()):,}, identical rows",
      (mine == feat.nbi_matched.values.astype(bool)).all())

# 6. calibration and interval coverage on the locked set
q = preds["T3"].reindex(te.crossing_code).values
check(f"T3 calibration slope of observed on predicted is {np.polyfit(q, y, 1)[0]:.3f} (expect about 1)", abs(np.polyfit(q, y, 1)[0] - 1) < 0.02)
oof = pd.read_parquet("results/plunkett/oof_T3_plunkett+nbi+osm_ensemble.parquet")["oof"].reindex(tag.index[~mask])
half = np.quantile((tag.loc[~mask, "aquatic_score"] - oof).abs().dropna(), 0.9)
cover = ((y >= np.clip(q - half, 0, 1)) & (y <= np.clip(q + half, 0, 1))).mean()
check(f"90% interval (+-{half:.3f}) covers {cover:.3f} of locked crossings", 0.87 < cover < 0.93)

print()
print("all checks passed" if not failures else f"{len(failures)} check(s) failed: {failures}")
sys.exit(1 if failures else 0)
