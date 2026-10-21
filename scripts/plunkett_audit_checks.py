"""Independent audit checks (no model fitting). Each prints PASS/FLAG context; results -> results/plunkett/audit_checks.json"""
import json, re
import numpy as np, pandas as pd, geopandas as gpd
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from pyproj import Transformer
from sklearn.metrics import r2_score
from src import plunkett_data as pdm, plunkett_cv as cv, nbi as nbim
R = {}
d = pd.read_parquet("data/processed/plunkett_features_osm.parquet"); lock = cv.locked_test_mask(d)

# ---- A3: which catchment area is right?  Use a label-free physical relationship: structure SIZE ~ drainage area
old = pd.read_parquet("data/processed/subset_predictors.parquet")[["Crossing_Code", "drainage_area_km2", "Stream_Name", "Road"]]
x = d.merge(old, left_on="crossing_code", right_on="Crossing_Code", how="inner")
x = x[(x.nbi_matched == 1) & (x.nbi_length_m > 0) & (x.drainage_area_km2 > 0)]
res = {}
for col in ["nbi_length_m", "nbi_max_span_m", "nbi_deck_width_m"]:
    y = x[x[col] > 0]
    res[col] = {"n": int(len(y)), "rho_with_d8accum": float(spearmanr(y.d8accum, y[col])[0]), "rho_with_old_nldi_area": float(spearmanr(y.drainage_area_km2, y[col])[0])}
R["A3_structure_size_vs_catchment_area"] = res
print("A3 (label-free) Spearman of structure size with catchment area, NY crossings with NBI match:"); [print("  ", k, v) for k, v in res.items()]

# ---- A4: NBI match precision by name agreement (stream & road names), NY overlap
nyraw = pd.read_csv(nbim.NBI_DIR / "NY25.txt", dtype=str, low_memory=False, usecols=["STRUCTURE_NUMBER_008", "FEATURES_DESC_006A", "FACILITY_CARRIED_007", "LAT_016", "LONG_017", "SERVICE_UND_042B"])
for c in nyraw.columns: nyraw[c] = nyraw[c].str.strip().str.strip("'").str.strip()
nyraw = nyraw[nyraw.SERVICE_UND_042B.isin(nbim.WATERWAY_SERVICE)].copy()
nyraw["lat"] = nbim._dms_to_deg(nyraw.LAT_016, 2); nyraw["lon"] = -nbim._dms_to_deg(nyraw.LONG_017, 3)
nyraw = nyraw.dropna(subset=["lat", "lon"]).reset_index(drop=True)
ny = d[d.state == "NY"].merge(old, left_on="crossing_code", right_on="Crossing_Code", how="inner").reset_index(drop=True)
tr = Transformer.from_crs(4326, 5070, always_xy=True)
cx, cy = tr.transform(ny.lon_nad83.values, ny.lat_nad83.values); nx, ny_ = tr.transform(nyraw.lon.values, nyraw.lat.values)
dist, idx = cKDTree(np.c_[nx, ny_]).query(np.c_[cx, cy])
tok = lambda s: set(w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if w not in {"creek", "brook", "river", "rd", "road", "st", "street", "trib", "to", "of", "the", "unnamed", "stream", "branch", "run", "ave", "route", "rt", "ny", "us", "co", "cr", "county", "hwy", "ln", "dr"} and len(w) > 1)
def agree(a, b):
    ta, tb = tok(a), tok(b); return float(bool(ta & tb)) if ta and tb else np.nan
stream_ok = np.array([agree(a, b) for a, b in zip(ny.Stream_Name, nyraw.FEATURES_DESC_006A.values[idx])])
rng = np.random.default_rng(0); perm = rng.permutation(len(ny))
base = np.array([agree(a, b) for a, b in zip(ny.Stream_Name, nyraw.FEATURES_DESC_006A.values[idx][perm])])
bins = [0, 10, 30, 60, 100, 200, 500]
tab = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (dist > lo) & (dist <= hi)
    tab.append({"dist_m": f"{lo}-{hi}", "n": int(m.sum()), "stream_name_agree": float(np.nanmean(stream_ok[m])) if m.sum() else None, "shuffled_baseline": float(np.nanmean(base[m])) if m.sum() else None})
R["A4_nbi_match_name_agreement_NY"] = tab
print("A4 NBI nearest-structure stream-name agreement by distance (NY) vs shuffled-pair baseline:"); [print("  ", t) for t in tab]

# ---- A5: structure built/rebuilt AFTER the survey (NBI is a 2025 snapshot)
yr = pd.to_datetime(d.date_observed).dt.year
m = d.nbi_matched == 1
R["A5_temporal"] = {"matched": int(m.sum()), "built_after_survey": int((m & (d.nbi_year_built > yr)).sum()), "reconstructed_after_survey": int((m & (d.nbi_year_recon > yr)).sum()),
                    "survey_year_median_matched": float(yr[m].median())}
print("A5 temporal:", R["A5_temporal"])

# ---- A6: HUC8 boundary precision (generalised polygons ~100 m)
hu = gpd.read_file(pdm.HUC8_GEOJSON).to_crs(5070)
pts = gpd.GeoDataFrame(index=d.index, geometry=gpd.points_from_xy(d.lon_nad83, d.lat_nad83), crs=4326).to_crs(5070)
bnd = hu.boundary.union_all() if hasattr(hu.boundary, "union_all") else hu.boundary.unary_union
db = pts.distance(bnd)
j = gpd.sjoin(pts.to_crs(4326), gpd.read_file(pdm.HUC8_GEOJSON)[["huc8", "geometry"]], predicate="within")
R["A6_huc8_boundary"] = {"within_150m_of_boundary": int((db < 150).sum()), "share": float((db < 150).mean()), "points_in_multiple_polygons": int(j.index.duplicated().sum()),
                         "points_matched_to_no_polygon": int(len(d) - j.index.nunique())}
print("A6", R["A6_huc8_boundary"])

# ---- A7: locked-set leakage: performance vs distance to nearest development crossing
pr = pd.read_csv("results/plunkett/locked_predictions_T3.csv").set_index("crossing_code")
te = d[lock].set_index("crossing_code").join(pr[["pred"]]); dv = d[~lock]
dd, _ = cKDTree(np.c_[dv._x, dv._y]).query(np.c_[te._x, te._y]); te["dist_km"] = dd / 1000
qs = pd.qcut(te.dist_km, 5, duplicates="drop"); rows = []
for k, gdf in te.groupby(qs, observed=True):
    rows.append({"dist_to_nearest_dev_crossing_km": str(k), "n": len(gdf), "r2": float(r2_score(gdf.aquatic_score, gdf.pred))})
R["A7_locked_r2_by_distance_to_dev"] = rows
print("A7 locked-set R2 by distance to nearest development crossing:"); [print("  ", r) for r in rows]

# ---- A8: crossing_type vs bridge flag; score by type
print("A8 score by crossing type:")
t = d.groupby("crossing_type").aquatic_score.agg(["size", "mean", "std", lambda s: (s >= 0.999).mean()]); t.columns = ["n", "mean", "sd", "share_eq_1"]
print(t.round(3).to_string()); R["A8_by_type"] = t.round(4).reset_index().to_dict("records")
print("  bridge flag by type:", d.groupby("crossing_type").bridge.mean().round(3).to_dict())

# ---- A11/A12: independent R2 and bootstrap re-check
R["A11_sklearn_r2_locked_T3"] = float(r2_score(te.aquatic_score, te.pred)); print("A11 sklearn r2_score on locked T3:", R["A11_sklearn_r2_locked_T3"])
json.dump(R, open("results/plunkett/audit_checks.json", "w"), indent=1, default=str)
