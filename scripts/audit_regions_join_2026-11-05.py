"""Checks on the state matching in the outlet-drop join, and on why about 45% of crossings have no outlet drop.

 1  For every accepted match, does the state in the public layer equal Plunkett's state? Survey ids repeat across state databases.
 2  For crossings without a value, is there a layer point nearby (and if so, is its score different), or is the crossing missing
    from the layer?
 3  Coverage by state and by survey id range. Plunkett's file draws on two databases, the older UMass Stream Continuity Project
    and the 2021 NAACC export.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

K = 111320.0
xy = lambda lat, lon: np.c_[np.asarray(lon) * K * np.cos(np.radians(40.0)), np.asarray(lat) * 110540.0]
d = pd.read_parquet("data/processed/plunkett_features_osm.parquet", columns=["crossing_code", "crossing_survey_id", "state", "aquatic_score", "lat_nad83", "lon_nad83"]).reset_index(drop=True)
j = pd.read_parquet("data/processed/plunkett_outlet_drop.parquet")
d = d.join(j[["drop_ws", "match_pass", "ambiguous"]])
n = pd.read_parquet("data/raw/naacc_public_layer.parquet").reset_index(drop=True)
L = pd.DataFrame({"sid": n.Survey_Id.astype("Int64"), "lat": n["_lat"], "lon": n["_lon"], "sc": pd.to_numeric(n.Aqua_Pass_, errors="coerce"), "st": n.State})
d["sid"] = d.crossing_survey_id.astype("Int64")

print("=== 1. state agreement on accepted id-matches (pass 1)")
m = d[d.match_pass == 1].merge(L, on="sid", suffixes=("", "_l"))
m["km"] = np.hypot((m.lat_nad83 - m.lat) * 110.54, (m.lon_nad83 - m.lon) * 111.32 * np.cos(np.radians(m.lat_nad83)))
m = m[(m.km <= .03) & ((m.aquatic_score - m.sc).abs() < .02)]
print(f" accepted pairs {len(m)}; layer State != Plunkett state: {int((m.st != m.state).sum())}")
print(m[m.st != m.state].groupby(["state", "st"]).size().head(10).to_dict())

print("\n=== 2. crossings without a value: is there a layer point nearby?")
tree = cKDTree(xy(L.lat.dropna(), L.lon.dropna()))
Lc = L.dropna(subset=["lat", "lon"]).reset_index(drop=True)
no = d[d.drop_ws.isna()].copy()
dist, idx = tree.query(xy(no.lat_nad83, no.lon_nad83))
no["near_m"] = dist
no["near_score_diff"] = (no.aquatic_score.values - Lc.sc.values[idx])
no["near_same_state"] = (Lc.st.values[idx] == no.state.values)
print(f" without a value: {len(no)}; layer point within 30 m: {(no.near_m <= 30).mean():.3f}; within 100 m: {(no.near_m <= 100).mean():.3f}; none within 100 m: {(no.near_m > 100).mean():.3f}")
close = no[no.near_m <= 30]
print(f"  of those within 30 m: score differs by >= 0.02: {(close.near_score_diff.abs() >= .02).mean():.3f}; ambiguous-excluded: {close.ambiguous.mean():.3f}")
print(" by state, share of no-value crossings with NO layer point within 100 m:")
print(no.groupby("state").apply(lambda g: pd.Series({"n_no_value": len(g), "absent_from_layer": round((g.near_m > 100).mean(), 2)}), include_groups=False).to_string())

print("\n=== 3. survey-id range (older database ids are small)")
d["has"] = d.drop_ws.notna()
d["idbin"] = pd.cut(d.crossing_survey_id, [0, 20000, 40000, 60000, 80000, 100000, 120000, 1e9])
print(d.groupby("idbin", observed=True).agg(n=("has", "size"), with_value=("has", "mean")).round(3).to_string())
print(d.groupby(["state"]).crossing_survey_id.agg(["min", "median", "max"]).to_string())
print("share of crossings with survey id < 20000 by state:", d.groupby("state").apply(lambda g: round((g.crossing_survey_id < 20000).mean(), 2), include_groups=False).to_dict())
print("done")
