"""Check that the PBF-based OSM parser reproduces the Overpass result (Rhode Island), so using it for PA is sound."""
import json
import geopandas as gpd, pandas as pd, numpy as np
from src import osm_structures as o, plunkett_data as pdm
pbf = o.load_state_structures_pbf(o.PBF_DIR / "rhode-island.osm.pbf")
js = json.loads((o.OSM_DIR / "RI_structures.json").read_text(encoding="utf-8"))
print("PBF features by kind:", pbf.osm_kind.value_counts().to_dict())
from shapely.geometry import LineString
rows, geoms = [], []
for e in js["elements"]:
    t = e.get("tags", {}); g = e.get("geometry")
    if not g or len(g) < 2: continue
    is_hw = "highway" in t
    rows.append({"osm_kind": "road_bridge" if (is_hw and t.get("bridge", "no") != "no") else ("road_culvert" if is_hw else "stream_culvert")})
print("Overpass features by kind:", pd.DataFrame(rows).osm_kind.value_counts().to_dict())
# crossing-level agreement: match RI crossings against each source
d = pdm.load(); ri = d[d.state == "RI"].copy()
def flags(st):
    from shapely import STRtree
    s = st.to_crs(5070); pts = gpd.GeoDataFrame(index=ri.index, geometry=gpd.points_from_xy(ri.lon_nad83, ri.lat_nad83), crs=4326).to_crs(5070)
    out = {}
    for k in ["road_bridge", "stream_culvert"]:
        sub = s[s.osm_kind == k]
        j = gpd.sjoin_nearest(pts, sub[["geometry"]], how="left", max_distance=25)
        out[k] = ~j[~j.index.duplicated()].index_right.isna()
    return pd.DataFrame(out)
a, b = flags(pbf), flags(gpd.GeoDataFrame(rows, geometry=[LineString([(p["lon"], p["lat"]) for p in e["geometry"]]) for e in js["elements"] if e.get("geometry") and len(e["geometry"]) >= 2], crs=4326))
for k in a.columns:
    print(f"{k}: PBF flags {a[k].mean():.3f}  Overpass flags {b[k].mean():.3f}  per-crossing agreement {(a[k] == b[k]).mean():.4f}  (n={len(ri)})")
