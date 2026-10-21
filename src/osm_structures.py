"""OpenStreetMap bridge / culvert ways -> per-crossing features (nationally consistent).

Full state OSM extracts download at ~250 KB/s here (hours for 13 states), so instead
we ask the Overpass API only for the tagged structures that matter:

* highway ways with a ``bridge`` tag      (road carried over something)
* highway ways with ``tunnel=culvert``    (road carried over a stream in a culvert)
* waterway ways with ``tunnel=culvert``   (stream passes through a culvert)

Each crossing is matched to the nearest such way within ``MATCH_M`` metres
(point-to-line distance in EPSG:5070). OSM is volunteer-mapped, so completeness
differs by state; that is a real limitation, tested explicitly with
leave-one-state-out evaluation and by checking per-state match rates.
Results are cached per state under ``data/raw/osm/``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString

OSM_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "osm"
HEADERS = {"User-Agent": "NAACC-passability-research/1.0 (student project; aksheru09@gmail.com)"}
ENDPOINTS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
MATCH_M = 25.0

QUERY = """[out:json][timeout:600];
area["ISO3166-2"="US-{st}"]->.a;
(
 way["highway"]["bridge"](area.a);
 way["highway"]["tunnel"="culvert"](area.a);
 way["waterway"]["tunnel"="culvert"](area.a);
);
out tags geom;"""


TILE_QUERY = """[out:json][timeout:600];
(
 way["highway"]["bridge"]({b});
 way["highway"]["tunnel"="culvert"]({b});
 way["waterway"]["tunnel"="culvert"]({b});
);
out tags geom;"""
# States too large for one Overpass request are fetched as bbox tiles (S, W, N, E) and merged.
TILED = {"NY": (40.4, -79.9, 45.1, -71.7), "PA": (39.7, -80.6, 42.3, -74.6)}


_PTS = None   # this state's crossings; set by fetch_state_tiled so empty quadrants are skipped


def _has_crossing(s, w, n, e, pad=0.02) -> bool:
    return _PTS is None or bool(((_PTS.lat_nad83.between(s - pad, n + pad)) & (_PTS.lon_nad83.between(w - pad, e + pad))).any())


def _fetch_tile(st: str, s: float, w: float, n: float, e: float, depth: int = 0) -> list[dict]:
    """One bbox tile; after 3 failed tries split into 4 quadrants (adaptive tiling)."""
    b = f"{s:.4f},{w:.4f},{n:.4f},{e:.4f}"
    tile = OSM_DIR / "tiles" / f"{b.replace(',', '_')}.json"   # shared across states
    tile.parent.mkdir(parents=True, exist_ok=True)
    if tile.exists() and tile.stat().st_size > 50:
        return json.loads(tile.read_text(encoding="utf-8"))["elements"]
    for attempt in range(3):
        try:
            r = requests.post(ENDPOINTS[0], data={"data": TILE_QUERY.format(b=b)}, headers=HEADERS, timeout=400)
            if r.status_code == 200 and r.text.lstrip().startswith("{") and "elements" in r.json():
                tile.write_text(r.text, encoding="utf-8")
                return r.json()["elements"]
            print(f"[{st} {b}] {r.status_code}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[{st} {b}] {repr(ex)[:100]}", flush=True)
        time.sleep(8 * (attempt + 1))
    if depth >= 4:
        raise RuntimeError(f"tile {b} failed at max depth")
    ms, mw = (s + n) / 2, (w + e) / 2
    out = []
    for (a1, b1, c1, d1) in [(s, w, ms, mw), (s, mw, ms, e), (ms, w, n, mw), (ms, mw, n, e)]:
        if _has_crossing(a1, b1, c1, d1):
            out += _fetch_tile(st, a1, b1, c1, d1, depth + 1)
    return out


def fetch_state_tiled(st: str, step: float = 0.8) -> Path:
    f = OSM_DIR / f"{st}_structures.json"
    s0, w0, n0, e0 = TILED[st]
    seen, elements = set(), []
    # only tiles that contain a crossing OF THIS STATE (other states' crossings are covered by their own fetch)
    pts = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "processed" / "plunkett_tagged.parquet",
                          columns=["lat_nad83", "lon_nad83", "state"])
    pts = pts[pts["state"] == st]
    global _PTS
    _PTS = pts
    lat = s0
    while lat < n0:
        lon = w0
        while lon < e0:
            la1, lo1 = min(lat + step, n0), min(lon + step, e0)
            pad = 0.02   # skip tiles with no crossing nearby (ocean, other states): nothing there can match
            if not ((pts.lat_nad83.between(lat - pad, la1 + pad)) & (pts.lon_nad83.between(lon - pad, lo1 + pad))).any():
                lon += step
                continue
            for el in _fetch_tile(st, lat, lon, la1, lo1):
                if el["id"] not in seen:
                    seen.add(el["id"])
                    elements.append(el)
            lon += step
        lat += step
    f.write_text(json.dumps({"elements": elements}), encoding="utf-8")
    return f


def fetch_state(st: str, force: bool = False) -> Path:
    if st in TILED:
        f = OSM_DIR / f"{st}_structures.json"
        return f if (f.exists() and f.stat().st_size > 1000 and not force) else fetch_state_tiled(st)
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    f = OSM_DIR / f"{st}_structures.json"
    if f.exists() and f.stat().st_size > 1000 and not force:   # 0-byte/partial files are not a cache
        return f
    last = None
    for attempt in range(8):
        ep = ENDPOINTS[attempt % len(ENDPOINTS)]
        try:
            r = requests.post(ep, data={"data": QUERY.format(st=st)}, headers=HEADERS, timeout=900)
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                js = r.json()
                if "elements" in js and not js.get("remark", "").startswith("runtime error"):
                    f.write_text(r.text, encoding="utf-8")
                    return f
            last = f"{ep} -> {r.status_code} {r.text[:150]!r}"
        except Exception as e:  # noqa: BLE001
            last = f"{ep} -> {e!r}"[:300].encode("ascii", "replace").decode()
        print(f"[{st}] retry {attempt}: {last}", flush=True)
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Overpass failed for {st}: {last}")


def load_state_structures(st: str) -> gpd.GeoDataFrame:
    f = OSM_DIR / f"{st}_structures.json"
    pbf = OSM_DIR / "pbf" / {"PA": "pennsylvania", "RI": "rhode-island"}.get(st, st.lower()) 
    pbf = pbf.with_suffix(".osm.pbf")
    if not (f.exists() and f.stat().st_size > 1000) and pbf.exists():
        return load_state_structures_pbf(pbf)
    js = json.loads(fetch_state(st).read_text(encoding="utf-8"))
    rows, geoms = [], []
    for e in js["elements"]:
        g = e.get("geometry")
        if not g or len(g) < 2:
            continue
        t = e.get("tags", {})
        is_hw = "highway" in t
        rows.append({
            "osm_kind": "road_bridge" if (is_hw and t.get("bridge", "no") != "no") else
                        ("road_culvert" if is_hw else "stream_culvert"),
            "osm_highway": t.get("highway"),
            "osm_bridge": t.get("bridge"),
            "osm_bridge_structure": t.get("bridge:structure"),
            "osm_layer": t.get("layer"),
            "osm_lanes": t.get("lanes"),
            "osm_surface": t.get("surface"),
            "osm_maxspeed": t.get("maxspeed"),
            "osm_oneway": t.get("oneway"),
            "osm_has_name": float("name" in t),
        })
        geoms.append(LineString([(p["lon"], p["lat"]) for p in g]))
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=4326)


_HW_RANK = {"motorway": 1, "motorway_link": 1, "trunk": 2, "trunk_link": 2, "primary": 3, "primary_link": 3,
            "secondary": 4, "secondary_link": 4, "tertiary": 5, "tertiary_link": 5, "unclassified": 6,
            "residential": 7, "service": 8, "track": 9, "path": 10, "footway": 10, "cycleway": 10}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")


def match_crossings(cross: pd.DataFrame, states: list[str], lat="lat_nad83", lon="lon_nad83",
                    max_m: float = MATCH_M) -> pd.DataFrame:
    """Per-crossing OSM features; unmatched => osm_* NaN and ``osm_bridge_flag`` = 0.

    All crossings are matched against structures from ALL fetched states (state borders
    are not special), keeping the nearest within ``max_m`` for each of the 3 kinds.
    """
    st_frames = [load_state_structures(s) for s in states]
    st = pd.concat(st_frames, ignore_index=True)
    st = gpd.GeoDataFrame(st, geometry="geometry", crs=4326).to_crs(5070)
    pts = gpd.GeoDataFrame(index=cross.index, geometry=gpd.points_from_xy(cross[lon], cross[lat]), crs=4326).to_crs(5070)
    out = pd.DataFrame(index=cross.index)
    for kind in ["road_bridge", "road_culvert", "stream_culvert"]:
        sub = st[st["osm_kind"] == kind].reset_index(drop=True)
        col = f"osm_{kind}_dist_m"
        if sub.empty:
            out[col] = np.nan
            continue
        j = gpd.sjoin_nearest(pts, sub, how="left", max_distance=max_m, distance_col="_d")
        j = j[~j.index.duplicated()]
        out[col] = j["_d"].reindex(cross.index)
        if kind == "road_bridge":
            hw = j["osm_highway"].reindex(cross.index)
            out["osm_bridge_hw_rank"] = hw.map(_HW_RANK)
            out["osm_bridge_lanes"] = _num(j["osm_lanes"].reindex(cross.index).fillna(""))
            out["osm_bridge_maxspeed"] = _num(j["osm_maxspeed"].reindex(cross.index).fillna(""))
            out["osm_bridge_layer"] = pd.to_numeric(j["osm_layer"].reindex(cross.index), errors="coerce")
            out["osm_bridge_is_named"] = j["osm_has_name"].reindex(cross.index)
            bs = j["osm_bridge_structure"].reindex(cross.index)
            out["osm_bridge_is_culvertlike"] = (j["osm_bridge"].reindex(cross.index).isin(["culvert"]) |
                                                bs.isin(["culvert", "box_culvert"])).astype(float)
    out["osm_bridge_flag"] = out["osm_road_bridge_dist_m"].notna().astype(float)
    out["osm_road_culvert_flag"] = out["osm_road_culvert_dist_m"].notna().astype(float)
    out["osm_stream_culvert_flag"] = out["osm_stream_culvert_dist_m"].notna().astype(float)
    return out



# ---------------------------------------------------------------------------
# Alternative source: a Geofabrik state extract (.pbf). Used for Pennsylvania because the
# public Overpass servers kept timing out on its dense tiles. Produces the SAME table as
# load_state_structures(); validated against the Overpass result on Rhode Island
# (see reports/04_multistate_plunkett_results.md).
# ---------------------------------------------------------------------------
import re as _re

PBF_DIR = OSM_DIR / "pbf"
_TAG = _re.compile(r'"([^"]+)"=>"((?:[^"\\]|\\.)*)"')


def load_state_structures_pbf(path) -> gpd.GeoDataFrame:
    from pyogrio import read_dataframe
    sql = ("SELECT osm_id, highway, waterway, other_tags FROM lines WHERE "
           "(highway IS NOT NULL AND other_tags LIKE '%\"bridge\"=>%') OR "
           "((highway IS NOT NULL OR waterway IS NOT NULL) AND other_tags LIKE '%\"tunnel\"=>\"culvert\"%')")
    g = read_dataframe(path, sql=sql, sql_dialect="OGRSQL")
    tags = [dict(_TAG.findall(t or "")) for t in g["other_tags"]]
    rows = []
    for hw, ww, t in zip(g["highway"], g["waterway"], tags):
        is_hw = hw is not None and hw == hw
        rows.append({
            "osm_kind": "road_bridge" if (is_hw and t.get("bridge", "no") != "no") else
                        ("road_culvert" if is_hw else "stream_culvert"),
            "osm_highway": hw if is_hw else None, "osm_bridge": t.get("bridge"),
            "osm_bridge_structure": t.get("bridge:structure"), "osm_layer": t.get("layer"),
            "osm_lanes": t.get("lanes"), "osm_surface": t.get("surface"), "osm_maxspeed": t.get("maxspeed"),
            "osm_oneway": t.get("oneway"), "osm_has_name": float("name" in t)})
    out = gpd.GeoDataFrame(rows, geometry=g.geometry.values, crs=g.crs)
    return out.to_crs(4326)


if __name__ == "__main__":
    import sys
    for s in sys.argv[1:]:
        p = fetch_state(s)
        print(s, p, p.stat().st_size)
