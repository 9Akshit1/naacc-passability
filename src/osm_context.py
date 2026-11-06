"""Road and stream context from OpenStreetMap for each crossing, read from a Geofabrik state extract.

osm_structures.py reads only ways that are tagged as bridges or culverts. This module reads every road and waterway near a
crossing and keeps what might help tell a small pipe on a ditch from a built stream crossing:

- the road: class, whether it is paved, lanes, speed limit, width and whether it is one-way
- the waterway: kind (stream, river, ditch, drain or canal), whether it is intermittent, and whether it is named
- the angle between the road and the waterway where they meet, since a skewed crossing needs a longer pipe

Each crossing is matched to the nearest way within MATCH_M metres in EPSG:5070. The extracts are 50 to 350 MB each, so this was
tried only on the two states that were already downloaded.
"""
from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

MATCH_M = 30.0
_TAG = re.compile(r'"([^"]+)"=>"((?:[^"\\]|\\.)*)"')
ROAD_RANK = {"motorway": 9, "trunk": 8, "primary": 7, "secondary": 6, "tertiary": 5, "unclassified": 4, "residential": 3,
             "service": 2, "track": 1, "motorway_link": 9, "trunk_link": 8, "primary_link": 7, "secondary_link": 6,
             "tertiary_link": 5, "living_street": 3}
PAVED = {"asphalt", "paved", "concrete", "concrete:plates", "concrete:lanes", "paving_stones", "sett", "chipseal"}
UNPAVED = {"unpaved", "gravel", "dirt", "ground", "compacted", "fine_gravel", "grass", "earth", "sand", "mud", "pebblestone"}
WATER_KIND = {"stream": 1, "river": 2, "ditch": 3, "drain": 4, "canal": 5}


def _num(value) -> float:
    m = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    return float(m.group()) if m else np.nan


def read_lines(pbf: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    from pyogrio import read_dataframe
    sql = ("SELECT osm_id, highway, waterway, other_tags FROM lines WHERE "
           "(highway IS NOT NULL OR waterway IS NOT NULL)")
    g = read_dataframe(pbf, sql=sql, sql_dialect="OGRSQL")
    tags = [dict(_TAG.findall(t or "")) for t in g["other_tags"]]
    g["surface"] = [t.get("surface") for t in tags]
    g["lanes"] = [_num(t.get("lanes")) for t in tags]
    g["maxspeed"] = [_num(t.get("maxspeed")) for t in tags]
    g["width"] = [_num(t.get("width")) for t in tags]
    g["oneway"] = [float(t.get("oneway") in ("yes", "1", "true")) for t in tags]
    g["intermittent"] = [float(t.get("intermittent") == "yes") for t in tags]
    g["named"] = [float("name" in t) for t in tags]
    g = g.drop(columns="other_tags").to_crs(5070)
    roads = g[g["highway"].isin(ROAD_RANK)].reset_index(drop=True)
    water = g[g["waterway"].isin(WATER_KIND)].reset_index(drop=True)
    return roads, water


def _bearing(line, point, half=15.0) -> float:
    """Direction of ``line`` (degrees, 0-180) around the point on it nearest to ``point``."""
    at = line.project(point)
    a, b = line.interpolate(max(at - half, 0)), line.interpolate(min(at + half, line.length))
    return float(np.degrees(np.arctan2(b.x - a.x, b.y - a.y)) % 180.0)


def context_for(cross: pd.DataFrame, pbf: Path, lat="lat_nad83", lon="lon_nad83") -> pd.DataFrame:
    roads, water = read_lines(pbf)
    pts = gpd.GeoDataFrame(index=cross.index, geometry=gpd.points_from_xy(cross[lon], cross[lat]), crs=4326).to_crs(5070)
    out = pd.DataFrame(index=cross.index)

    jr = gpd.sjoin_nearest(pts, roads, how="left", max_distance=MATCH_M, distance_col="_d")
    jr = jr[~jr.index.duplicated()].reindex(cross.index)
    out["osm_road_found"] = jr["_d"].notna().astype(float)
    out["osm_road_rank"] = jr["highway"].map(ROAD_RANK)
    out["osm_road_paved"] = jr["surface"].map(lambda s: 1.0 if s in PAVED else (0.0 if s in UNPAVED else np.nan))
    out["osm_road_lanes"] = jr["lanes"]
    out["osm_road_maxspeed"] = jr["maxspeed"]
    out["osm_road_width"] = jr["width"]
    out["osm_road_oneway"] = jr["oneway"]

    jw = gpd.sjoin_nearest(pts, water, how="left", max_distance=MATCH_M, distance_col="_d")
    jw = jw[~jw.index.duplicated()].reindex(cross.index)
    out["osm_water_found"] = jw["_d"].notna().astype(float)
    out["osm_water_kind"] = jw["waterway"].map(WATER_KIND)
    out["osm_water_ditchlike"] = jw["waterway"].isin(["ditch", "drain"]).astype(float).where(jw["_d"].notna())
    out["osm_water_intermittent"] = jw["intermittent"]
    out["osm_water_named"] = jw["named"]

    # skew: angle between road and waterway, 0 = square crossing, 90 = they run nearly parallel
    skew = np.full(len(cross), np.nan)
    ridx, widx = jr["index_right"].values, jw["index_right"].values
    for k, geom in enumerate(pts.geometry.values):
        if np.isnan(ridx[k]) or np.isnan(widx[k]):
            continue
        br = _bearing(roads.geometry.iloc[int(ridx[k])], geom)
        bw = _bearing(water.geometry.iloc[int(widx[k])], geom)
        diff = abs(br - bw)
        diff = min(diff, 180 - diff)            # 0 = parallel, 90 = square
        skew[k] = 90.0 - diff
    out["osm_skew_deg"] = skew
    return out
