"""Fetch map-based predictors for road-stream crossings from public web services (the New York version).

DSL builds its predictors from regional raster processing, which this version does not reproduce. Each predictor below is
fetched from a public service and cached under data/interim/cache/, so a run that is interrupted can continue.

    drainage_area_km2, basin_span_km  USGS NLDI upstream basin polygon: its area and the diagonal of its bounding box
    elev_m                            USGS 3DEP point elevation
    terrain_slope_*, terrain_rough_*  slope and roughness from 3DEP elevations on rings 100 m and 500 m around the crossing
    county_income, county_pop_density US Census American Community Survey, by county
    road_class                        nearest OpenStreetMap highway or railway way (see below)
    sc_*                              14 EPA StreamCat measures for the upstream watershed
    stream_slope, stream_order        NHDPlusV2 value added attributes
    aadt, log_aadt                    New York DOT traffic counts, joined to the nearest road segment; the counts cover
                                      state and county roads, so many crossings have no match
    ecoregion                         EPA Level III ecoregion
    structure_*                       New York DOT bridge and large-culvert inventory, joined to the nearest point within 100 m

The road class can be fetched two ways. fetch_osm_road_class asks the Overpass service about each point, which suits a few
crossings but is limited to about one request per second. fetch_osm_road_class_bulk reads a Geofabrik extract of New York once
and answers all the crossings in one nearest-way join. Both write to the same cache and use the same tag classification.

The survey's own road type is kept as road_class_surveyed for comparison and is never a model input, because a crossing that
has not been surveyed does not have it.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

requests.packages.urllib3.disable_warnings()  # noqa

CACHE = Path("data/interim/cache")
CACHE.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "NAACC-passability research (aksheru09@gmail.com)"}
SESSION = requests.Session()
SESSION.headers.update(UA)
SESSION.verify = False


def _get(url, *, params=None, timeout=60, tries=6):
    """GET with exponential backoff on 429 / 5xx / network errors."""
    wait = 20.0
    for attempt in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(wait); wait = min(wait * 2, 300); continue
            return r
        except requests.RequestException:
            time.sleep(wait); wait = min(wait * 2, 300)
    return None


# County census data: income and population density


def fetch_ny_county_census() -> pd.DataFrame:
    """ACS 5-year via the Census Reporter API (no key required):
    median household income (B19013), population (B01003), county land area."""
    cache = CACHE / "ny_county_census.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"county_fips": str})

    r = SESSION.get("https://api.censusreporter.org/1.0/data/show/latest",
                    params={"table_ids": "B19013,B01003", "geo_ids": "050|04000US36"}, timeout=90)
    j = r.json()
    recs = []
    for geoid, tables in j.get("data", {}).items():
        geo = j["geography"].get(geoid, {})
        recs.append({
            "county_fips": geoid.split("US")[-1],
            "county_name": geo.get("name", "").replace(" County, NY", "").replace(", New York", ""),
            "county_income": tables.get("B19013", {}).get("estimate", {}).get("B19013001"),
            "county_population": tables.get("B01003", {}).get("estimate", {}).get("B01003001"),
        })
    out = pd.DataFrame(recs)
    out["release"] = j.get("release", {}).get("name", "ACS 5-year (Census Reporter)")

    # county land area from TIGERweb (no key)
    tg = SESSION.get(
        "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/1/query",
        params={"where": "STATE='36'", "outFields": "GEOID,NAME,AREALAND", "f": "json",
                "returnGeometry": "false"}, timeout=60)
    land = pd.DataFrame([f["attributes"] for f in tg.json()["features"]])
    land["county_fips"] = land["GEOID"].astype(str)
    land["land_area_km2"] = land["AREALAND"] / 1e6
    out = out.merge(land[["county_fips", "land_area_km2"]], on="county_fips", how="left")
    out["county_pop_density"] = out["county_population"] / out["land_area_km2"]
    out.to_csv(cache, index=False)
    return out


def attach_county_census(df: pd.DataFrame) -> pd.DataFrame:
    cen = fetch_ny_county_census()
    key = df["County"].astype("string").str.strip().str.replace(" County", "", regex=False).str.title()
    cen = cen.assign(_k=cen["county_name"].str.strip().str.title())
    m = df.assign(_k=key).merge(cen[["_k", "county_income", "county_pop_density"]],
                                on="_k", how="left").drop(columns="_k")
    return m


# Upstream drainage area from the USGS NLDI basin polygon


def _polygon_area_km2(coords: list[list[float]]) -> float:
    """Spherical polygon area (km^2) from lon/lat rings, via the shoelace formula
    on an equal-area local projection centred on the polygon."""
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    lat0 = math.radians(sum(lats) / len(lats))
    R = 6371.0088
    xs = [math.radians(lo) * R * math.cos(lat0) for lo in lons]
    ys = [math.radians(la) * R for la in lats]
    area = 0.0
    for i in range(len(xs) - 1):
        area += xs[i] * ys[i + 1] - xs[i + 1] * ys[i]
    return abs(area) / 2.0


def fetch_drainage_area(lon: float, lat: float, cid: str) -> dict | None:
    """Return {'comid', 'drainage_area_km2', 'basin_span_km'} or None."""
    cf = CACHE / "nldi_basin"
    cf.mkdir(exist_ok=True)
    f = cf / f"{cid}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except json.JSONDecodeError:
            pass
    # An area this large is almost certainly a snap to a trunk river or to a coastal or tidal segment, not the
    # small stream the crossing is on.
    MAX_PLAUSIBLE_KM2 = 20000.0
    try:
        pos = _get("https://api.water.usgs.gov/nldi/linked-data/comid/position",
                   params={"coords": f"POINT({lon} {lat})", "f": "json"}, timeout=45)
        if pos is None or pos.status_code != 200:
            return None
        comid = int(pos.json()["features"][0]["properties"]["comid"])
        b = _get(f"https://api.water.usgs.gov/nldi/linked-data/comid/{comid}/basin",
                 params={"f": "json"}, timeout=120)
        if b is None or b.status_code != 200:
            return None
        geom = b.json()["features"][0]["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        area = sum(_polygon_area_km2(poly[0]) for poly in polys)
        ext_pts = [pt for poly in polys for pt in poly[0]]
        lats = [p[1] for p in ext_pts]
        lons = [p[0] for p in ext_pts]
        span = _haversine_km(min(lats), min(lons), max(lats), max(lons))
        snap_ok = area <= MAX_PLAUSIBLE_KM2
        out = {"comid": comid,
               "drainage_area_km2": round(area, 4) if snap_ok else None,
               "drainage_area_raw_km2": round(area, 1),
               "basin_span_km": round(span, 3) if snap_ok else None,
               "snap_ok": snap_ok}
        f.write_text(json.dumps(out))
        return out
    except Exception:  # noqa
        return None


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


# Point elevation from USGS 3DEP (EPQS)


def fetch_elevation(lon: float, lat: float, cid: str) -> float | None:
    cf = CACHE / "epqs"
    cf.mkdir(exist_ok=True)
    f = cf / f"{cid}.json"
    if f.exists():
        try:
            return json.loads(f.read_text()).get("elev_m")
        except json.JSONDecodeError:
            pass
    try:
        r = _get("https://epqs.nationalmap.gov/v1/json",
                 params={"x": lon, "y": lat, "units": "Meters", "wkid": 4326}, timeout=45)
        if r is None:
            return None
        v = float(r.json()["value"])
        v = v if -30 < v < 2000 else None
        f.write_text(json.dumps({"elev_m": v}))
        return v
    except Exception:
        return None


# Road class from OpenStreetMap (Overpass API)

# OSM surface tag -> paved/unpaved, for crosswalking into the survey's buckets.
_PAVED_SURFACES = {
    "paved", "asphalt", "concrete", "concrete:plates", "concrete:lanes",
    "paving_stones", "sett", "cobblestone", "metal", "chipseal",
}
_UNPAVED_SURFACES = {
    "unpaved", "gravel", "dirt", "ground", "grass", "sand", "compacted",
    "earth", "fine_gravel", "pebblestone", "mud", "woodchips", "ice", "snow",
}
_TRAIL_HIGHWAYS = {"path", "footway", "bridleway", "cycleway", "steps"}


def _classify_osm_tags(tags: dict) -> str:
    """Crosswalk OSM way tags into the survey's Road_Type category space
    (paved / unpaved / trail / multilane road (>2 lanes) / driveway /
    railroad). Approximate by construction - OSM tagging is inconsistent
    between regions - and reported as such; it is never asked to be exact,
    only to remove the field-survey dependency."""
    if "railway" in tags:
        return "railroad"
    hw = (tags.get("highway") or "").lower()
    surface = (tags.get("surface") or "").lower()
    service = (tags.get("service") or "").lower()
    if hw == "service" and service == "driveway":
        return "driveway"
    if hw in _TRAIL_HIGHWAYS:
        return "trail"
    lanes_raw = tags.get("lanes")
    try:
        lanes = int(float(lanes_raw)) if lanes_raw is not None else None
    except ValueError:
        lanes = None
    if (lanes and lanes > 2) or (hw in {"motorway", "trunk"} and lanes is None):
        return "multilane road (>2 lanes)"
    if hw == "track":
        return "paved" if surface in _PAVED_SURFACES else "unpaved"
    if surface in _UNPAVED_SURFACES:
        return "unpaved"
    if surface in _PAVED_SURFACES:
        return "paved"
    if hw:
        # In this region an OpenStreetMap highway way with no surface tag is almost always a paved road, because
        # unpaved rural roads usually carry a surface tag.
        return "paved"
    return "unknown"


def fetch_osm_road_class(lon: float, lat: float, cid: str, *, radius_m: float = 75.0) -> dict | None:
    """Nearest OSM ``highway``/``railway`` way within ``radius_m`` of the
    crossing point, classified via :func:`_classify_osm_tags`. Returns
    ``{"osm_road_class", "distance_m", "tags"}`` or ``None`` on a fetch
    failure (caller should treat that as "unknown", same as the other
    fetchers in this module).

    Note: Overpass API has strict rate limits (~1 req/sec globally). For large
    batches, the `sleep` parameter in `build_predictor_table` should be >= 1.5s.
    On rate-limit (429) or timeout, returns ``None`` after exponential backoff
    to avoid blocking the caller."""
    cf = CACHE / "osm_road"
    cf.mkdir(exist_ok=True)
    f = cf / f"{cid}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except json.JSONDecodeError:
            pass
    query = (
        f"[out:json][timeout:25];"
        f"(way(around:{radius_m},{lat},{lon})[\"highway\"];"
        f"way(around:{radius_m},{lat},{lon})[\"railway\"];);"
        f"out tags center;"
    )
    try:
        # Try POST first (higher rate limit on Overpass), fall back to GET
        r = SESSION.post("https://overpass-api.de/api/interpreter",
                        data={"data": query}, timeout=45, verify=False)
        if r is None or r.status_code not in (200, 429):
            # If POST fails and not a rate-limit, try GET
            if r is None or r.status_code != 429:
                r = _get("https://overpass-api.de/api/interpreter", params={"data": query}, timeout=45)
        if r is None or r.status_code == 429:
            # Rate-limited; cache "unknown" to avoid re-querying this point
            out = {"osm_road_class": "unknown", "distance_m": None, "tags": {}, "_rate_limited": True}
            f.write_text(json.dumps(out))
            return out
        if r.status_code != 200:
            return None
        elements = r.json().get("elements", [])
        best, best_d = None, None
        for el in elements:
            c = el.get("center")
            if not c:
                continue
            d = _haversine_km(lat, lon, c["lat"], c["lon"]) * 1000.0
            if best_d is None or d < best_d:
                best, best_d = el, d
        if best is None:
            out = {"osm_road_class": "unknown", "distance_m": None, "tags": {}}
        else:
            tags = best.get("tags", {})
            out = {"osm_road_class": _classify_osm_tags(tags), "distance_m": round(best_d, 1), "tags": tags}
        f.write_text(json.dumps(out))
        return out
    except Exception:  # noqa
        return None


# Road class from a local OpenStreetMap extract (see the module docstring for why this exists next to the
# per-point fetcher above)

_OSM_PBF_URL = "https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf"
_OSM_PBF_PATH = CACHE / "_osm_ny_raw.pbf"
_OSM_WAYS_PATH = CACHE / "_osm_ny_ways.parquet"


def _download_osm_pbf() -> Path | None:
    if _OSM_PBF_PATH.exists() and _OSM_PBF_PATH.stat().st_size > 10_000_000:
        return _OSM_PBF_PATH
    try:
        with SESSION.get(_OSM_PBF_URL, timeout=1800, verify=False, stream=True) as r:
            if r.status_code != 200:
                return None
            tmp = _OSM_PBF_PATH.with_suffix(".partial")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            tmp.rename(_OSM_PBF_PATH)
        return _OSM_PBF_PATH
    except Exception:  # noqa
        return None


def _build_osm_way_index():
    """Load (or build once and cache) a GeoDataFrame of every NY
    highway/railway way with the tag columns ``_classify_osm_tags`` needs.
    ~20-25 minutes the first time (parsing a ~497 MB PBF via pyrosm);
    near-instant afterward via the GeoParquet cache. Returns ``None`` if
    ``pyrosm``/``geopandas`` aren't installed or the PBF can't be fetched -
    callers should degrade to "unknown", same as every other fetcher here."""
    try:
        import geopandas as gpd
    except ImportError:
        return None
    if _OSM_WAYS_PATH.exists():
        return gpd.read_parquet(_OSM_WAYS_PATH)
    pbf = _download_osm_pbf()
    if pbf is None:
        return None
    try:
        from pyrosm import OSM
        osm = OSM(str(pbf))
        gdf = osm.get_data_by_custom_criteria(
            custom_filter={"highway": True, "railway": True}, filter_type="keep",
            tags_as_columns=["highway", "railway", "surface", "service", "lanes"],
            keep_nodes=False, keep_ways=True, keep_relations=False,
        )
        keep = gdf[["highway", "railway", "surface", "service", "lanes", "geometry"]].copy()
        keep.to_parquet(_OSM_WAYS_PATH)
        return keep
    except Exception:  # noqa
        return None


def fetch_osm_road_class_bulk(df: pd.DataFrame, *, radius_m: float = 75.0) -> pd.Series:
    """Vectorized nearest-tagged-way lookup for every row of ``df`` (needs
    ``lon``/``lat``/``crossing_id``) via a local spatial index instead of
    Overpass. Returns a Series of ``osm_road_class`` values aligned to
    ``df.index``, sharing the per-crossing disk cache with the per-point
    fetcher (a rebuild resumes correctly regardless of which mechanism
    produced a given crossing's cached value)."""
    import geopandas as gpd

    cf = CACHE / "osm_road"
    cf.mkdir(exist_ok=True)

    cids = df["crossing_id"].astype(str)
    cached: dict[str, str] = {}
    need_mask = pd.Series(False, index=df.index)
    for i, cid in cids.items():
        f = cf / f"{cid}.json"
        if f.exists():
            try:
                cached[cid] = json.loads(f.read_text())["osm_road_class"]
                continue
            except (json.JSONDecodeError, KeyError):
                pass
        need_mask.loc[i] = True

    if need_mask.any():
        ways = _build_osm_way_index()
        sub = df.loc[need_mask]
        pts = gpd.GeoDataFrame(
            {"crossing_id": sub["crossing_id"].astype(str).values},
            geometry=gpd.points_from_xy(sub["lon"], sub["lat"]), crs="epsg:4326")
        by_cid = pd.DataFrame()
        if ways is not None and len(ways):
            # EPSG:5070 (CONUS Albers Equal-Area) so `max_distance` is real metres.
            pts_m = pts.to_crs("epsg:5070")
            ways_m = ways.to_crs("epsg:5070")
            joined = gpd.sjoin_nearest(pts_m, ways_m, max_distance=radius_m,
                                       distance_col="_dist_m", how="left")
            by_cid = joined.drop_duplicates(subset="crossing_id").set_index("crossing_id")
        for cid in sub["crossing_id"].astype(str):
            matched = cid in by_cid.index and pd.notna(by_cid.loc[cid].get("_dist_m"))
            if matched:
                row = by_cid.loc[cid]
                tags = {k: row.get(k) for k in ("highway", "railway", "surface", "service", "lanes")}
                tags = {k: v for k, v in tags.items() if pd.notna(v)}
                out = {"osm_road_class": _classify_osm_tags(tags),
                      "distance_m": round(float(row["_dist_m"]), 1), "tags": tags}
            else:
                out = {"osm_road_class": "unknown", "distance_m": None, "tags": {}}
            cached[cid] = out["osm_road_class"]
            (cf / f"{cid}.json").write_text(json.dumps(out))

    return cids.map(cached).fillna("unknown")


# EPA StreamCat upstream-watershed measures, looked up by COMID in batches. This runs after the drainage-area
# fetch because it needs the COMID.

# Raw StreamCat name -> (feature name, watershed area of interest, group). 'ws' means accumulated over the whole
# upstream watershed and not only the local catchment.
#
# We chose measures with a plausible link to culvert passability instead of taking all of the roughly 600
# available. With only 7 HUC8 groups for spatial CV, a large set of unrelated measures raises the risk of
# overfitting. Land cover fractions are summed into 3 bands (forest, wetland, agriculture) for the same reason.
_STREAMCAT_METRICS = {
    "precip8110":  ("sc_precip_mm", "climate", None),       # 30-yr normal precip - flashiness / design-storm driver
    "runoff":      ("sc_runoff_mm", "climate", None),       # mean runoff
    "bfi":         ("sc_baseflow_idx", "climate", None),    # baseflow index - flow regime character
    "rddens":      ("sc_road_density", "infra", None),      # road density - infra investment / engineering-standard proxy
    "rdcrs":       ("sc_road_stream_crossing_density", "infra", None),  # crossings per area - direct structural analogue
    "damdens":     ("sc_dam_density", "infra", None),       # upstream flow regulation
    "popden2010":  ("sc_pop_density", "infra", None),       # catchment-scale pop density (finer than county-level)
    "rckdep":      ("sc_bedrock_depth_cm", "soils", None),  # depth to bedrock - erosion/incision -> outlet-perching risk
    "perm":        ("sc_soil_perm", "soils", None),         # soil permeability
    "wetindex":    ("sc_wetness_index", "terrain", None),   # composite topographic wetness index - closest terrain proxy StreamCat has (no true slope variable)
    "pctimp2019":  ("sc_pct_impervious", "landcover", None),
    "pctdecid2019":   ("sc_pct_forest", "landcover", "sum"),
    "pctconif2019":   ("sc_pct_forest", "landcover", "sum"),
    "pctmxfst2019":   ("sc_pct_forest", "landcover", "sum"),
    "pctwdwet2019":   ("sc_pct_wetland", "landcover", "sum"),
    "pcthbwet2019":   ("sc_pct_wetland", "landcover", "sum"),
    "pctcrop2019":    ("sc_pct_agriculture", "landcover", "sum"),
    "pcthay2019":     ("sc_pct_agriculture", "landcover", "sum"),
}


# Local-catchment ('cat') versions of some of the measures above. These describe the ground that drains directly
# to the crossing's own reach and not the whole upstream basin. The same request returns both (aoi=ws,cat). They
# are requested only for land cover, infrastructure, soils and terrain, where the local and basin-wide values
# can differ, and not for climate or for population and dam density.
_STREAMCAT_CAT_METRICS = {
    "rddens":       ("sc_road_density_local", None),
    "rdcrs":        ("sc_road_stream_crossing_density_local", None),
    "rckdep":       ("sc_bedrock_depth_cm_local", None),
    "perm":         ("sc_soil_perm_local", None),
    "wetindex":     ("sc_wetness_index_local", None),
    "pctimp2019":   ("sc_pct_impervious_local", None),
    "pctdecid2019": ("sc_pct_forest_local", "sum"),
    "pctconif2019": ("sc_pct_forest_local", "sum"),
    "pctmxfst2019": ("sc_pct_forest_local", "sum"),
    "pctwdwet2019": ("sc_pct_wetland_local", "sum"),
    "pcthbwet2019": ("sc_pct_wetland_local", "sum"),
    "pctcrop2019":  ("sc_pct_agriculture_local", "sum"),
    "pcthay2019":   ("sc_pct_agriculture_local", "sum"),
}


def _aggregate_streamcat_cat_record(rec: dict) -> dict:
    """Same aggregation as ``_aggregate_streamcat_record``, but for the local-catchment measures in
    ``_STREAMCAT_CAT_METRICS``, reading the ``...cat`` fields instead of ``...ws``. It also carries ``catareasqkm``
    (StreamCat's own local-catchment area) as ``sc_local_catchment_area_km2``. This differs from
    ``sc_watershed_area_km2`` (the whole upstream basin) and shows how much of the total drainage is right here."""
    out = {"sc_local_catchment_area_km2": rec.get("catareasqkm")}
    sums = {}
    for raw_name, (feat_name, agg) in _STREAMCAT_CAT_METRICS.items():
        v = rec.get(f"{raw_name}cat")
        if agg == "sum":
            have, total = sums.get(feat_name, (False, 0.0))
            sums[feat_name] = (have or v is not None, total + (v or 0.0))
        else:
            out[feat_name] = v
    for feat_name, (have, total) in sums.items():
        out[feat_name] = total if have else np.nan
    return out


def _aggregate_streamcat_record(rec: dict) -> dict:
    """Pure function: raw StreamCat API record (``{"precip8110ws": ..., ...}``)
    -> engineered ``sc_*`` feature dict, per ``_STREAMCAT_METRICS``. Land-cover
    sub-classes are summed into composite bands; a composite is only NaN if
    every one of its sub-classes is missing (a missing sub-class alongside
    present ones is treated as 0%, not as "whole sum unknown"). Also carries
    through ``wsareasqkm`` - StreamCat's own independent upstream-watershed
    area, requested via ``showareasqkm=true`` - as ``sc_watershed_area_km2``,
    an independent cross-check/backfill for the primary NLDI-derived
    ``drainage_area_km2`` (a different data source computing the same
    physical quantity - the closer the two agree, the more confidence in
    both; where NLDI's basin-polygon fetch fails, this fills the gap)."""
    out = {"sc_watershed_area_km2": rec.get("wsareasqkm")}
    sums = {}
    for raw_name, (feat_name, _group, agg) in _STREAMCAT_METRICS.items():
        v = rec.get(f"{raw_name}ws")
        if agg == "sum":
            have, total = sums.get(feat_name, (False, 0.0))
            sums[feat_name] = (have or v is not None, total + (v or 0.0))
        else:
            out[feat_name] = v
    for feat_name, (have, total) in sums.items():
        out[feat_name] = total if have else np.nan
    return out


def fetch_streamcat_batch(comids: list, *, chunk_size: int = 300) -> pd.DataFrame:
    """Look up StreamCat measures for a list of COMIDs in batches.

    Each COMID gets its own cache file, like the other fetchers in this module, but requests go out in large chunks. The
    StreamCat API has no per-request rate limit, so this is far faster than the per-row Overpass fetcher. Each request asks
    for both ``aoi=ws`` (whole upstream watershed) and ``aoi=cat`` (local catchment), and the API returns both together.
    Returns a DataFrame indexed by ``comid`` with the ``sc_*`` (watershed) columns and the ``sc_*_local`` and
    ``sc_local_catchment_area_km2`` (local catchment) columns."""
    cf = CACHE / "streamcat"
    cf.mkdir(exist_ok=True)
    comids = sorted({int(c) for c in comids if pd.notna(c)})
    raw_names = sorted(set(_STREAMCAT_METRICS) | set(_STREAMCAT_CAT_METRICS))

    cached, missing = {}, []
    for c in comids:
        f = cf / f"{c}.json"
        if f.exists():
            try:
                rec = json.loads(f.read_text())
                # Cache files from before the local-catchment (aoi=cat) measures were added have no '...cat'
                # keys, so they are fetched again. 'catareasqkm' alone does not show this, because it comes from
                # showareasqkm=true whatever the aoi.
                if not rec or any(f"{n}cat" in rec for n in _STREAMCAT_CAT_METRICS):
                    cached[c] = rec
                    continue
            except json.JSONDecodeError:
                pass
        missing.append(c)

    for i in range(0, len(missing), chunk_size):
        chunk = missing[i:i + chunk_size]
        try:
            r = SESSION.post("https://api.epa.gov/StreamCat/streams/metrics",
                            headers={"Content-Type": "application/x-www-form-urlencoded"},
                            data={"comid": ",".join(str(c) for c in chunk),
                                  "aoi": "ws,cat", "name": ",".join(raw_names),
                                  "showareasqkm": "true"},
                            timeout=90, verify=False)
            items = r.json().get("items", []) if r is not None and r.status_code == 200 else []
        except Exception:  # noqa
            items = []
        by_comid = {int(it["comid"]): it for it in items if "comid" in it}
        for c in chunk:
            rec = by_comid.get(c, {})
            cached[c] = rec
            (cf / f"{c}.json").write_text(json.dumps(rec))

    rows = []
    for c in comids:
        out = {"comid": c}
        rec = cached.get(c, {})
        out.update(_aggregate_streamcat_record(rec))
        out.update(_aggregate_streamcat_cat_record(rec))
        rows.append(out)
    return pd.DataFrame(rows).set_index("comid")


# NHDPlusV2 value added attributes: stream slope and Strahler order, looked up by COMID. DSL names stream
# gradient as important and StreamCat has no slope. The 246 MB CONUS-wide file is downloaded once and reduced to
# a small per-COMID cache. The raw file is not committed (see .gitignore) and the cache is.

_VAA_URL = ("https://www.hydroshare.org/resource/"
           "6092c8a62fac45be97a09bfd0b0bf726/data/contents/nhdplusVAA.parquet")
_VAA_RAW_PATH = CACHE / "_nhdplus_vaa_raw.parquet"


def _download_nhdplus_vaa_raw() -> Path | None:
    if _VAA_RAW_PATH.exists() and _VAA_RAW_PATH.stat().st_size > 1_000_000:
        return _VAA_RAW_PATH
    try:
        with SESSION.get(_VAA_URL, timeout=600, verify=False, stream=True) as r:
            if r.status_code != 200:
                return None
            tmp = _VAA_RAW_PATH.with_suffix(".partial")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            tmp.rename(_VAA_RAW_PATH)
        return _VAA_RAW_PATH
    except Exception:  # noqa
        return None


def _clean_vaa_slope(v) -> float | None:
    """NHDPlus VAA uses -9998 for "no data" in ``slope`` (real slopes are never negative). This affects about 0.9% of
    CONUS flowlines. It is treated as missing and not as a value, because passing -9998 to the model would do far more
    damage than a NaN, which the median imputer handles."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or v < 0:
        return None
    return float(v)


def fetch_nhdplus_vaa_batch(comids: list) -> pd.DataFrame:
    """Batched NHDPlusV2 VAA lookup for a list of COMIDs: ``stream_slope``
    (dimensionless, m/m) and ``stream_order`` (Strahler). Caches one JSON file
    per COMID; only downloads the 246 MB source parquet once, and only if any
    requested COMID is not already cached."""
    cf = CACHE / "nhdplus_vaa"
    cf.mkdir(exist_ok=True)
    comids = sorted({int(c) for c in comids if pd.notna(c)})

    cached, missing = {}, []
    for c in comids:
        f = cf / f"{c}.json"
        if f.exists():
            try:
                cached[c] = json.loads(f.read_text())
                continue
            except json.JSONDecodeError:
                pass
        missing.append(c)

    if missing:
        raw = _download_nhdplus_vaa_raw()
        if raw is not None:
            import pyarrow.parquet as pq
            tbl = pq.read_table(str(raw), columns=["comid", "slope", "streamorde"],
                                filters=[("comid", "in", missing)])
            vaa = tbl.to_pandas()
            by_comid = {int(row.comid): {"stream_slope": _clean_vaa_slope(row.slope),
                                         "stream_order": int(row.streamorde) if pd.notna(row.streamorde) else None}
                       for row in vaa.itertuples()}
        else:
            by_comid = {}
        for c in missing:
            rec = by_comid.get(c, {"stream_slope": None, "stream_order": None})
            cached[c] = rec
            (cf / f"{c}.json").write_text(json.dumps(rec))

    rows = [{"comid": c, **cached.get(c, {"stream_slope": None, "stream_order": None})} for c in comids]
    return pd.DataFrame(rows).set_index("comid")


# New York DOT annual average daily traffic (AADT). The source is the public statewide ArcGIS FeatureServer
# https://gis.dot.ny.gov/hostingny/rest/services/Roadways/Traffic_Monitoring/FeatureServer/1 (layer 1, field
# AADT, with a value on 157,331 of 174,842 features). It is downloaded once in pages of 2,000 (the service's
# maxRecordCount), cached as GeoParquet, and joined to crossings the way road_class is, with geopandas
# sjoin_nearest in EPSG:5070 so that max_distance is in metres. There is no per-crossing cache, because the join
# is vectorised and has no rate limit.

_AADT_QUERY_URL = "https://gis.dot.ny.gov/hostingny/rest/services/Roadways/Traffic_Monitoring/FeatureServer/1/query"
_AADT_PATH = CACHE / "_aadt_ny_raw.parquet"


def _download_aadt_layer():
    """Paginated bulk download of the statewide AADT polyline layer. Returns
    a GeoDataFrame (cached as GeoParquet after the first call) or ``None`` if
    geopandas isn't available or the service is unreachable."""
    try:
        import geopandas as gpd
    except ImportError:
        return None
    if _AADT_PATH.exists():
        return gpd.read_parquet(_AADT_PATH)
    offset = 0
    frames = []
    while True:
        r = _get(_AADT_QUERY_URL, params={
            "where": "AADT IS NOT NULL",
            "outFields": "AADT,CalculationYear,FunctionalClass,RoadwayName",
            "outSR": 4326, "f": "geojson",
            "resultOffset": offset, "resultRecordCount": 2000,
        }, timeout=120)
        if r is None or r.status_code != 200:
            return None if not frames else pd.concat(frames, ignore_index=True)
        j = r.json()
        feats = j.get("features", [])
        if not feats:
            break
        frames.append(gpd.GeoDataFrame.from_features(feats, crs="epsg:4326"))
        if len(feats) < 2000:
            break
        offset += 2000
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out = out[out.geometry.notna() & out["AADT"].notna()].reset_index(drop=True)
    out.to_parquet(_AADT_PATH)
    return out


def fetch_aadt_bulk(df: pd.DataFrame, *, radius_m: float = 200.0) -> pd.Series:
    """Vectorized nearest-AADT-segment lookup for every row of ``df`` (needs
    ``lon``/``lat``). Returns a Series of raw AADT values aligned to
    ``df.index`` (NaN where nothing is within ``radius_m`` - true for most
    minor/local roads and driveways, since NYSDOT's count program covers
    primarily the state highway system, not every local street)."""
    import geopandas as gpd

    aadt = _download_aadt_layer()
    if aadt is None or len(aadt) == 0:
        return pd.Series(np.nan, index=df.index)
    pts = gpd.GeoDataFrame(
        {"_idx": df.index}, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="epsg:4326")
    pts_m = pts.to_crs("epsg:5070")
    aadt_m = aadt[["AADT", "geometry"]].to_crs("epsg:5070")
    joined = gpd.sjoin_nearest(pts_m, aadt_m, max_distance=radius_m,
                               distance_col="_dist_m", how="left")
    joined = joined.drop_duplicates(subset="_idx").set_index("_idx")
    return pd.to_numeric(joined["AADT"], errors="coerce").reindex(df.index)


# Local terrain slope and roughness from the USGS 3DEP ImageServer getSamples operation. It takes a POST of at
# least 500 points and returns elevation at the best available resolution, down to 1 m where lidar exists. For
# each crossing, 4 points are sampled at each of two radii (100 m and 500 m). Slope is the central-difference
# gradient between opposite points, and roughness is the standard deviation of the elevations on the ring.
# Points with no coverage (open water, outside the state) are dropped by the service, so a crossing can have a
# partial or missing slope or roughness at a radius. That is handled like any other missing value.

_TERRAIN_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/getSamples"
_TERRAIN_RADII_M = (100.0, 500.0)


def _offset_latlon(lat: float, lon: float, dx_m: float, dy_m: float) -> tuple[float, float]:
    """Offset (lat, lon) by (dx_m east, dy_m north) using the standard
    spherical-earth metres-per-degree approximation (111,320 m/deg latitude;
    111,320*cos(lat) m/deg longitude) - accurate to a small fraction of a
    percent at these scales; not a fabricated constant, the same
    approximation used throughout GIS for small local offsets."""
    dlat = dy_m / 111_320.0
    denom = 111_320.0 * math.cos(math.radians(lat))
    dlon = dx_m / denom if abs(denom) > 1e-6 else 0.0
    return lat + dlat, lon + dlon


def fetch_terrain_features_bulk(df: pd.DataFrame, *, batch_size: int = 500) -> pd.DataFrame:
    """For every row of ``df`` (``lon``/``lat``/``crossing_id``), sample a
    compass ring of points at ``_TERRAIN_RADII_M`` via the bulk 3DEP
    ``getSamples`` endpoint and derive per-radius slope (m/m) and roughness
    (elevation std across the ring, m). Cached per-crossing like every other
    fetcher here. Returns a DataFrame aligned to ``df.index`` with columns
    ``terrain_slope_{r}m`` / ``terrain_rough_{r}m`` for each radius."""
    cf = CACHE / "terrain"
    cf.mkdir(exist_ok=True)
    cols = ([f"terrain_slope_{int(r)}m" for r in _TERRAIN_RADII_M]
           + [f"terrain_rough_{int(r)}m" for r in _TERRAIN_RADII_M])

    cids = df["crossing_id"].astype(str)
    cached: dict[str, dict] = {}
    need = []
    for i, cid in cids.items():
        f = cf / f"{cid}.json"
        if f.exists():
            try:
                cached[cid] = json.loads(f.read_text())
                continue
            except json.JSONDecodeError:
                pass
        need.append((cid, float(df.loc[i, "lon"]), float(df.loc[i, "lat"])))

    # One flat list of (crossing id, radius, direction) and the matching (lon, lat) points for every crossing
    # that needs a fetch, so that one batched pass covers all of them.
    requests_meta: list[tuple[str, float, str]] = []
    points: list[tuple[float, float]] = []
    for cid, lon, lat in need:
        for r in _TERRAIN_RADII_M:
            for dname, (dx, dy) in (("N", (0.0, r)), ("S", (0.0, -r)),
                                    ("E", (r, 0.0)), ("W", (-r, 0.0))):
                olat, olon = _offset_latlon(lat, lon, dx, dy)
                requests_meta.append((cid, r, dname))
                points.append((olon, olat))

    elevs: dict[int, float] = {}
    for start in range(0, len(points), batch_size):
        chunk = points[start:start + batch_size]
        geom = json.dumps({"points": [[lo, la] for lo, la in chunk],
                           "spatialReference": {"wkid": 4326}})
        # A call can fail once and work on a retry (one batch returned zero samples during development). Without
        # a retry that would blank a whole batch of crossings, so this retries with backoff like the other
        # network calls in this module.
        samples = []
        wait = 5.0
        for attempt in range(4):
            try:
                r = SESSION.post(_TERRAIN_URL, data={
                    "geometryType": "esriGeometryMultipoint", "geometry": geom, "f": "json",
                }, timeout=120)
                if r is not None and r.status_code == 200:
                    samples = r.json().get("samples", [])
                    if samples:
                        break
            except Exception:  # noqa
                pass
            time.sleep(wait); wait = min(wait * 2, 60)
        for s in samples:
            try:
                elevs[start + int(s["locationId"])] = float(s["value"])
            except (KeyError, ValueError, TypeError):
                pass

    by_cid: dict[str, dict[float, dict[str, float]]] = {}
    for idx, (cid, r, dname) in enumerate(requests_meta):
        v = elevs.get(idx)
        if v is not None:
            by_cid.setdefault(cid, {}).setdefault(r, {})[dname] = v

    for cid, _lon, _lat in need:
        out = {}
        rings = by_cid.get(cid, {})
        for r in _TERRAIN_RADII_M:
            ring = rings.get(r, {})
            slope_ns = abs(ring["N"] - ring["S"]) / (2 * r) if "N" in ring and "S" in ring else None
            slope_ew = abs(ring["E"] - ring["W"]) / (2 * r) if "E" in ring and "W" in ring else None
            if slope_ns is not None and slope_ew is not None:
                slope = math.hypot(slope_ns, slope_ew)
            else:
                slope = slope_ns if slope_ns is not None else slope_ew
            vals = list(ring.values())
            rough = float(np.std(vals)) if len(vals) >= 2 else None
            out[f"terrain_slope_{int(r)}m"] = slope
            out[f"terrain_rough_{int(r)}m"] = rough
        cached[cid] = out
        (cf / f"{cid}.json").write_text(json.dumps(out))

    rows = [{c: cached.get(cid, {}).get(c) for c in cols} for cid in cids]
    return pd.DataFrame(rows, index=df.index)


# EPA Level III ecoregion, so that the model can account for regional differences. Spatial (HUC8 hold-out) CV
# scored much worse than random CV, and the model using drainage area alone had R2 below 0, which fits the
# relationship between landscape and passability differing by region.
#
# The source is the public EPA ArcGIS MapServer
# https://geodata.epa.gov/arcgis/rest/services/ORD/USEPA_Ecoregions_Level_III_and_IV/MapServer/11 (field
# US_L3NAME). 11 ecoregions intersect New York's bounding box. The layer's own STATE_NAME filter finds only 4 of
# them, so a geometry intersect with the bounding box is used instead. It is downloaded once, cached as
# GeoParquet, and joined by point-in-polygon.

_ECOREGION_QUERY_URL = ("https://geodata.epa.gov/arcgis/rest/services/ORD/"
                       "USEPA_Ecoregions_Level_III_and_IV/MapServer/11/query")
_ECOREGION_PATH = CACHE / "_ecoregion_ny_raw.parquet"
_NY_BBOX = (-79.8, 40.4, -71.8, 45.1)  # lon_min, lat_min, lon_max, lat_max - covers NY with margin


def _download_ecoregion_layer():
    try:
        import geopandas as gpd
    except ImportError:
        return None
    if _ECOREGION_PATH.exists():
        return gpd.read_parquet(_ECOREGION_PATH)
    r = _get(_ECOREGION_QUERY_URL, params={
        "geometry": ",".join(str(x) for x in _NY_BBOX),
        "geometryType": "esriGeometryEnvelope", "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "US_L3NAME,US_L3CODE", "outSR": 4326, "f": "geojson",
    }, timeout=120)
    if r is None or r.status_code != 200:
        return None
    feats = r.json().get("features", [])
    if not feats:
        return None
    gdf = gpd.GeoDataFrame.from_features(feats, crs="epsg:4326")
    gdf = gdf[gdf.geometry.notna()].reset_index(drop=True)
    gdf.to_parquet(_ECOREGION_PATH)
    return gdf


def fetch_ecoregion_bulk(df: pd.DataFrame) -> pd.Series:
    """Point-in-polygon join to the EPA Level III ecoregion each crossing
    falls in. Returns a Series of ``US_L3NAME`` values (``"unknown"`` where
    the point falls outside every downloaded polygon, e.g. right at the
    bbox edge, or if the service/geopandas is unavailable)."""
    import geopandas as gpd

    eco = _download_ecoregion_layer()
    if eco is None or len(eco) == 0:
        return pd.Series("unknown", index=df.index)
    pts = gpd.GeoDataFrame(
        {"_idx": df.index}, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="epsg:4326")
    joined = gpd.sjoin(pts, eco[["US_L3NAME", "geometry"]], how="left", predicate="within")
    joined = joined.drop_duplicates(subset="_idx").set_index("_idx")
    return joined["US_L3NAME"].reindex(df.index).fillna("unknown")


# New York DOT structure inventory (bridges and large culverts). NAACC's own score is built from measured
# structure properties such as outlet drop, structure type and constriction, so actual structure data (age,
# material, condition rating, geometry) is closer to the score than any landscape predictor.
#
# The source is the ArcGIS FeatureServer
# https://gis.dot.ny.gov/hostingny/rest/services/Asset/NYSDOT_Structures/FeatureServer. Layer 0 ('NYSDOT
# Bridges') has 20,051 features with a condition rating from 1 to 7, where higher is better. Layer 1 ('NYSDOT
# Large Culverts') has 7,985 features with the same schema. Coverage is well below 100%, because this is the
# state-maintained inventory and not every small local culvert (build_predictor_table prints the match rate).
# Both layers are downloaded once, put into one schema, and joined by nearest point within radius_m, as for
# road_class and aadt.

_STRUCTURES_URL = "https://gis.dot.ny.gov/hostingny/rest/services/Asset/NYSDOT_Structures/FeatureServer"
_STRUCTURES_PATH = CACHE / "_structures_ny_raw.parquet"
_STRUCTURES_CURRENT_YEAR = 2026  # used to compute structure_age_yr


def _download_structures_layer():
    """Download the NYSDOT Bridges (layer 0) and Large Culverts (layer 1) and put them in one GeoDataFrame with a common
    schema: ``YearBuilt``, ``ConditionRating``, ``GTMSMaterial``, ``length_ft``, ``is_bridge``, ``NBI_SuperstructureCondition``
    and ``FHWA_Condition_Status``. The last two exist only for bridges. In the Bridges layer ``NBI_DeckCondition`` and
    ``NBI_SubstructureCondition`` are empty on all 20,051 features and are not used. ``NBI_SuperstructureCondition`` is filled
    on 19,920 (about 99.3%) and ``FHWA_Condition_Status`` on all 20,051. They correlate only 0.75 with ``ConditionRating``, so
    they add something. Large culverts have neither field, so both are missing for culvert matches.

    The culvert opening width is ``width_ft``, NYSDOT's ``SpanLength``. NYSDOT's Culvert Inventory and Inspection Manual
    defines it as the distance perpendicular to the centreline of the culvert (across the waterway, not along it), and says
    a structure with a total span over 20 ft is classified as a bridge. A sample of 2,000 rows agrees: the field is always
    filled and ranges from 2 to 22 ft. We first tried ``OutToOutWidth`` and dropped it, because the manual ties it to
    headwalls and not to the waterway opening, and it correlates only 0.04 with ``SpanLength``. We also tried a width for
    bridges as ``DeckAreaSqFt / BridgeLengthft`` and dropped it, because 207 of 20,051 bridges gave impossible values (up to
    4,230 ft): ``BridgeLengthft`` runs along the road, and ``DeckAreaSqFt`` includes ramps and medians on skewed or
    interchange-style bridges. The manual confirms that ``BridgeLengthft`` (already used as ``structure_length_m``) is the
    axis that spans the channel for bridges, unlike a culvert's ``StructureLength``, which runs along the culvert.
    ``fetch_bankfull_width_bulk`` handles that difference in the constriction ratio.

    Other fields kept: ``num_spans`` (both types, ``NumberOfSpans``, always filled), ``streambed_material`` (culverts only,
    NYSDOT's natural-bottom continuity field, which matches a NAACC scoring criterion) and ``culvert_skew_deg`` (culverts
    only, ``CulvertSkew``). In a sample, 51.5% of skews read exactly 0. Unlike the width, a skew of 0 is a real and common
    value (a perpendicular crossing), so it is not treated as a placeholder. Two fields were considered and not used:
    ``GeneralRecommendation`` correlates 0.72 with ``ConditionRating`` in a 2,000-row sample, so it mostly repeats it, and
    ``AbutmentHeight`` is filled on only 193 of 2,000 culverts, which is under 2% of all crossings once it is combined with
    the match rate of about 20 to 25%."""
    try:
        import geopandas as gpd
    except ImportError:
        return None
    if _STRUCTURES_PATH.exists():
        return gpd.read_parquet(_STRUCTURES_PATH)

    def _fetch_layer(layer_id, length_field, is_bridge, extra_fields=""):
        offset = 0
        frames = []
        fields = f"YearBuilt,ConditionRating,GTMSMaterial,{length_field}{extra_fields}"
        while True:
            r = _get(f"{_STRUCTURES_URL}/{layer_id}/query", params={
                "where": "1=1", "outFields": fields, "outSR": 4326, "f": "geojson",
                "resultOffset": offset, "resultRecordCount": 2000,
            }, timeout=90)
            if r is None or r.status_code != 200:
                break
            feats = r.json().get("features", [])
            if not feats:
                break
            gdf = gpd.GeoDataFrame.from_features(feats, crs="epsg:4326")
            gdf = gdf.rename(columns={length_field: "length_ft"})
            gdf["is_bridge"] = is_bridge
            frames.append(gdf)
            if len(feats) < 2000:
                break
            offset += 2000
        return pd.concat(frames, ignore_index=True) if frames else None

    bridges = _fetch_layer(0, "BridgeLengthft", True,
                           extra_fields=",NBI_SuperstructureCondition,FHWA_Condition_Status,"
                                        "NumberOfSpans,GTMSStructure,LastInspectionDate")
    culverts = _fetch_layer(1, "StructureLength", False,
                            extra_fields=",NumberOfSpans,SpanLength,CulvertSkew,StreamBedMaterial,"
                                         "TypeMaxSpanDesign,LastInspectionDate")
    parts = [p for p in (bridges, culverts) if p is not None]
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs="epsg:4326")
    out = out[out.geometry.notna()].reset_index(drop=True)

    is_b = out["is_bridge"] == True  # noqa: E712
    span_length = pd.to_numeric(out.get("SpanLength"), errors="coerce")
    width_ft = pd.Series(np.nan, index=out.index)
    width_ft[~is_b] = span_length[~is_b]
    out["width_ft"] = width_ft
    out["num_spans"] = pd.to_numeric(out.get("NumberOfSpans"), errors="coerce")

    # Structure type and shape. Bridges and culverts use different NYSDOT fields for this (GTMSStructure and
    # TypeMaxSpanDesign, found with a schema query). Both are coded as 'NN - Description' strings such as '40 -
    # Single Box Culvert'. Dropping the code leaves a shape category (box, pipe, arch, slab, frame, girder,
    # truss and so on) that relates to NAACC's structure-shape criterion. It is a different axis from
    # structure_material.
    def _strip_code(s):
        if not isinstance(s, str):
            return None
        return s.split(" - ", 1)[1].strip() if " - " in s else s.strip()
    struct_type = pd.Series(np.nan, index=out.index, dtype="object")
    struct_type[is_b] = out.loc[is_b, "GTMSStructure"].map(_strip_code) if "GTMSStructure" in out else None
    struct_type[~is_b] = (out.loc[~is_b, "TypeMaxSpanDesign"].map(_strip_code)
                          if "TypeMaxSpanDesign" in out else None)
    out["structure_type"] = struct_type

    # Years since last inspection. LastInspectionDate is an epoch-millisecond timestamp that is filled for about
    # 99.9% of structures. It differs from structure_age_yr, which is the time since the structure was built.
    #
    # NYSDOT's culvert inventory manual describes a biennial inspection cycle, and 92.8% of matched structures
    # were inspected 0 to 4 years ago, which fits. A second, much denser cluster appears at 40 to 48 years (13
    # to 759 structures per year), almost all with a LastInspectionDate in 1978 or 1979. The manual gives that
    # period as the change from the legacy inventory system. Thousands of structures going 40 years without
    # inspection under a two-year requirement is not plausible, so these are most likely placeholder dates. That
    # is an inference and not something NYSDOT states. Values of 40 or more are therefore set to missing, as the
    # ConditionRating == 0 placeholder was.
    insp_ms = pd.to_numeric(out.get("LastInspectionDate"), errors="coerce")
    insp_year = pd.to_datetime(insp_ms, unit="ms", errors="coerce").dt.year
    yrs_insp = (_STRUCTURES_CURRENT_YEAR - insp_year).clip(lower=0)
    out["yrs_since_inspection"] = yrs_insp.where(yrs_insp < 40)

    out.to_parquet(_STRUCTURES_PATH)
    return out


def _clean_structure_condition(v) -> float | None:
    """NYSDOT's condition rating runs from 1 to 7 (7 is best). A value of 0 (66 of 4,492 matches, with a clean gap to the
    next value at 1.0) means "not yet rated" and is not a real condition, so it becomes missing. NHDPlus VAA's -9998 slope
    is handled the same way in ``_clean_vaa_slope``."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or v <= 0:
        return None
    return float(v)


def fetch_structure_inventory_bulk(df: pd.DataFrame, *, radius_m: float = 100.0) -> pd.DataFrame:
    """For every row of ``df``, find the nearest NYSDOT bridge or large culvert within ``radius_m`` and return:

    ``structure_age_yr``, ``structure_condition`` (1 to 7, higher is better), ``structure_material`` (categorical),
    ``structure_length_m``, ``structure_is_bridge`` ("bridge", "culvert" or "none"), ``structure_superstructure_condition``
    (NBI 0 to 9 scale, bridges only), ``structure_fhwa_status`` (Good, Fair or Poor, bridges only),
    ``structure_opening_width_m``, ``structure_num_spans``, ``structure_streambed_material`` (culverts only),
    ``structure_culvert_skew_deg`` (culverts only), ``structure_type`` (shape: box, pipe, arch, slab, frame, girder, truss
    and so on, or "none" if there is no match) and ``structure_yrs_since_inspection`` (years since NYSDOT's
    LastInspectionDate, which is not the same as the time since the structure was built). See
    ``_download_structures_layer`` for how each field was checked."""
    import geopandas as gpd

    cols = ["structure_age_yr", "structure_condition", "structure_material",
            "structure_length_m", "structure_is_bridge",
            "structure_superstructure_condition", "structure_fhwa_status",
            "structure_opening_width_m", "structure_num_spans", "structure_culvert_skew_deg",
            "structure_yrs_since_inspection"]
    structs = _download_structures_layer()
    if structs is None or len(structs) == 0:
        out = pd.DataFrame({c: np.nan for c in cols}, index=df.index)
        out["structure_material"] = "none"
        out["structure_is_bridge"] = "none"
        out["structure_fhwa_status"] = "none"
        out["structure_streambed_material"] = "none"
        out["structure_type"] = "none"
        return out

    pts = gpd.GeoDataFrame(
        {"_idx": df.index}, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="epsg:4326")
    pts_m = pts.to_crs("epsg:5070")
    structs_m = structs.to_crs("epsg:5070")
    joined = gpd.sjoin_nearest(pts_m, structs_m, max_distance=radius_m,
                               distance_col="_dist_m", how="left")
    joined = joined.drop_duplicates(subset="_idx").set_index("_idx")

    matched = joined["_dist_m"].notna() if "_dist_m" in joined else pd.Series(dtype=bool)
    matched = matched.reindex(df.index).fillna(False)

    yr = pd.to_numeric(joined.get("YearBuilt"), errors="coerce").reindex(df.index)
    cond = pd.to_numeric(joined.get("ConditionRating"), errors="coerce").reindex(df.index).map(_clean_structure_condition)
    length_ft = pd.to_numeric(joined.get("length_ft"), errors="coerce").reindex(df.index)
    mat = (joined.get("GTMSMaterial").reindex(df.index) if "GTMSMaterial" in joined
          else pd.Series(index=df.index, dtype="object"))
    ib = (joined.get("is_bridge").reindex(df.index) if "is_bridge" in joined
         else pd.Series(index=df.index, dtype="object"))
    # NBI_SuperstructureCondition uses 'N' for not applicable next to the 0 to 9 scale.
    # to_numeric(errors='coerce') turns it into NaN, which is the right result.
    super_cond = (pd.to_numeric(joined.get("NBI_SuperstructureCondition"), errors="coerce").reindex(df.index)
                 if "NBI_SuperstructureCondition" in joined else pd.Series(index=df.index, dtype="float64"))
    fhwa = (joined.get("FHWA_Condition_Status").reindex(df.index) if "FHWA_Condition_Status" in joined
           else pd.Series(index=df.index, dtype="object"))
    width_ft = pd.to_numeric(joined.get("width_ft"), errors="coerce").reindex(df.index)
    num_spans = pd.to_numeric(joined.get("num_spans"), errors="coerce").reindex(df.index)
    skew = (pd.to_numeric(joined.get("CulvertSkew"), errors="coerce").reindex(df.index)
           if "CulvertSkew" in joined else pd.Series(index=df.index, dtype="float64"))
    streambed = (joined.get("StreamBedMaterial").reindex(df.index) if "StreamBedMaterial" in joined
                else pd.Series(index=df.index, dtype="object"))
    stype = (joined.get("structure_type").reindex(df.index) if "structure_type" in joined
            else pd.Series(index=df.index, dtype="object"))
    yrs_insp = (pd.to_numeric(joined.get("yrs_since_inspection"), errors="coerce").reindex(df.index)
               if "yrs_since_inspection" in joined else pd.Series(index=df.index, dtype="float64"))

    out = pd.DataFrame(index=df.index)
    out["structure_age_yr"] = (_STRUCTURES_CURRENT_YEAR - yr).where(matched)
    out["structure_condition"] = cond.where(matched)
    out["structure_material"] = mat.where(matched).fillna("none")
    out["structure_length_m"] = (length_ft * 0.3048).where(matched)
    out["structure_superstructure_condition"] = super_cond.where(matched)
    out["structure_fhwa_status"] = fhwa.where(matched).fillna("none")
    out["structure_opening_width_m"] = (width_ft * 0.3048).where(matched)
    out["structure_num_spans"] = num_spans.where(matched)
    # CulvertSkew and StreamBedMaterial exist only for culverts (NaN or None on the bridges layer), so
    # .where(matched) is enough.
    out["structure_culvert_skew_deg"] = skew.where(matched)
    out["structure_streambed_material"] = streambed.where(matched).fillna("none")
    out["structure_type"] = stype.where(matched).fillna("none")
    out["structure_yrs_since_inspection"] = yrs_insp.where(matched)
    out["structure_is_bridge"] = pd.Series("none", index=df.index)
    out.loc[matched & (ib == True), "structure_is_bridge"] = "bridge"  # noqa: E712
    out.loc[matched & (ib == False), "structure_is_bridge"] = "culvert"  # noqa: E712
    return out


# Constriction ratio: the structure's opening width divided by the stream's natural bankfull width. A narrow
# structure compared with the channel changes velocity and depth and can perch. Bankfull width comes from the
# USGS regional regression (Mulvihill et al. 2009, SIR 2009-5144, table 3), with 7 hydrologic regions and a
# statewide equation as a fallback. The region boundaries come from the USGS StreamStats ArcGIS service
# (gis.streamstats.usgs.gov/arcgis/rest/services/nss/regions/MapServer/28, grid_name == 'bkfullreg_g'). 98.0% of
# crossings fall in a region by point-in-polygon and the rest use the statewide equation. It was tried as a
# model feature and dropped, but it is still computed and cached.

_BANKFULL_REGIONS_URL = ("https://gis.streamstats.usgs.gov/arcgis/rest/services/"
                         "nss/regions/MapServer/28/query")
_BANKFULL_REGIONS_PATH = CACHE / "_bankfull_regions_ny.geojson"

# Table 3 of Mulvihill et al. 2009: bankfull width (ft) = a * drainage_area(mi^2)^b, fitted for each hydrologic
# region from cross-section surveys. The values were taken from the published table itself. They are keyed by
# the region polygon's Name field, so the join needs no other lookup.
_BANKFULL_WIDTH_EQUATIONS = {
    "Bankfull_Regions_1_and_2_SIR2009_5144": (21.5, 0.362),
    "Bankfull_Region_3_SIR2009_5144": (24.0, 0.292),
    "Bankfull_Region_4_SIR2009_5144": (17.1, 0.460),
    "Bankfull_Region_4a_SIR2009_5144": (9.1, 0.545),
    "Bankfull_Region_5_SIR2009_5144": (13.5, 0.449),
    "Bankfull_Region_6_SIR2009_5144": (16.9, 0.419),
    "Bankfull_Region_7_SIR2009_5144": (10.8, 0.458),
}
_BANKFULL_WIDTH_STATEWIDE = (16.9, 0.401)  # same table's statewide-pooled fit (n=281, R2=0.84)

_KM2_TO_MI2 = 0.3861021585


def _download_bankfull_regions():
    """One-time download of the 7 bankfull-region polygons (see module
    docstring above) - a live ArcGIS FeatureServer query, not a bundled
    shapefile of unknown provenance."""
    try:
        import geopandas as gpd
    except ImportError:
        return None
    if _BANKFULL_REGIONS_PATH.exists():
        return gpd.read_file(_BANKFULL_REGIONS_PATH)
    r = _get(_BANKFULL_REGIONS_URL, params={
        "where": "grid_name='bkfullreg_g'", "outFields": "Name", "outSR": 4326, "f": "geojson",
    }, timeout=60)
    if r is None or r.status_code != 200:
        return None
    feats = r.json().get("features", [])
    if not feats:
        return None
    gdf = gpd.GeoDataFrame.from_features(feats, crs="epsg:4326")
    gdf.to_file(_BANKFULL_REGIONS_PATH, driver="GeoJSON")
    return gdf


def fetch_bankfull_width_bulk(df: pd.DataFrame) -> pd.Series:
    """For every row of ``df`` (needs ``lat``, ``lon``, ``drainage_area_km2``),
    returns the regionally-appropriate bankfull channel width in meters -
    NaN only where drainage area itself is missing (the region join always
    resolves, via the statewide fallback, since it's a point-in-polygon
    match against a 7-region-plus-everything-else partition, not a nearest-
    feature search that can come up empty)."""
    import geopandas as gpd

    regions = _download_bankfull_regions()
    da_mi2 = pd.to_numeric(df["drainage_area_km2"], errors="coerce") * _KM2_TO_MI2

    if regions is None or len(regions) == 0:
        a, b = _BANKFULL_WIDTH_STATEWIDE
        return (a * da_mi2.clip(lower=0.01) ** b * 0.3048).rename("bankfull_width_m")

    pts = gpd.GeoDataFrame(
        {"_idx": df.index}, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="epsg:4326")
    joined = gpd.sjoin(pts, regions[["Name", "geometry"]], how="left", predicate="within")
    joined = joined.drop_duplicates(subset="_idx").set_index("_idx")
    region_name = joined["Name"].reindex(df.index)

    a = region_name.map(lambda n: _BANKFULL_WIDTH_EQUATIONS.get(n, (np.nan, np.nan))[0])
    b = region_name.map(lambda n: _BANKFULL_WIDTH_EQUATIONS.get(n, (np.nan, np.nan))[1])
    fallback_a, fallback_b = _BANKFULL_WIDTH_STATEWIDE
    a = a.fillna(fallback_a)
    b = b.fillna(fallback_b)
    # A drainage area of 0 or less is not physical and DA^b would be undefined for fractional b, so it is
    # clipped to a tiny positive value and the row is kept, as log_drainage_area does.
    width_ft = a * da_mi2.clip(lower=0.01) ** b
    return (width_ft * 0.3048).rename("bankfull_width_m")


def compute_constriction_ratio(df: pd.DataFrame) -> pd.Series:
    """``structure_span_m / bankfull_width_m`` - the direct opening-vs-
    channel comparison. The "opening" side is NOT the same source column for
    both structure types, because the manual-verified physical meaning
    differs (see ``_download_structures_layer``): for a bridge,
    ``structure_length_m`` (from ``BridgeLengthft``) already IS the
    channel-spanning axis, confirmed by NYSDOT's own *Bridge and Large
    Culvert Inventory Manual* ("Structure Length must be greater than or
    equal to the Span Length"); for a culvert, ``structure_length_m``
    (from ``StructureLength``/"Out to Out Length") is instead the
    tunnel/embankment axis, and it's ``structure_opening_width_m`` (from the
    verified ``SpanLength`` field) that's the channel-spanning one. Returns
    NaN wherever either side is unavailable (no matched structure, or no
    drainage area to compute a bankfull width from)."""
    is_bridge = df["structure_is_bridge"] == "bridge"
    is_culvert = df["structure_is_bridge"] == "culvert"
    span_m = pd.Series(np.nan, index=df.index)
    span_m[is_bridge] = df.loc[is_bridge, "structure_length_m"]
    span_m[is_culvert] = df.loc[is_culvert, "structure_opening_width_m"]
    bankfull = pd.to_numeric(df["bankfull_width_m"], errors="coerce")
    return (span_m / bankfull.where(bankfull > 0)).rename("constriction_ratio")


# Short-duration precipitation intensity. StreamCat's sc_precip_mm is mean annual precipitation. This is storm
# intensity, which decides whether a fixed-size structure is overwhelmed whatever the yearly average is.
#
# The source is NOAA Atlas 14 Volume 10, Precipitation Frequency Data Server
# (hdsc.nws.noaa.gov/pub/hdsc/data/ne/, gridded ASCII in ESRI format, NAD83, about 1 km cells, values in
# thousandths of an inch). We checked it against NOAA's published point estimates for Albany (3.799 in) and New
# York City (5.516 in). The columns are precip_24h_10yr_in (24 hour, 10 year storm depth), precip_60m_10yr_in (1
# hour, 10 year) and their ratio precip_flashiness_ratio. The model uses only the 24 hour depth (see models.py).

_PRECIP_GRIDS = {
    # (zip URL, inner .asc filename, output column, 1000ths-inch -> inches)
    "precip_24h_10yr_in": ("https://hdsc.nws.noaa.gov/pub/hdsc/data/ne/ne10yr24ha.zip", "ne10yr24ha.asc"),
    "precip_60m_10yr_in": ("https://hdsc.nws.noaa.gov/pub/hdsc/data/ne/ne10yr60ma.zip", "ne10yr60ma.asc"),
}


def _load_precip_grid(col: str) -> tuple[dict, np.ndarray] | None:
    """Downloads (once) + caches one NOAA Atlas 14 grid as a compressed
    .npz (header fields + the array) - much faster to reload than
    re-parsing an 8MB ASCII grid with ``np.loadtxt`` on every run."""
    npz_path = CACHE / f"_precip_{col}.npz"
    if npz_path.exists():
        z = np.load(npz_path)
        header = {k: float(z[k]) for k in ("ncols", "nrows", "xllcorner", "yllcorner",
                                           "cellsize", "nodata_value")}
        return header, z["arr"]

    import io
    import zipfile

    url, inner_name = _PRECIP_GRIDS[col]
    r = _get(url, timeout=120)
    if r is None or r.status_code != 200:
        return None
    try:
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        raw = zf.read(inner_name).decode("ascii")
    except (zipfile.BadZipFile, KeyError):
        return None
    lines = raw.splitlines()
    header = {}
    for line in lines[:6]:
        k, v = line.split()
        header[k.lower()] = float(v)
    arr = np.loadtxt(lines[6:])
    np.savez_compressed(npz_path, arr=arr, **header)
    return header, arr


def fetch_precip_intensity_bulk(df: pd.DataFrame) -> pd.DataFrame:
    """Nearest-cell lookup into the two NOAA Atlas 14 grids for every row of
    ``df`` - returns ``precip_24h_10yr_in``, ``precip_60m_10yr_in`` (both
    inches), and ``precip_flashiness_ratio`` (60min/24h depth ratio). NaN
    wherever a grid failed to load or a point falls on a NODATA cell (not
    expected within NY, kept as a safety net)."""
    out = pd.DataFrame(index=df.index)
    lat = pd.to_numeric(df["lat"], errors="coerce")
    lon = pd.to_numeric(df["lon"], errors="coerce")
    for col in _PRECIP_GRIDS:
        loaded = _load_precip_grid(col)
        if loaded is None:
            out[col] = np.nan
            continue
        h, arr = loaded
        col_idx = np.floor((lon - h["xllcorner"]) / h["cellsize"]).astype("Int64")
        row_idx = np.floor((h["yllcorner"] + h["nrows"] * h["cellsize"] - lat) / h["cellsize"]).astype("Int64")
        in_bounds = (col_idx >= 0) & (col_idx < h["ncols"]) & (row_idx >= 0) & (row_idx < h["nrows"])
        vals = pd.Series(np.nan, index=df.index)
        ci = col_idx[in_bounds].astype(int).to_numpy()
        ri = row_idx[in_bounds].astype(int).to_numpy()
        raw = arr[ri, ci]
        raw = np.where(raw == h["nodata_value"], np.nan, raw)
        vals.loc[in_bounds] = raw / 1000.0  # 1000ths of an inch -> inches
        out[col] = vals
    out["precip_flashiness_ratio"] = (out["precip_60m_10yr_in"] /
                                      out["precip_24h_10yr_in"].where(out["precip_24h_10yr_in"] > 0))
    return out


# Building the predictor table


def build_predictor_table(df: pd.DataFrame, *, do_drainage=True, do_elev=True,
                          do_road_class=True, road_class_mode: str = "per_point",
                          do_streamcat=True, do_vaa=True,
                          do_aadt=False, do_terrain=False,
                          do_ecoregion=False, do_structures=False,
                          do_constriction=False, do_precip=False,
                          sleep=0.15, log_every=200) -> pd.DataFrame:
    """``road_class_mode``: ``"per_point"`` (default) queries Overpass live,
    one crossing at a time - right for small/occasional lookups (e.g.
    predict.py). ``"bulk"`` builds/reuses a local OSM spatial index and
    classifies every crossing in one vectorized pass - right for
    dataset-scale builds (build_subset.py), since it has no per-request rate
    limit once the one-time ~500MB extract is local. See module docstring."""
    df = attach_county_census(df).copy()
    if "Road_Type" in df.columns:
        # Kept for comparison only and not a model feature, because a crossing that has not been surveyed does
        # not have it.
        df["road_class_surveyed"] = df["Road_Type"].astype("string").str.strip().str.lower()

    per_point_road = do_road_class and road_class_mode == "per_point"
    n = len(df)
    da, span, comid, elev, road = [], [], [], [], []
    for i, (_, row) in enumerate(df.iterrows()):
        cid = str(row["crossing_id"])
        # Only the per-point OpenStreetMap (Overpass) fetch is rate limited, so skip the sleep when its result
        # is already cached. A resumed run then avoids the delay for rows that need no network call.
        road_was_cached = (CACHE / "osm_road" / f"{cid}.json").exists()
        if do_drainage:
            d = fetch_drainage_area(row["lon"], row["lat"], cid)
            da.append(d["drainage_area_km2"] if d else np.nan)
            span.append(d["basin_span_km"] if d else np.nan)
            comid.append(d["comid"] if d else np.nan)
        if do_elev:
            elev.append(fetch_elevation(row["lon"], row["lat"], cid))
        if per_point_road:
            rc = fetch_osm_road_class(row["lon"], row["lat"], cid)
            road.append(rc["osm_road_class"] if rc else "unknown")
        if i % log_every == 0:
            ok = int(pd.to_numeric(pd.Series(da), errors="coerce").notna().sum()) if da else 0
            print(f"  {i}/{n}  drainage_ok={ok}", flush=True)
        if per_point_road and not road_was_cached:
            time.sleep(sleep)

    if do_drainage:
        df["drainage_area_km2"] = da
        df["basin_span_km"] = span
        df["comid"] = comid
    if do_elev:
        df["elev_m"] = elev
    if do_road_class:
        if road_class_mode == "bulk":
            df["road_class"] = fetch_osm_road_class_bulk(df).values
        else:
            df["road_class"] = pd.Series(road, index=df.index).fillna("unknown")
    if do_streamcat:
        if not do_drainage:
            raise ValueError("do_streamcat requires do_drainage=True (needs COMID)")
        sc = fetch_streamcat_batch(df["comid"].tolist())
        df = df.merge(sc, left_on="comid", right_index=True, how="left", suffixes=("", "_sc"))
        # StreamCat's own watershed-area estimate fills rows where the NLDI basin polygon failed (a bad snap, a
        # timeout). It is a different source for the same quantity, and it only fills gaps and never replaces a
        # value that NLDI produced.
        da_col = pd.to_numeric(df["drainage_area_km2"], errors="coerce")
        sc_area = pd.to_numeric(df["sc_watershed_area_km2"], errors="coerce")
        backfilled = da_col.isna() & sc_area.notna()
        df["drainage_area_km2"] = da_col.where(~backfilled, sc_area)
        if backfilled.any():
            print(f"  drainage_area backfilled from StreamCat for {int(backfilled.sum())} rows", flush=True)
    if do_drainage:
        df["log_drainage_area"] = np.log10(np.clip(pd.to_numeric(df["drainage_area_km2"]), 1e-3, None))
    if do_vaa:
        if not do_drainage:
            raise ValueError("do_vaa requires do_drainage=True (needs COMID)")
        vaa = fetch_nhdplus_vaa_batch(df["comid"].tolist())
        df = df.merge(vaa, left_on="comid", right_index=True, how="left", suffixes=("", "_vaa"))
    if do_aadt:
        aadt = fetch_aadt_bulk(df)
        df["aadt"] = aadt.values
        df["log_aadt"] = np.log10(np.clip(pd.to_numeric(df["aadt"]), 1.0, None))
        print(f"  AADT matched for {int(aadt.notna().sum())}/{len(df)} rows", flush=True)
    if do_terrain:
        terr = fetch_terrain_features_bulk(df)
        for c in terr.columns:
            df[c] = terr[c].values
        print(f"  terrain matched for {int(terr['terrain_slope_100m'].notna().sum())}/{len(df)} rows", flush=True)
    if do_ecoregion:
        df["ecoregion"] = fetch_ecoregion_bulk(df).values
        print(f"  ecoregion matched for {int((df['ecoregion'] != 'unknown').sum())}/{len(df)} rows", flush=True)
    if do_structures:
        st = fetch_structure_inventory_bulk(df)
        for c in st.columns:
            df[c] = st[c].values
        print(f"  structure inventory matched for "
             f"{int((df['structure_is_bridge'] != 'none').sum())}/{len(df)} rows", flush=True)
    if do_constriction:
        if not do_structures or not do_drainage:
            raise ValueError("do_constriction requires do_structures=True and do_drainage=True")
        df["bankfull_width_m"] = fetch_bankfull_width_bulk(df).values
        df["constriction_ratio"] = compute_constriction_ratio(df).values
        print(f"  constriction ratio computed for "
             f"{int(df['constriction_ratio'].notna().sum())}/{len(df)} rows", flush=True)
    if do_precip:
        pr = fetch_precip_intensity_bulk(df)
        for c in pr.columns:
            df[c] = pr[c].values
        print(f"  precip intensity matched for "
             f"{int(df['precip_24h_10yr_in'].notna().sum())}/{len(df)} rows", flush=True)
    return df
