"""Find the NHDPlusV2 stream line (COMID) under every crossing, and how far the point had to move to reach it.

This calls the USGS NLDI "position" service once per crossing with 12 threads, at about 18 crossings per second. Results are
appended to data/interim/comids.csv and finished crossings are skipped, so the run can be stopped and restarted. The snap
distance is the shortest distance from the crossing to the stream line that was returned. A snap to a neighbouring stream can
be found later by comparing the area with d8accum.

    python -m scripts.fetch_comids
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, Point

OUT = Path("data/interim/comids.csv")
URL = "https://api.water.usgs.gov/nldi/linked-data/comid/position"
HEAD = {"User-Agent": "NAACC-passability student research"}


def one(args):
    sid, lon, lat = args
    for attempt in range(4):
        try:
            r = requests.get(URL, params={"coords": f"POINT({lon} {lat})", "f": "json"}, headers=HEAD, timeout=45)
            if r.status_code == 200:
                feats = r.json().get("features") or []
                if not feats:
                    return sid, None, None
                f = feats[0]
                line = LineString(f["geometry"]["coordinates"])
                return sid, int(f["properties"]["comid"]), (lon, lat, line.wkt)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(5 * (attempt + 1))
                continue
            return sid, None, None
        except Exception:
            time.sleep(3)
    return sid, -1, None          # -1 = failed after retries, try again next run


if __name__ == "__main__":
    d = pd.read_parquet("data/processed/plunkett_features_osm.parquet", columns=["crossing_survey_id", "lon_nad83", "lat_nad83"])
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(prev.loc[prev.comid != -1, "crossing_survey_id"])
    todo = [(r.crossing_survey_id, r.lon_nad83, r.lat_nad83) for r in d.itertuples() if r.crossing_survey_id not in done]
    print(f"{len(done)} done, {len(todo)} to fetch", flush=True)
    t0 = time.time()
    for start in range(0, len(todo), 1000):
        chunk = todo[start:start + 1000]
        with ThreadPoolExecutor(12) as ex:
            res = list(ex.map(one, chunk))
        rows = []
        for sid, comid, g in res:
            snap = None
            if g is not None:
                lon, lat, wkt = g
                from shapely import wkt as W
                line = gpd.GeoSeries([W.loads(wkt)], crs=4326).to_crs(5070).iloc[0]
                pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(5070).iloc[0]
                snap = line.distance(pt)
            rows.append(dict(crossing_survey_id=sid, comid=comid if comid is not None else 0, snap_m=snap))
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"{start + len(chunk)}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)
