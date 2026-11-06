"""Download the NHDPlusV2 stream lines (with COMID) around every crossing from the public EPA NHDPlus MapServer, one request per
0.1 degree tile.

This replaces fetch_comids.py, whose one-request-per-crossing lookups were refused by USGS NLDI with a 403 after about 8,000 calls.
A tile request returns every stream line in the tile (about 100 lines and 45 KB in a typical tile), so a few thousand requests
cover all 56,525 crossings. The lines are simplified to about 5 m. The nearest line to each crossing is found afterwards on this
computer (src/streamcat_feats.link_crossings), which gives both the snap distance and the second-nearest line.

The script stops at the first 403 or 429 response instead of retrying, to stay inside the service's limits.

    python -m scripts.fetch_flowlines
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://watersgeo.epa.gov/arcgis/rest/services/NHDPlus/NHDPlus/MapServer/2/query"
OUT = Path("data/interim/nhd_flowlines.jsonl")
HEAD = {"User-Agent": "NAACC-passability student research"}
TILE = 0.1
PAD = 0.005
FIELDS = "comid,ftype,streamorder,hydroseq"
BLOCKED = []


def fetch_tile(tile):
    ix, iy = tile
    x0, y0 = ix * TILE - PAD, iy * TILE - PAD
    box = f"{x0},{y0},{x0 + TILE + 2 * PAD},{y0 + TILE + 2 * PAD}"
    feats, offset = [], 0
    while True:
        if BLOCKED:
            return tile, None
        try:
            r = requests.post(URL, headers=HEAD, timeout=120, data={
                "geometry": box, "geometryType": "esriGeometryEnvelope", "inSR": 4326, "spatialRel": "esriSpatialRelIntersects",
                "outFields": FIELDS, "returnGeometry": "true", "outSR": 4326, "maxAllowableOffset": 0.00005,
                "geometryPrecision": 6, "resultOffset": offset, "resultRecordCount": 2000, "orderByFields": "comid", "f": "json"})
        except Exception:
            time.sleep(5)
            continue
        if r.status_code in (403, 429):
            BLOCKED.append(r.status_code)
            return tile, None
        if r.status_code != 200:
            time.sleep(5)
            continue
        j = r.json()
        if "error" in j:
            time.sleep(5)
            continue
        got = j.get("features", [])
        feats += got
        if not j.get("exceededTransferLimit") and len(got) < 2000:
            return tile, feats
        offset += len(got)


if __name__ == "__main__":
    d = pd.read_parquet("data/processed/plunkett_features_osm.parquet", columns=["lon_nad83", "lat_nad83"])
    tiles = sorted({(int(np.floor(x / TILE)), int(np.floor(y / TILE))) for x, y in zip(d.lon_nad83, d.lat_nad83)})
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            done.add(tuple(json.loads(line)["tile"]))
    todo = [t for t in tiles if t not in done]
    print(f"{len(tiles)} tiles, {len(done)} done, {len(todo)} to fetch", flush=True)
    t0 = time.time()
    with OUT.open("a") as fh, ThreadPoolExecutor(4) as ex:
        for n, (tile, feats) in enumerate(ex.map(fetch_tile, todo), 1):
            if feats is None:
                continue
            rows = [{"comid": f["attributes"]["comid"], "ftype": f["attributes"]["ftype"], "order": f["attributes"]["streamorder"],
                     "hydroseq": f["attributes"]["hydroseq"], "paths": f["geometry"]["paths"]} for f in feats if f.get("geometry")]
            fh.write(json.dumps({"tile": list(tile), "lines": rows}) + "\n")
            fh.flush()
            if n % 100 == 0:
                print(f"{n}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    if BLOCKED:
        print("STOPPED: the service answered", BLOCKED[0], "- not retrying. Re-run later to resume.", flush=True)
        sys.exit(1)
    print("done", flush=True)
