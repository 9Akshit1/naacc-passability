"""Fetch HUC8 (sub-basin) polygons from the USGS WBD service for the study bbox.

Used to give every crossing in the Plunkett multi-state dataset a HUC8 id, so the
cross-validation can hold out whole watersheds (same standard as the NY-only work).
The polygons are lightly generalised (maxAllowableOffset ~ 100 m) - fine for
point-in-polygon on crossings, and it keeps the download to a few MB.

    python scripts/fetch_huc8_boundaries.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

URL = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/4/query"
# Bounding box of the Plunkett dataset (lon -82.73..-66.99, lat 36.66..47.36), padded.
BBOX = (-83.2, 36.4, -66.5, 47.6)
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "boundaries" / "huc8_study_area.geojson"
PAGE = 40


def main() -> None:
    feats, offset = [], 0
    while True:
        params = {
            "where": "1=1", "geometry": ",".join(map(str, BBOX)), "geometryType": "esriGeometryEnvelope",
            "inSR": 4326, "spatialRel": "esriSpatialRelIntersects", "outFields": "huc8,name,states",
            "outSR": 4326, "maxAllowableOffset": 0.001, "f": "geojson",
            "resultOffset": offset, "resultRecordCount": PAGE, "orderByFields": "huc8",
        }
        for attempt in range(6):
            try:
                r = requests.get(URL, params=params, timeout=120)
                r.raise_for_status()
                js = r.json()
                if "features" not in js:
                    raise RuntimeError(str(js)[:200])
                break
            except Exception as e:  # noqa: BLE001
                print("retry", attempt, e)
                time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit("giving up")
        got = js["features"]
        feats += got
        print(f"offset {offset}: +{len(got)} (total {len(feats)})")
        if len(got) < PAGE:
            break
        offset += PAGE
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    print("wrote", OUT, len(feats), "HUC8 polygons")


if __name__ == "__main__":
    main()
