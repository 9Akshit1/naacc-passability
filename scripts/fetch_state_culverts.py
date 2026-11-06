"""Download state DOT culvert inventories that are public ArcGIS REST layers and need no login.

These layers list culverts whether or not NAACC has surveyed them, so anything taken from them would also exist for unsurveyed
crossings. Only attributes are kept (size, length, material, elevations, ratings), plus the geometry as a WGS84 polyline for a
nearest-culvert join.

    python -m scripts.fetch_state_culverts CT VT
"""
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, MultiLineString

OUT = Path("data/raw/state_culverts")

LAYERS = {
    # Connecticut DOT drainage network, "Culverts" layer (8,773 lines, updated by CTDOT).
    "CT": dict(
        url="https://services1.arcgis.com/FCaUeJ5SOVtImake/ArcGIS/rest/services/CTDOT_Drainage_Network/FeatureServer/0",
        fields=["CulvertID", "YearInstalled", "WaterSource", "WatercourseName", "Material", "ConveyanceType", "Diameter",
                "Width", "Height", "Length", "ElevationIn", "ElevationOut", "PercentSlope", "DataSource", "OwnedBy"]),
    # Vermont AOT (VTrans) Small Culvert Inventory, "SCI Culverts" layer (48,830 lines).
    "VT": dict(
        url="https://maps.vtrans.vermont.gov/arcgis/rest/services/HWY/AMP_SmallCulvertInventory/MapServer/0",
        fields=["ASSET_SYS_ID", "DELETE_RECORD", "RETIRE_DATE", "DESCRIPTION", "INSTALL_DATE", "DRAIN_TYPE", "SYS_TYPE",
                "STRUC_TYPE", "MATERIAL", "SIZE", "HEIGHT", "WIDTH", "FILL_DEPTH", "BARL_CONDITION", "BARL_SEDIMENT",
                "OUT_TREATMENT", "OUT_EXTENSION", "OUT_CONDITION", "OUT_SEDIMENT", "OUT_EROSION", "STONE_PAD",
                "STRUCTURAL_RATING", "FUNCTIONAL_RATING", "FUNCTIONAL_STATUS", "LENGTH_FT", "IN_ZFEET", "OUT_ZFEET",
                "IN_CONDITION", "IN_SEDIMENT", "INSPECTION_DATE"]),
}


def query_all(url: str, fields: list[str], page: int = 1000) -> gpd.GeoDataFrame:
    meta = requests.get(url, params={"f": "json"}, timeout=60).json()
    have = {f["name"] for f in meta["fields"]}
    fields = [f for f in fields if f in have]
    rows, geoms, offset = [], [], 0
    while True:
        r = requests.get(url + "/query", params={
            "where": "1=1", "outFields": ",".join(fields), "returnGeometry": "true", "outSR": 4326,
            "resultOffset": offset, "resultRecordCount": page, "orderByFields": "OBJECTID", "f": "json"}, timeout=120).json()
        if "error" in r:
            raise RuntimeError(r["error"])
        feats = r.get("features", [])
        for f in feats:
            paths = (f.get("geometry") or {}).get("paths") or []
            lines = [LineString(p) for p in paths if len(p) >= 2]
            geoms.append(None if not lines else lines[0] if len(lines) == 1 else MultiLineString(lines))
            rows.append(f["attributes"])
        print(f"  {offset + len(feats)} rows", flush=True)
        if not r.get("exceededTransferLimit") and len(feats) < page:
            break
        offset += len(feats)
        time.sleep(0.2)
    return gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geoms, crs="EPSG:4326")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for st in sys.argv[1:] or LAYERS:
        spec = LAYERS[st]
        print(st, spec["url"], flush=True)
        g = query_all(spec["url"], spec["fields"])
        g = g[g.geometry.notna()]
        g.to_parquet(OUT / f"{st}.parquet")
        (OUT / f"{st}_PROVENANCE.json").write_text(json.dumps(
            {"state": st, "url": spec["url"], "rows": len(g), "fetched": time.strftime("%Y-%m-%d")}, indent=1))
        print(st, "saved", len(g), flush=True)
