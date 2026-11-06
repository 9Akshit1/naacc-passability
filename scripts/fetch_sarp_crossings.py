"""Download the national set of surveyed road-stream crossings from the SARP Aquatic Barrier Prioritization tool (read only).

The layer is NACC_SurveyedCrossings_Results on the USFWS ArcGIS organisation, with about 360,000 rows. Only the fields needed to
compare with the NAACC scores are kept. It downloads pages of 2,000 rows one after another, can be resumed
(data/raw/sarp/pages/*.json), and stops at the first 403 or 429. It writes data/raw/sarp/surveyed_crossings.parquet and a
provenance file, and touches nothing outside data/raw/sarp.

    python -m scripts.fetch_sarp_crossings
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

URL = ("https://services.arcgis.com/QVENGdaPbd4LUkLV/arcgis/rest/services/"
       "Aquatic_Barrier_Prioritization_RoadBarriers_v360/FeatureServer/0/query")
OUT = Path("data/raw/sarp")
PAGES = OUT / "pages"
HEAD = {"User-Agent": "NAACC-passability student research"}
FIELDS = ("OBJECTID,lat,lon,SARPID,Source,SourceID,CrossingCode,PartnerID,SurveyID,Snapped,NHDPlusID,StreamSizeClass,TotDASqKm,"
          "RoadType,CrossingType,Constriction,BarrierSeverity,SARP_Score,ProtocolUsed,Recon,Resurveyed,Removed,YearSurveyed,"
          "Condition,StreamOrder,Intermittent,Canal,State,HUC8,HUC12")
PAGE = 2000

if __name__ == "__main__":
    PAGES.mkdir(parents=True, exist_ok=True)
    n = requests.get(URL, params={"where": "1=1", "returnCountOnly": "true", "f": "json"}, headers=HEAD, timeout=60).json()["count"]
    print(f"{n} rows, {-(-n // PAGE)} pages", flush=True)
    for k, off in enumerate(range(0, n, PAGE)):
        f = PAGES / f"{off:07d}.json"
        if f.exists():
            continue
        for attempt in range(4):
            r = requests.get(URL, params={"where": "1=1", "outFields": FIELDS, "returnGeometry": "false", "orderByFields": "OBJECTID",
                                          "resultOffset": off, "resultRecordCount": PAGE, "f": "json"}, headers=HEAD, timeout=120)
            if r.status_code in (403, 429):
                print("STOPPED: service answered", r.status_code, "- not retrying. Re-run later to resume.", flush=True)
                sys.exit(1)
            j = r.json() if r.status_code == 200 else {}
            if "features" in j:
                f.write_text(json.dumps([x["attributes"] for x in j["features"]]))
                break
            time.sleep(5 * (attempt + 1))
        else:
            print("failed page", off, flush=True)
            sys.exit(1)
        if k % 20 == 0:
            print(f"page {k}", flush=True)
        time.sleep(0.3)
    rows = []
    for f in sorted(PAGES.glob("*.json")):
        rows += json.loads(f.read_text())
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "surveyed_crossings.parquet")
    (OUT / "PROVENANCE.json").write_text(json.dumps({"url": URL, "rows": len(df), "fetched": time.strftime("%Y-%m-%d")}, indent=1))
    print("done", len(df), flush=True)
