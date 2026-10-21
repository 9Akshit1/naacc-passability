"""Download the public NAACC_Culvert_Assessment_2 ArcGIS FeatureServer layer (no login needed).

Compiled by NAACC / The Nature Conservancy; 13 northeastern states + DC. Used ONLY for auditing
(target definition vs NAACC's own published score, and what Plunkett's filtering removed) - it is a
different snapshot from Plunkett's file. Saved to data/raw/naacc_public_layer.parquet."""
import time
from pathlib import Path
import pandas as pd, requests

BASE = "https://services.arcgis.com/cJ9YHowT8TU7DUyn/ArcGIS/rest/services/NAACC_Culvert_Assessment_2/FeatureServer/0"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "naacc_public_layer.parquet"
H = {"User-Agent": "NAACC-passability-research/1.0 (student project; aksheru09@gmail.com)"}

def main():
    n = requests.get(f"{BASE}/query", params={"where": "1=1", "returnCountOnly": "true", "f": "json"}, headers=H, timeout=60).json()["count"]
    rows, offset, page = [], 0, 1000
    while offset < n:
        for attempt in range(6):
            try:
                r = requests.get(f"{BASE}/query", params={"where": "1=1", "outFields": "*", "returnGeometry": "true", "outSR": 4326,
                                 "orderByFields": "FID", "resultOffset": offset, "resultRecordCount": page, "f": "json"}, headers=H, timeout=180)
                feats = r.json()["features"]; break
            except Exception as e:  # noqa: BLE001
                print("retry", attempt, repr(e)[:80]); time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit("failed")
        for f in feats:
            a = dict(f["attributes"]); g = f.get("geometry") or {}
            a["_lon"], a["_lat"] = g.get("x"), g.get("y"); rows.append(a)
        offset += len(feats); print(offset, "/", n, flush=True)
        if not feats: break
    df = pd.DataFrame(rows); OUT.parent.mkdir(parents=True, exist_ok=True); df.to_parquet(OUT)
    print("saved", OUT, df.shape)

if __name__ == "__main__":
    main()
