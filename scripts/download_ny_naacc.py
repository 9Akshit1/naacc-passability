"""Re-download the NY NAACC survey layer from the NYS DEC ArcGIS FeatureServer."""
import csv, json, time, pathlib, requests
requests.packages.urllib3.disable_warnings()
BASE = ("https://services6.arcgis.com/DZHaqZm9cxOD4CWM/arcgis/rest/services/"
        "NYS_NAACCSurveys_Projected/FeatureServer/0")
H = {"User-Agent": "NAACC research"}

def main():
    meta = requests.get(BASE, params={"f": "json"}, headers=H, timeout=60, verify=False).json()
    step = min(meta.get("maxRecordCount", 2000), 2000)
    total = requests.get(BASE + "/query", params={"where": "1=1", "returnCountOnly": "true",
                         "f": "json"}, headers=H, timeout=60, verify=False).json()["count"]
    rows, offset = [], 0
    while offset < total:
        j = requests.get(BASE + "/query", params={"where": "1=1", "outFields": "*", "f": "json",
                         "resultOffset": offset, "resultRecordCount": step,
                         "returnGeometry": "true", "outSR": 4326}, headers=H, timeout=120,
                         verify=False).json()
        for f in j.get("features", []):
            a = dict(f["attributes"]); g = f.get("geometry") or {}
            a["_geom_x"], a["_geom_y"] = g.get("x"), g.get("y")
            rows.append(a)
        offset += step; time.sleep(0.2)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = pathlib.Path("data/raw"); out.mkdir(parents=True, exist_ok=True)
    with open(out / "ny_naacc_surveys.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    json.dump({"source": BASE, "accessed": time.strftime("%Y-%m-%d"), "n_records": len(rows)},
              open(out / "ny_naacc_PROVENANCE.json", "w"), indent=2)
    print(f"wrote {len(rows)} records")

if __name__ == "__main__":
    main()
