"""Build the extra labelled crossings from SARP's national compilation, with public-data features only.

The input is data/raw/sarp/surveyed_crossings.parquet, from scripts/fetch_sarp_crossings.py. A row is kept when:
  - it has a usable SARP_Score (0 or more). For rows with the NAACC protocol it follows the NAACC aquatic score almost exactly
    (Spearman 0.98 against our aquatic_score at 52,221 shared crossings, mean absolute difference 0.009)
  - its protocol is NAACC, NACC, Modified NAACC or one of the NAACC-SARP variants (other agencies' protocols are not used)
  - it is not within 40 m of a crossing that is already in the Plunkett file, since that would be the same crossing
  - it is not inside the seven test HUC6 basins, so the test set stays untouched
Duplicates inside SARP (the same place in several source files) are reduced to one row within 15 m.

It then finds the NHDPlusV2 stream line for each point (EPA tiles, in a separate cache file), StreamCat for the new COMIDs (a separate
cache) and NHDPlus value added attributes, and writes data/interim/sarp_new_features.parquet. Existing caches and files are only read.

    python -m scripts.sarp_build
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

from scripts import fetch_flowlines as ff, fetch_streamcat_all as fsa
from src import streamcat_feats as sf

RAW = Path("data/raw/sarp/surveyed_crossings.parquet")
FLOW = Path("data/interim/sarp_flowlines.jsonl")
SC = Path("data/interim/sarp_streamcat.jsonl")
OUT = Path("data/interim/sarp_new_features.parquet")
LOCKED_HUC6 = ("020402", "020501", "020600", "020801", "020802", "043002", "050100")
PROTOCOLS = ("NAACC", "NACC", "Modified NAACC", "NACC Surveys", "NAACC/SARP", "NAACC/SARP Coarse", "NAACC Tidal", "NAACC/SARP Tidal")


def xy(lon, lat):
    return np.c_[np.radians(lon) * 6371000 * np.cos(np.radians(38)), np.radians(lat) * 6371000]


def select_rows() -> pd.DataFrame:
    s = pd.read_parquet(RAW)
    s = s[(s.SARP_Score >= 0) & s.ProtocolUsed.isin(PROTOCOLS)].reset_index(drop=True)
    p = pd.read_parquet("data/processed/plunkett_features_osm.parquet", columns=["lon_nad83", "lat_nad83"])
    dist, _ = cKDTree(xy(p.lon_nad83.values, p.lat_nad83.values)).query(xy(s.lon.values, s.lat.values))
    n0 = len(s)
    s = s[dist > 40]
    s = s[~s.HUC8.astype(str).str[:6].isin(LOCKED_HUC6)].reset_index(drop=True)
    pairs = cKDTree(xy(s.lon.values, s.lat.values)).query_pairs(15)
    drop = {max(a, b) for a, b in pairs}
    s = s.drop(index=list(drop)).reset_index(drop=True)
    print(f"{n0} NAACC-protocol scored rows -> {len(s)} new (not in Plunkett, not in locked basins, de-duplicated)", flush=True)
    return s


def fetch_flow(s: pd.DataFrame):
    tiles = sorted({(int(np.floor(x / ff.TILE)), int(np.floor(y / ff.TILE))) for x, y in zip(s.lon, s.lat)})
    done = set()
    if FLOW.exists():
        done = {tuple(json.loads(l)["tile"]) for l in FLOW.read_text().splitlines()}
    todo = [t for t in tiles if t not in done]
    print(f"{len(tiles)} tiles, {len(done)} done, {len(todo)} to fetch", flush=True)
    t0 = time.time()
    with FLOW.open("a") as fh, ThreadPoolExecutor(4) as ex:
        for n, (tile, feats) in enumerate(ex.map(ff.fetch_tile, todo), 1):
            if feats is None:
                continue
            rows = [{"comid": f["attributes"]["comid"], "ftype": f["attributes"]["ftype"], "order": f["attributes"]["streamorder"],
                     "hydroseq": f["attributes"]["hydroseq"], "paths": f["geometry"]["paths"]} for f in feats if f.get("geometry")]
            fh.write(json.dumps({"tile": list(tile), "lines": rows}) + "\n")
            fh.flush()
            if n % 200 == 0:
                print(f"  {n}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    if ff.BLOCKED:
        print("STOPPED: the service answered", ff.BLOCKED[0], "- not retrying. Re-run later to resume.", flush=True)
        sys.exit(1)


def fetch_sc(comids):
    have = set()
    for f in (sf.STREAMCAT, SC):
        if f.exists():
            have |= {json.loads(l)["comid"] for l in f.read_text().splitlines()}
    todo = [c for c in sorted(comids) if c not in have]
    print(f"StreamCat: {len(comids)} comids, {len(todo)} to fetch", flush=True)
    with SC.open("a") as fh:
        for i in range(0, len(todo), fsa.CHUNK):
            chunk = todo[i:i + fsa.CHUNK]
            items = []
            for attempt in range(4):
                try:
                    r = requests.post(fsa.URL, headers=fsa.HEAD, timeout=120, data={
                        "comid": ",".join(map(str, chunk)), "aoi": "ws,cat", "name": ",".join(fsa.NAMES), "showareasqkm": "true"})
                    if r.status_code in (403, 429):
                        print("STOPPED: StreamCat answered", r.status_code, flush=True)
                        sys.exit(1)
                    if r.status_code == 200:
                        items = r.json().get("items", [])
                        break
                    time.sleep(5 * (attempt + 1))
                except Exception:
                    time.sleep(5)
            got = {int(it["comid"]): it for it in items if "comid" in it}
            for cm in chunk:
                fh.write(json.dumps(got.get(cm, {"comid": cm, "_empty": True})) + "\n")
            fh.flush()
            if (i // fsa.CHUNK) % 10 == 0:
                print(f"  {i + len(chunk)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    s = select_rows()
    fetch_flow(s)
    sf.FLOWLINES = FLOW
    pts = s.rename(columns={"lon": "lon_nad83", "lat": "lat_nad83"}).assign(crossing_survey_id=s.SARPID.astype(str).values)
    links = sf.link_crossings(pts)
    print(f"linked within 500 m: {links.comid.notna().mean():.3f}; within 50 m: {(links.snap_m <= 50).mean():.3f}", flush=True)
    fetch_sc({int(c) for c in links.comid.dropna()})
    rows = [json.loads(l) for f in (sf.STREAMCAT, SC) if f.exists() for l in f.read_text().splitlines()]
    scdf = pd.DataFrame(rows)
    scdf = scdf[scdf["_empty"].isna()] if "_empty" in scdf else scdf
    scdf = scdf.drop_duplicates("comid").set_index("comid")
    keep = [c for c in scdf.columns if c.endswith(("ws", "cat")) or c in ("wsareasqkm", "catareasqkm")]
    scdf = scdf[keep].apply(pd.to_numeric, errors="coerce").rename(columns={"wsareasqkm": "wsarea", "catareasqkm": "catarea"}).add_prefix("sc_")
    sc = scdf.reindex(links["comid"].values).set_axis(s.index)
    cid = pd.Series(links["comid"].values, index=s.index)
    vaa = sf.load_vaa(cid)
    keepcols = ["SARPID", "Source", "ProtocolUsed", "State", "HUC8", "lon", "lat", "SARP_Score", "CrossingType", "Constriction", "BarrierSeverity"]
    out = pd.concat([s[keepcols].reset_index(drop=True), links[["comid", "snap_m", "snap2_m", "order_line", "ftype_line"]].reset_index(drop=True),
                     sc.reset_index(drop=True), vaa.reset_index(drop=True)], axis=1)
    out.to_parquet(OUT)
    print("wrote", OUT, out.shape, flush=True)
    print("done", flush=True)
