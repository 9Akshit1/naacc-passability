"""Download EPA StreamCat attributes for every COMID found by scripts/make_links.py.

There are 62 measures (soils, lithology and glacial deposits, land cover, imperviousness, climate, roads, dams and population),
each for the whole upstream watershed ("ws") and for the local catchment ("cat"), plus StreamCat's own watershed and catchment
areas. They exist for every NHDPlusV2 stream, so they exist for unsurveyed crossings too.

Each batch of 300 COMIDs is appended to data/interim/streamcat_multistate.jsonl and skipped on the next run, so the script can be
stopped and restarted. Normal TLS verification is used.

    python -m scripts.fetch_streamcat_all
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/interim/streamcat_multistate.jsonl")
URL = "https://api.epa.gov/StreamCat/streams/metrics"
HEAD = {"User-Agent": "NAACC-passability student research"}
NAMES = ("agkffact bfi clay compstrgth damdens elev fert huden2010 hydrlcond kffact manure minedens om pctalkintruvol "
         "pctalluvcoast pctbl2019 pctcarbresid pctcoastcrs pctcolluvsed pctconif2019 pctcrop2019 pctdecid2019 pcteolcrs "
         "pcteolfine pctextruvol pctglaclakecrs pctglaclakefine pctglactilclay pctglactilcrs pctglactilloam pctgrs2019 "
         "pcthay2019 pcthbwet2019 pcthydric pctice2019 pctimp2019 pctmxfst2019 pctnoncarbresid pctow2019 pctsallake "
         "pctshrb2019 pctsilicic pcturbhi2019 pcturblo2019 pcturbmd2019 pcturbop2019 pctwater pctwdwet2019 perm "
         "popden2010 precip8110 rckdep rdcrs rddens runoff sand tmax8110 tmean8110 tmin8110 wetindex wtdep").split()
CHUNK = 300


if __name__ == "__main__":
    c = pd.read_csv("data/interim/links.csv")
    want = sorted(set(int(x) for x in c.comid if x > 0))
    have = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            have.add(json.loads(line)["comid"])
    todo = [x for x in want if x not in have]
    print(f"{len(want)} comids, {len(have)} already fetched, {len(todo)} to go", flush=True)
    t0 = time.time()
    with OUT.open("a") as fh:
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i:i + CHUNK]
            items = []
            for attempt in range(4):
                try:
                    r = requests.post(URL, headers=HEAD, timeout=120, data={
                        "comid": ",".join(map(str, chunk)), "aoi": "ws,cat", "name": ",".join(NAMES), "showareasqkm": "true"})
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
            print(f"{i + len(chunk)}/{len(todo)}  returned {len(got)}/{len(chunk)}  {time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)
