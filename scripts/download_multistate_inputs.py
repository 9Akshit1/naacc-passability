"""Download the outside inputs for the multi-state pipeline. It can be run again safely.

Plunkett's dataset is not downloaded here. Ethan Plunkett sent it to me and it is his to distribute, so it is not in this
repository. Put passability_data.parquet and passability_predvars.txt in data/raw/plunkett/, or set NAACC_PLUNKETT_DIR.

This script fetches the Census state boundaries, the FHWA National Bridge Inventory 2025 files for 13 states, the USGS HUC8
polygons, and the OpenStreetMap bridge and culvert ways.

    python -m scripts.download_multistate_inputs [--skip-osm]
"""
import subprocess, sys, zipfile
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
STATES = ["CT", "DE", "MA", "MD", "ME", "NH", "NJ", "NY", "PA", "RI", "VA", "VT", "WV"]


def get(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return
    r = requests.get(url, timeout=300); r.raise_for_status()
    dest.write_bytes(r.content); print("fetched", dest.name, len(r.content))


def main():
    from src.plunkett_data import FTP_DIR as ftp
    assert (ftp / "passability_data.parquet").exists(), f"put Plunkett's files in {ftp}"
    b = ROOT / "data" / "raw" / "boundaries"
    get("https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_500k.zip", b / "states.zip")
    if not (b / "states").exists():
        zipfile.ZipFile(b / "states.zip").extractall(b / "states")
    for s in STATES:
        get(f"https://www.fhwa.dot.gov/bridge/nbi/2025/delimited/{s}25.txt", ROOT / "data" / "raw" / "nbi2025" / f"{s}25.txt")
    if not (b / "huc8_study_area.geojson").exists():
        subprocess.check_call([sys.executable, "scripts/fetch_huc8_boundaries.py"], cwd=ROOT)
    if "--skip-osm" not in sys.argv:
        subprocess.check_call([sys.executable, "-m", "src.osm_structures", *STATES], cwd=ROOT)


if __name__ == "__main__":
    main()
