"""FHWA National Bridge Inventory (NBI, 2025 delimited files) -> per-crossing features.

Why NBI rather than a state DOT layer: it is ONE federally-defined dataset with the
same fields and coding in every state, so using it does not inject a state-specific
data source (the concern Ethan Plunkett raised about the NYSDOT layer). It covers
every public-road bridge and every culvert/bridge-like structure longer than 6.1 m
(20 ft) along the road. Crossings with no NBI structure within the match radius get
``nbi_matched = 0`` and missing values - for those the model is effectively told
"not a large public-road structure", which is a nationally consistent statement.

NBI fields are public inventory data - available for UNSURVEYED crossings - so they
are legitimate deployment-time predictors (unlike the NAACC survey fields).

Match rule (label-free, fixed before modelling): structures with waterway service
under (item 42B in {5,6,7,8,9}); mutual nearest neighbour between crossing and
structure within ``radius_m``; the radius is chosen from the distance distribution
(see ``match_diagnostics``), never from model performance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import Transformer

NBI_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "nbi2025"
WATERWAY_SERVICE = {"5", "6", "7", "8", "9"}

_KEEP = {
    "STRUCTURE_NUMBER_008": "nbi_id", "LAT_016": "lat_raw", "LONG_017": "lon_raw",
    "SERVICE_UND_042B": "service_under", "RECORD_TYPE_005A": "record_type",
    "STRUCTURE_KIND_043A": "nbi_kind", "STRUCTURE_TYPE_043B": "nbi_type",
    "YEAR_BUILT_027": "nbi_year_built", "YEAR_RECONSTRUCTED_106": "nbi_year_recon",
    "STRUCTURE_LEN_MT_049": "nbi_length_m", "MAX_SPAN_LEN_MT_048": "nbi_max_span_m",
    "MAIN_UNIT_SPANS_045": "nbi_main_spans", "DEGREES_SKEW_034": "nbi_skew_deg",
    "DECK_WIDTH_MT_052": "nbi_deck_width_m", "ROADWAY_WIDTH_MT_051": "nbi_roadway_width_m",
    "CHANNEL_COND_061": "nbi_channel_cond", "CULVERT_COND_062": "nbi_culvert_cond",
    "SUBSTRUCTURE_COND_060": "nbi_substructure_cond", "WATERWAY_EVAL_071": "nbi_waterway_eval",
    "SCOUR_CRITICAL_113": "nbi_scour_critical", "OWNER_022": "nbi_owner",
    "ADT_029": "nbi_adt", "FUNCTIONAL_CLASS_026": "nbi_func_class",
    "TRAFFIC_LANES_ON_028A": "nbi_lanes", "VERT_CLR_UND_054B": "nbi_vert_clr_under_m",
    "LAT_UND_MT_055B": "nbi_lat_clr_under_m", "IMP_LEN_MT_076": "nbi_impr_len_m",
}


def _dms_to_deg(s: pd.Series, deg_digits: int) -> pd.Series:
    """NBI lat 'DDMMSSss' / lon 'DDDMMSSss' (implied decimals) -> decimal degrees."""
    s = s.astype(str).str.strip().str.zfill(deg_digits + 6)
    d = pd.to_numeric(s.str[:deg_digits], errors="coerce")
    m = pd.to_numeric(s.str[deg_digits:deg_digits + 2], errors="coerce")
    sec = pd.to_numeric(s.str[deg_digits + 2:], errors="coerce") / 100.0
    out = d + m / 60.0 + sec / 3600.0
    return out.where((m < 60) & (sec < 60))


def load_nbi(states: list[str], year: int = 2025, directory: Path | str | None = None) -> pd.DataFrame:
    """Read the FHWA state files (``NY25.txt`` style) and keep waterway structures with a valid position."""
    folder = Path(directory) if directory else NBI_DIR
    frames = []
    for st in states:
        f = folder / f"{st}{str(year)[2:]}.txt"
        raw = pd.read_csv(f, dtype=str, low_memory=False, usecols=list(_KEEP))
        raw = raw.rename(columns=_KEEP)
        raw["src_state"] = st  # bookkeeping only; must never be a model input
        frames.append(raw)
    df = pd.concat(frames, ignore_index=True)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip().str.strip("'").str.strip()
    df["lat"] = _dms_to_deg(df["lat_raw"], 2)
    df["lon"] = -_dms_to_deg(df["lon_raw"], 3)
    df = df[df["service_under"].isin(WATERWAY_SERVICE)].copy()
    df = df[df["lat"].between(36, 48) & df["lon"].between(-84, -66)].copy()
    num = ["nbi_year_built", "nbi_year_recon", "nbi_length_m", "nbi_max_span_m", "nbi_main_spans",
           "nbi_skew_deg", "nbi_deck_width_m", "nbi_roadway_width_m", "nbi_waterway_eval",
           "nbi_adt", "nbi_func_class", "nbi_lanes", "nbi_vert_clr_under_m", "nbi_lat_clr_under_m",
           "nbi_impr_len_m"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # NBI sentinels: year 0 / 9999 = unknown; condition 'N' (not applicable) / blank -> NaN
    df.loc[~df["nbi_year_built"].between(1700, 2026), "nbi_year_built"] = np.nan
    df.loc[~df["nbi_year_recon"].between(1700, 2026), "nbi_year_recon"] = np.nan
    for c in ["nbi_length_m", "nbi_max_span_m", "nbi_deck_width_m", "nbi_roadway_width_m"]:
        df.loc[df[c] <= 0, c] = np.nan
    for c in ["nbi_channel_cond", "nbi_culvert_cond", "nbi_substructure_cond", "nbi_scour_critical"]:
        df[c] = pd.to_numeric(df[c].where(df[c].str.fullmatch(r"[0-9]", na=False)), errors="coerce")
    df["nbi_is_culvert"] = (df["nbi_type"] == "19").astype(float)
    df["nbi_kind"] = pd.to_numeric(df["nbi_kind"], errors="coerce")
    df["nbi_type"] = pd.to_numeric(df["nbi_type"], errors="coerce")
    df["nbi_owner"] = pd.to_numeric(df["nbi_owner"], errors="coerce")
    return df.reset_index(drop=True)


def match(cross: pd.DataFrame, nbi: pd.DataFrame, radius_m: float = 60.0,
          lat="lat_nad83", lon="lon_nad83", mutual_only: bool = True, keep_id: bool = False) -> pd.DataFrame:
    """Match each crossing to the nearest NBI structure within ``radius_m``.

    By default the pair must be mutual nearest neighbours (the structure's nearest crossing is
    also this crossing), which stops one structure being claimed by two crossings. With
    ``mutual_only=False`` every crossing takes its nearest structure if it is close enough.
    Returns a frame indexed like ``cross`` with the nbi_* columns (NaN where unmatched),
    ``nbi_matched`` and ``nbi_dist_m``; ``keep_id`` adds the NBI structure number."""
    tr = Transformer.from_crs(4326, 5070, always_xy=True)
    cx, cy = tr.transform(cross[lon].values, cross[lat].values)
    nx, ny = tr.transform(nbi["lon"].values, nbi["lat"].values)
    ct, nt = cKDTree(np.c_[cx, cy]), cKDTree(np.c_[nx, ny])
    d_cn, i_cn = nt.query(np.c_[cx, cy])        # nearest structure to each crossing
    d_nc, i_nc = ct.query(np.c_[nx, ny])        # nearest crossing to each structure
    mutual = d_cn <= radius_m
    if mutual_only:
        mutual &= i_nc[i_cn] == np.arange(len(cross))
    cols = [c for c in nbi.columns if c.startswith("nbi_") and c != "nbi_id"]
    out = pd.DataFrame(np.nan, index=cross.index, columns=cols, dtype=float)
    out.loc[mutual, cols] = nbi.iloc[i_cn[mutual]][cols].values
    if keep_id:
        out.insert(0, "nbi_structure_number", pd.Series(nbi["nbi_id"].values[i_cn], index=cross.index).where(mutual))
    out["nbi_matched"] = mutual.astype(float)
    out["nbi_dist_m"] = np.where(mutual, d_cn, np.nan)
    out["_nearest_dist_m"] = d_cn           # diagnostics only; never a model input
    out["_mutual"] = mutual
    return out


def match_diagnostics(cross, nbi, lat="lat_nad83", lon="lon_nad83") -> pd.DataFrame:
    """Distance distribution of the nearest structure for each crossing (label-free)."""
    tr = Transformer.from_crs(4326, 5070, always_xy=True)
    cx, cy = tr.transform(cross[lon].values, cross[lat].values)
    nx, ny = tr.transform(nbi["lon"].values, nbi["lat"].values)
    d, _ = cKDTree(np.c_[nx, ny]).query(np.c_[cx, cy])
    bins = [0, 5, 10, 20, 30, 45, 60, 80, 100, 150, 250, 500, 1000, 1e9]
    return pd.cut(pd.Series(d), bins).value_counts().sort_index().to_frame("n_crossings")


NBI_URL = "https://www.fhwa.dot.gov/bridge/nbi/{year}/delimited/{state}{yy}.txt"


def download(states: list[str], year: int = 2025, directory: Path | str | None = None) -> list[Path]:
    """Fetch the FHWA delimited state files that are not already on disk."""
    import requests
    folder = Path(directory) if directory else NBI_DIR
    folder.mkdir(parents=True, exist_ok=True)
    got = []
    for st in states:
        dest = folder / f"{st}{str(year)[2:]}.txt"
        if not (dest.exists() and dest.stat().st_size > 1000):
            r = requests.get(NBI_URL.format(year=year, state=st, yy=str(year)[2:]), timeout=300)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"downloaded {dest.name} ({len(r.content) / 1e6:.1f} MB)")
        got.append(dest)
    return got


def _read_table(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path, low_memory=False)


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m src.nbi", description="Download FHWA NBI state files and join them to a list of crossings.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="fetch the state files")
    dl.add_argument("--states", nargs="+", required=True, help="two-letter codes, e.g. NY PA CT")
    dl.add_argument("--year", type=int, default=2025)
    dl.add_argument("--dir", default=None, help=f"where to keep the files (default {NBI_DIR})")

    jn = sub.add_parser("join", help="attach the nearest waterway NBI structure to each crossing")
    jn.add_argument("--crossings", required=True, help="CSV or parquet with an id column and latitude/longitude (WGS84 or NAD83)")
    jn.add_argument("--id-col", default="crossing_code")
    jn.add_argument("--lat-col", default="lat_nad83")
    jn.add_argument("--lon-col", default="lon_nad83")
    jn.add_argument("--states", nargs="+", default=None, help="state files to use (default: every file found in --dir)")
    jn.add_argument("--year", type=int, default=2025)
    jn.add_argument("--dir", default=None)
    jn.add_argument("--radius", type=float, default=60.0, help="metres (default 60)")
    jn.add_argument("--one-way", action="store_true", help="nearest structure within the radius, without the mutual-nearest test")
    jn.add_argument("--matched-only", action="store_true", help="write only the crossings that matched a structure")
    jn.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "download":
        download(a.states, a.year, a.dir)
        return

    folder = Path(a.dir) if a.dir else NBI_DIR
    suffix = f"{str(a.year)[2:]}.txt"
    states = a.states or sorted(p.name[:-len(suffix)] for p in folder.glob(f"??{suffix}"))
    if not states:
        raise SystemExit(f"no NBI files found in {folder}; run the download command first")
    cross = _read_table(a.crossings)
    for c in (a.id_col, a.lat_col, a.lon_col):
        if c not in cross.columns:
            raise SystemExit(f"column {c!r} not found; columns are {list(cross.columns)[:12]}...")
    structures = load_nbi(states, a.year, folder)
    m = match(cross, structures, a.radius, a.lat_col, a.lon_col, mutual_only=not a.one_way, keep_id=True)
    out = pd.concat([cross[[a.id_col]], m.drop(columns=["_nearest_dist_m", "_mutual"])], axis=1)
    if a.matched_only:
        out = out[out["nbi_matched"] == 1]
    out.to_csv(a.out, index=False)
    n = int(m["nbi_matched"].sum())
    print(f"{len(cross)} crossings, {len(structures)} waterway structures from {len(states)} state files; "
          f"{n} matched ({100 * n / len(cross):.1f}%); wrote {a.out}")


if __name__ == "__main__":
    main()
