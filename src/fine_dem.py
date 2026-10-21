"""Fine-scale (lidar-era, ~1 m) ground-elevation samples around each crossing from the USGS 3DEP ImageServer.

For each crossing: the centre point plus 8 compass directions x 6 radii (3, 6, 10, 15, 25, 40 m) = 49 samples.
The service returns 1 m-scale values where lidar DEMs exist (verified 2026-10-09 in NY: 37-40 distinct elevations
on 41 points spaced 2 m apart); elsewhere it falls back to coarser data, so a per-crossing `fine_ok` flag is kept
(fraction of distinct values) and coverage is reported rather than assumed.
Samples are cached in batches under data/interim/cache/fine_dem/ so interrupted runs resume.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/getSamples"
CACHE = Path(__file__).resolve().parents[1] / "data" / "interim" / "cache" / "fine_dem"
RADII = [3, 6, 10, 15, 25, 40]
DIRS = {"N": (0, 1), "NE": (1, 1), "E": (1, 0), "SE": (1, -1), "S": (0, -1), "SW": (-1, -1), "W": (-1, 0), "NW": (-1, 1)}
PER = 1 + len(RADII) * len(DIRS)
H = {"User-Agent": "NAACC-passability-research/1.0 (student project; aksheru09@gmail.com)"}


def sample_points(lat: float, lon: float) -> list[tuple[float, float]]:
    mlat, mlon = 111_320.0, 111_320.0 * np.cos(np.radians(lat))
    pts = [(lon, lat)]
    for r in RADII:
        for dx, dy in DIRS.values():
            n = np.hypot(dx, dy)
            pts.append((lon + r * dx / n / mlon, lat + r * dy / n / mlat))
    return pts


def _fetch_batch(args):
    bid, pts = args
    f = CACHE / f"b{bid:06d}.npy"
    if f.exists():
        return bid
    for attempt in range(6):
        try:
            r = requests.post(URL, data={"geometryType": "esriGeometryMultipoint", "f": "json",
                                         "geometry": json.dumps({"points": [list(p) for p in pts], "spatialReference": {"wkid": 4326}})},
                              headers=H, timeout=180)
            if r.status_code == 200:
                s = r.json().get("samples", [])
                if len(s) == len(pts):
                    z = np.array([float(x["value"]) if x.get("value") not in (None, "", "NoData") else np.nan for x in s], dtype="float32")
                    np.save(f, z)
                    return bid
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5 * (attempt + 1))
    return None


def fetch(df: pd.DataFrame, lat="lat_nad83", lon="lon_nad83", crossings_per_batch: int = 10, workers: int = 4) -> np.ndarray:
    """Return an (n, PER) float32 array of elevations aligned to df rows (NaN where missing)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    pts = [p for la, lo in zip(df[lat].values, df[lon].values) for p in sample_points(la, lo)]
    # batch id is derived from a stable hash of the first crossing's coords so different subsets can share cache
    bs = crossings_per_batch * PER
    jobs = []
    for i in range(0, len(pts), bs):
        chunk = pts[i:i + bs]
        bid = abs(hash((round(chunk[0][0], 6), round(chunk[0][1], 6), len(chunk)))) % 10**9
        jobs.append((bid, chunk))
    with ThreadPoolExecutor(workers) as ex:
        done = 0
        for r in ex.map(_fetch_batch, jobs):
            done += 1
            if done % 50 == 0:
                print(f"  batches {done}/{len(jobs)}", flush=True)
    out = []
    for bid, chunk in jobs:
        f = CACHE / f"b{bid:06d}.npy"
        out.append(np.load(f) if f.exists() else np.full(len(chunk), np.nan, dtype="float32"))
    return np.concatenate(out).reshape(len(df), PER)


def derive(z: np.ndarray) -> pd.DataFrame:
    """Orientation-aware profile features from the (n, 49) sample array. Uses elevations only (no labels).

    The 'stream axis' of a crossing is the opposite-direction pair (N-S, NE-SW, E-W, SE-NW) with the lowest mean
    elevation at 10-25 m; its two ends give an upstream (higher) and downstream (lower) side; the perpendicular pair is
    the 'road axis'. Embankment height ~ centre elevation minus the stream-axis elevation at r metres."""
    n = len(z)
    z0 = z[:, 0]
    ring = z[:, 1:].reshape(n, len(RADII), 8)                      # [crossing, radius, direction]
    f: dict[str, np.ndarray] = {}
    for i, r in enumerate(RADII):
        a = ring[:, i, :]
        with np.errstate(all="ignore"):
            f[f"fd_ring_mean_{r}"] = np.nanmean(a, 1) - z0
            f[f"fd_ring_min_{r}"] = np.nanmin(a, 1) - z0
            f[f"fd_ring_max_{r}"] = np.nanmax(a, 1) - z0
            f[f"fd_ring_sd_{r}"] = np.nanstd(a, 1)
    with np.errstate(all="ignore"):
        pm = np.stack([np.nanmean(np.concatenate([ring[:, 2:5, j], ring[:, 2:5, j + 4]], axis=1), axis=1) for j in range(4)], 1)
    ok = np.isfinite(pm).any(1)
    ax = np.where(ok, np.nanargmin(np.where(np.isfinite(pm), pm, np.inf), axis=1), 0)
    rd = (ax + 2) % 4
    idx = np.arange(n)
    endA, endB = ring[idx, :, ax], ring[idx, :, ax + 4]                          # (n, radii)
    a_high = np.nan_to_num(endA[:, -1], nan=-1e9) >= np.nan_to_num(endB[:, -1], nan=-1e9)
    up = np.where(a_high[:, None], endA, endB); dn = np.where(a_high[:, None], endB, endA)
    for i, r in enumerate(RADII):
        f[f"fd_up_{r}"] = up[:, i] - z0
        f[f"fd_dn_{r}"] = dn[:, i] - z0
        with np.errstate(all="ignore"):
            f[f"fd_roadax_{r}"] = np.nanmean(np.stack([ring[idx, i, rd], ring[idx, i, rd + 4]], 1), 1) - z0
            f[f"fd_embank_{r}"] = z0 - np.nanmean(np.stack([endA[:, i], endB[:, i]], 1), 1)
    sl = lambda ri: (up[:, ri] - dn[:, ri]) / (2.0 * RADII[ri])
    f["fd_slope_local"], f["fd_slope_mid"], f["fd_slope_far"] = sl(2), sl(4), sl(5)
    with np.errstate(all="ignore"):
        f["fd_slope_ratio_local_far"] = f["fd_slope_local"] / (np.abs(f["fd_slope_far"]) + 1e-3)
    f["fd_step_dn_3_10"] = dn[:, 0] - dn[:, 2]
    f["fd_step_dn_10_25"] = dn[:, 2] - dn[:, 4]
    f["fd_step_up_3_10"] = up[:, 2] - up[:, 0]
    f["fd_axis_idx"] = ax.astype(float)
    with np.errstate(all="ignore"):
        f["fd_sd_all"] = np.nanstd(z, 1)
    distinct = np.array([len(np.unique(np.round(row[np.isfinite(row)], 2))) for row in z])
    f["fd_distinct"] = distinct.astype(float)                                      # ~49 where 1 m lidar DEM exists
    out = pd.DataFrame(f)
    out.loc[~np.isfinite(z[:, 0]), :] = np.nan
    return out
