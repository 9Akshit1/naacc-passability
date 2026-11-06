"""Pennsylvania pilot: is the Plunkett file the set of surveys that sit on a road and stream intersection?

The hypothesis is that DSL kept only surveys within 30 m of a road-stream crossing found from GIS data, so that most of the extra
SARP rows are at sites with no such crossing nearby, which would explain why a model fitted on the Plunkett rows does not predict them.
The test builds road and stream-line intersection points for Pennsylvania (OpenStreetMap roads except tracks, and the NHDPlusV2
stream lines already downloaded), measures each survey's distance to the nearest intersection, and compares the Plunkett rows with the
extra rows. It then scores the extra rows with the model fitted on Plunkett development rows (public-data inputs), split by whether an
intersection is within 30 m. Modelling uses development rows and the extra rows outside the test basins only. The table of distances
uses all Pennsylvania Plunkett rows.
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from shapely import STRtree, get_coordinates, intersection
from shapely.geometry import LineString
from sklearn.metrics import r2_score

from scripts.plunkett_streamcat_eval import reg
from src import osm_context as oc, plunkett_cv as cv, plunkett_features as pf, streamcat_feats as sf

PBF = Path("data/raw/osm/pbf/pennsylvania.osm.pbf")
BBOX = (-80.6, 39.6, -74.6, 42.4)


def nhd_lines():
    seen, ids, geoms = set(), [], []
    for f in ("data/interim/nhd_flowlines.jsonl", "data/interim/sarp_flowlines.jsonl"):
        for row in Path(f).read_text().splitlines():
            for ln in json.loads(row)["lines"]:
                if ln["comid"] in seen or not ln["paths"]:
                    continue
                xs = [p for path in ln["paths"] for p in path]
                if not (BBOX[0] <= xs[0][0] <= BBOX[2] and BBOX[1] <= xs[0][1] <= BBOX[3]):
                    continue
                seen.add(ln["comid"])
                for path in ln["paths"]:
                    if len(path) >= 2:
                        ids.append(ln["comid"]); geoms.append(LineString(path))
    return gpd.GeoSeries(geoms, crs=4326).to_crs(5070)


def intersection_points(roads, lines):
    tree = STRtree(roads.geometry.values)
    li, ri = tree.query(lines.values, predicate="intersects")[0], tree.query(lines.values, predicate="intersects")[1]
    pts = intersection(lines.values[li], roads.geometry.values[ri])
    out = []
    for g in pts:
        if g.is_empty:
            continue
        out.append(get_coordinates(g.representative_point() if g.geom_type != "Point" else g)[0])
    return np.array(out)


if __name__ == "__main__":
    roads, _ = oc.read_lines(PBF)
    roads = roads[roads.highway != "track"].reset_index(drop=True)
    lines = nhd_lines()
    X = intersection_points(roads, lines)
    print(f"{len(roads)} roads, {len(lines)} stream line pieces, {len(X)} intersection points", flush=True)
    tree = cKDTree(X)

    def dist_to_cross(lon, lat):
        p = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(5070)
        return tree.query(np.c_[p.x, p.y])[0]

    d = pd.read_parquet("data/processed/plunkett_features_osm.parquet")
    d, _ = pf.add_derived(d)
    d = pd.concat([d, sf.build(d).drop(columns="comid")], axis=1)
    pa = d[d.state == "PA"].copy()
    pa["dx"] = dist_to_cross(pa.lon_nad83, pa.lat_nad83)
    B = pd.read_parquet("data/interim/sarp_new_features.parquet")
    Bp = B[B.State == "Pennsylvania"].copy()
    Bp["dx"] = dist_to_cross(Bp.lon, Bp.lat)
    print("\nDistance from the survey point to the nearest road x stream intersection (Pennsylvania):")
    for name, x in (("Plunkett rows", pa.dx), ("extra SARP rows", Bp.dx)):
        print(f"  {name:16s} n={len(x):5d}  within 15 m {np.mean(x <= 15):.3f}  within 30 m {np.mean(x <= 30):.3f}  within 100 m {np.mean(x <= 100):.3f}  median {np.median(x):.0f} m")

    A = d[~cv.locked_test_mask(d)].reset_index(drop=True)
    P = ["snap_m", "snap2_m", "order_line", "ftype_line"] + [c for c in B.columns if c.startswith(("sc_", "vaa_"))]
    P = [c for c in P if c in A.columns]
    p = np.mean([reg(s).fit(A[P], A.aquatic_score).predict(Bp[P]) for s in (0, 1, 2)], axis=0)
    Bp["p"], Bp["y"] = p, Bp.SARP_Score
    print("\nModel fitted on Plunkett development rows (public-data inputs), scored on the extra Pennsylvania rows:")
    for name, m in (("intersection within 30 m", Bp.dx <= 30), ("intersection within 15 m", Bp.dx <= 15),
                    ("no intersection within 100 m", Bp.dx > 100), ("all extra PA rows", Bp.dx >= 0)):
        g = Bp[m]
        print(f"  {name:30s} n={len(g):5d}  R2={r2_score(g.y, g.p):7.3f}  Spearman={spearmanr(g.y, g.p)[0]:.3f}  mean y={g.y.mean():.3f}")
    ap = A[A.state == "PA"]
    dxa = dist_to_cross(ap.lon_nad83, ap.lat_nad83)
    print("\nReference, Plunkett Pennsylvania development rows, same model, HUC8 CV, by the same distance:")
    oof = pd.Series(np.nan, index=A.index)
    for tr, te in cv.random_group_kfold(A, "huc8", 5, 0):
        oof.loc[te] = reg(0).fit(A.loc[tr, P], A.aquatic_score.loc[tr]).predict(A.loc[te, P])
    for name, m in (("within 30 m", dxa <= 30), ("beyond 30 m", dxa > 30)):
        g = ap[m]
        if len(g) > 50:
            print(f"  {name:30s} n={len(g):5d}  R2={r2_score(g.aquatic_score, oof.loc[g.index]):7.3f}")
    print("done", flush=True)
