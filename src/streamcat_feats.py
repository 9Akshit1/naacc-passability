"""Catchment attributes for each crossing, taken from national datasets that exist for every NHDPlusV2 stream.

- EPA StreamCat: 62 measures (soils, lithology and glacial deposits, land cover, imperviousness, climate, roads, dams and
  population) for the whole upstream watershed (ws) and for the local catchment (cat), fetched by scripts/fetch_streamcat_all.py.
- NHDPlusV2 value added attributes (cache/_nhdplus_vaa_raw.parquet): reach slope, stream order, total drainage area, mean annual
  flow (totma), roughness, travel time, and the slope of the next reaches upstream and downstream.

Each crossing is linked to the nearest NHDPlusV2 stream line (scripts/fetch_flowlines.py and link_crossings below). `snap_m` is the
distance from the crossing to that line and `snap2_m` is the distance to the next different line. Both are kept as inputs. A
crossing that is far from a mapped line is often on a small stream that NHDPlus does not show, and then the attributes describe a
different and larger stream.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

COMIDS = Path("data/interim/links.csv")
FLOWLINES = Path("data/interim/nhd_flowlines.jsonl")
STREAMCAT = Path("data/interim/streamcat_multistate.jsonl")
VAA = Path("data/interim/cache/_nhdplus_vaa_raw.parquet")

GROUPS = {
    "soil_lithology": ("clay sand om kffact wtdep rckdep perm hydrlcond compstrgth pcthydric pctalkintruvol pctalluvcoast "
                       "pctcarbresid pctcoastcrs pctcolluvsed pcteolcrs pcteolfine pctextruvol pctglaclakecrs pctglaclakefine "
                       "pctglactilclay pctglactilcrs pctglactilloam pctnoncarbresid pctsallake pctsilicic pctwater").split(),
    "land_cover": ("pctbl2019 pctconif2019 pctcrop2019 pctdecid2019 pctgrs2019 pcthay2019 pcthbwet2019 pctice2019 pctimp2019 "
                   "pctmxfst2019 pctow2019 pctshrb2019 pcturbhi2019 pcturblo2019 pcturbmd2019 pcturbop2019 pctwdwet2019").split(),
    "climate_terrain": "bfi precip8110 runoff tmax8110 tmean8110 tmin8110 elev wetindex".split(),
    "human": "damdens fert huden2010 manure minedens popden2010 rdcrs rddens agkffact".split(),
}


def load_streamcat() -> pd.DataFrame:
    rows = [json.loads(line) for line in STREAMCAT.read_text().splitlines()]
    sc = pd.DataFrame(rows)
    sc = sc[sc.get("_empty").isna()] if "_empty" in sc else sc
    sc = sc.set_index("comid")
    keep = [c for c in sc.columns if c.endswith(("ws", "cat")) or c in ("wsareasqkm", "catareasqkm")]
    sc = sc[keep].apply(pd.to_numeric, errors="coerce")
    return sc.rename(columns={"wsareasqkm": "wsarea", "catareasqkm": "catarea"}).add_prefix("sc_")


def load_vaa(comids: pd.Series) -> pd.DataFrame:
    v = pd.read_parquet(VAA).drop_duplicates("comid").set_index("comid")
    v["slope"] = v["slope"].where(v["slope"] >= 0)                       # -9998 = missing in NHDPlus VAA
    by_seq = v.reset_index().set_index("hydroseq")
    out = pd.DataFrame(index=v.index)
    out["vaa_slope"] = v["slope"]
    out["vaa_order"] = v["streamorde"]
    out["vaa_log_totda"] = np.log10(v["totdasqkm"].where(v["totdasqkm"] > 0))
    out["vaa_log_flow"] = np.log10(v["totma"].where(v["totma"] > 0))     # mean annual flow, cfs
    out["vaa_roughness"] = v["roughness"].where(v["roughness"] > 0)
    out["vaa_travel_time"] = v["pathtimema"].where(v["pathtimema"] > 0)
    out["vaa_log_arbolate"] = np.log10(v["arbolatesu"].where(v["arbolatesu"] > 0))
    out["vaa_canal"] = v["ftype"].isin(["CanalDitch", "ArtificialPath", "Connector"]).astype(float)
    dn = v["dnhydroseq"].map(by_seq["slope"]) if "slope" in by_seq else np.nan
    up = v["uphydroseq"].map(by_seq["slope"])
    out["vaa_slope_down"] = dn.where(dn >= 0)
    out["vaa_slope_up"] = up.where(up >= 0)
    out["vaa_slope_drop_down"] = out["vaa_slope"] - out["vaa_slope_down"]
    out["vaa_slope_drop_up"] = out["vaa_slope_up"] - out["vaa_slope"]
    return out.reindex(comids.values).set_axis(comids.index)


def link_crossings(d: pd.DataFrame, max_m: float = 500.0) -> pd.DataFrame:
    """Nearest stream line (by distance in EPSG:5070) for every crossing, and the distance to the next different line."""
    import geopandas as gpd
    from shapely import STRtree
    from shapely.geometry import LineString

    lines = {}
    for row in FLOWLINES.read_text().splitlines():
        for ln in json.loads(row)["lines"]:
            if ln["comid"] not in lines and ln["paths"]:
                lines[ln["comid"]] = (ln["order"], ln["ftype"], [LineString(p) for p in ln["paths"] if len(p) >= 2])
    ids, geoms, order, ftype = [], [], [], []
    for cid, (o, ft, ls) in lines.items():
        for g in ls:
            ids.append(cid); geoms.append(g); order.append(o); ftype.append(ft)
    gl = gpd.GeoSeries(geoms, crs=4326).to_crs(5070)
    tree = STRtree(gl.values)
    pts = gpd.GeoSeries(gpd.points_from_xy(d["lon_nad83"], d["lat_nad83"]), crs=4326).to_crs(5070)
    ids, order, ftype = np.array(ids), np.array(order, float), np.array(ftype, float)
    out = np.full((len(d), 5), np.nan)
    hits = tree.query(pts.values, predicate="dwithin", distance=max_m)       # (2, n_pairs): point index, line index
    pi, li = hits
    from shapely import distance as sdist
    dd = sdist(pts.values[pi], gl.values[li])
    frame = pd.DataFrame({"p": pi, "l": li, "d": dd}).sort_values(["p", "d"])
    frame["comid"] = ids[frame["l"].values]
    best = frame.drop_duplicates("p")
    out[best["p"].values, 0] = best["comid"].values
    out[best["p"].values, 1] = best["d"].values
    out[best["p"].values, 3] = order[best["l"].values]
    out[best["p"].values, 4] = ftype[best["l"].values]
    second = frame[frame["comid"] != frame["p"].map(best.set_index("p")["comid"])].drop_duplicates("p")
    out[second["p"].values, 2] = second["d"].values
    return pd.DataFrame(out, columns=["comid", "snap_m", "snap2_m", "order_line", "ftype_line"]).assign(
        crossing_survey_id=d["crossing_survey_id"].values)


def build(d: pd.DataFrame) -> pd.DataFrame:
    """Rows aligned to d (by crossing_survey_id). Columns: comid, snap_m, sc_*, vaa_*. NaN where no stream was linked."""
    c = pd.read_csv(COMIDS)
    c = c[c.comid > 0].drop_duplicates("crossing_survey_id").set_index("crossing_survey_id")
    j = c.reindex(d["crossing_survey_id"].values)
    j.index = d.index
    sc = load_streamcat().reindex(j["comid"].values).set_axis(d.index)
    vaa = load_vaa(j["comid"].astype("Int64").astype("float"))
    vaa.index = d.index
    out = pd.concat([j[["comid", "snap_m", "snap2_m", "order_line", "ftype_line"]], sc, vaa], axis=1)
    return out
