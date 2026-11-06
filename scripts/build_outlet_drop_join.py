"""Join outlet drop values from the public NAACC layer to Ethan Plunkett's crossings, for all 13 states.

Plunkett's file does not have the outlet drop. The public layer (data/raw/naacc_public_layer.parquet, 53,173 surveys) does,
and covers every state. A layer row is accepted for a crossing only when both of these hold:
  1. the two points are within MAX_M metres of each other
  2. the NAACC aquatic score in the layer and Plunkett's aquatic_score differ by less than SCORE_TOL

Pass 1 tries rows with the same survey id, because ids repeat across state databases and the two checks above are needed.
Pass 2 matches the remaining crossings to the nearest unused layer point. Pass 3 fills crossings that are still empty from the
older New York export, which has the same outlet drop field.

The layer has one row for each measured pipe, so a crossing with several pipes can have different drops. A crossing whose
accepted rows differ by more than AMBIG_FT gets no value. A crossing with no accepted row is left empty and is never filled in
by a model.

Output: data/processed/plunkett_outlet_drop.parquet with crossing_code, drop_ws (drop to the water surface), drop_sb (drop to
the stream bottom), match_pass, match_m, score_diff and ambiguous. Nothing existing is changed.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

MAX_M, SCORE_TOL, AMBIG_FT = 30.0, 0.02, 0.05
K = 111320.0


def xy(lat, lon):
    return np.c_[np.asarray(lon) * K * np.cos(np.radians(40.0)), np.asarray(lat) * 110540.0]


def dist_m(lat1, lon1, lat2, lon2):
    return np.hypot((lat1 - lat2) * 110540.0, (lon1 - lon2) * K * np.cos(np.radians(lat1)))


if __name__ == "__main__":
    d = pd.read_parquet("data/processed/plunkett_features_osm.parquet",
                        columns=["crossing_code", "crossing_survey_id", "state", "aquatic_score", "lat_nad83", "lon_nad83"]).reset_index(drop=True)
    n = pd.read_parquet("data/raw/naacc_public_layer.parquet").reset_index(drop=True)
    n = pd.DataFrame({"lid": np.arange(len(n)), "sid": n.Survey_Id.astype("Int64"), "lat": n["_lat"], "lon": n["_lon"],
                      "score": pd.to_numeric(n.Aqua_Pass_, errors="coerce"),
                      "drop_ws": pd.to_numeric(n.Outlet_D_1, errors="coerce"), "drop_sb": pd.to_numeric(n.Outlet_Dro, errors="coerce")})
    # The shapefile truncates both names to 'Outlet_Dro...'. Checked against the older NY survey export (same survey ids, nonzero rows):
    # Outlet_D_1 = Outlet_Drop_To_Water_Surface (97% equal), Outlet_Dro = Outlet_Drop_To_Stream_Bottom (96% equal).
    # -1 in these fields is the survey's "not measured" code, so it becomes NaN, never a drop of -1.
    n[["drop_ws", "drop_sb"]] = n[["drop_ws", "drop_sb"]].where(n[["drop_ws", "drop_sb"]] >= 0)
    n = n.dropna(subset=["lat", "lon"])
    d["sid"] = d.crossing_survey_id.astype("Int64")
    print(f"Plunkett {len(d)}, layer {len(n)}", flush=True)

    # pass 1: same survey id, then the location + score checks (several layer rows can share an id: keep the closest passing one)
    m1 = d.reset_index().rename(columns={"index": "di"}).merge(n, on="sid", suffixes=("", "_l"))
    m1["match_m"] = dist_m(m1.lat_nad83, m1.lon_nad83, m1.lat, m1.lon)
    m1["score_diff"] = (m1.aquatic_score - m1.score).abs()
    cand1 = m1[(m1.match_m <= MAX_M) & (m1.score_diff < SCORE_TOL)]
    # The layer holds one row per measured pipe, so the same survey (same point, same crossing-level score) can appear two
    # or more times with different outlet drops. It is not defined which pipe a single yes/no should describe, so a crossing
    # whose accepted rows disagree by more than AMBIG_FT gets no outlet-drop value (1,076 crossings, 431 of them with a
    # different perched yes/no).
    sp1 = cand1.groupby("di").drop_ws.agg(lambda s: s.max() - s.min() if s.notna().any() else 0.0)
    ambiguous = set(sp1[sp1 > AMBIG_FT].index)
    ok = cand1.sort_values(["di", "match_m"]).drop_duplicates("di")
    ok = ok.assign(match_pass=1)
    print(f"pass 1 (id + location + score): {len(ok)} accepted of {len(m1)} id matches", flush=True)

    # pass 2: location + score among the layer rows not yet used
    left_d = d.loc[~d.index.isin(ok.di)]
    left_n = n[~n.lid.isin(ok.lid)].reset_index(drop=True)
    tree = cKDTree(xy(left_n.lat, left_n.lon))
    cand = tree.query_ball_point(xy(left_d.lat_nad83, left_d.lon_nad83), r=MAX_M * 1.2)
    rows = []
    for di, c in zip(left_d.index, cand):
        if not c:
            continue
        r = left_n.iloc[c]
        dm = dist_m(d.at[di, "lat_nad83"], d.at[di, "lon_nad83"], r.lat.values, r.lon.values)
        sd = np.abs(d.at[di, "aquatic_score"] - r.score.values)
        good = np.where((dm <= MAX_M) & (sd < SCORE_TOL))[0]
        if len(good):
            j = good[np.argmin(dm[good])]
            rows.append((di, int(r.lid.iat[j]), dm[j], sd[j]))
            v = r.drop_ws.values[good]
            if np.isfinite(v).any() and np.nanmax(v) - np.nanmin(v) > AMBIG_FT:
                ambiguous.add(di)
    p2 = pd.DataFrame(rows, columns=["di", "lid", "match_m", "score_diff"])
    p2 = p2.sort_values("match_m").drop_duplicates("lid")                      # one layer row serves one crossing
    p2 = p2.merge(n[["lid", "drop_ws", "drop_sb"]], on="lid").assign(match_pass=2)
    print(f"pass 2 (location + score): {len(p2)} accepted", flush=True)

    out = pd.concat([ok[["di", "lid", "drop_ws", "drop_sb", "match_pass", "match_m", "score_diff"]], p2], ignore_index=True)
    out = out.sort_values("match_m").drop_duplicates("lid").drop_duplicates("di")
    res = d[["crossing_code", "state"]].join(out.set_index("di")[["drop_ws", "drop_sb", "match_pass", "match_m", "score_diff"]])
    res["ambiguous"] = res.index.isin(ambiguous)
    res.loc[res.ambiguous, ["drop_ws", "drop_sb"]] = np.nan
    print(f"ambiguous (accepted rows disagree by > {AMBIG_FT} ft): {int(res.ambiguous.sum())} crossings set to no value", flush=True)
    # pass 3: crossings still blank get the value from the older NY survey export (same crossing code); where both sources hold a value
    # they are compared and the agreement is printed, the public-layer value wins.
    old = pd.read_parquet("data/processed/subset_predictors.parquet")
    o = pd.DataFrame({"crossing_code": old.Crossing_Code,
                      "o_ws": pd.to_numeric(old.Outlet_Drop_To_Water_Surface, errors="coerce").where(lambda x: x >= 0)})
    res = res.merge(o, on="crossing_code", how="left")
    both = res.drop_ws.notna() & res.o_ws.notna()
    print(f"NY export vs public layer where both hold a value: {both.sum()} crossings, equal {np.isclose(res.drop_ws[both], res.o_ws[both]).mean():.3f}")
    fill = res.drop_ws.isna() & res.o_ws.notna() & ~res.ambiguous           # the older export keeps one pipe: never used for ambiguous ones
    res.loc[fill, "drop_ws"] = res.loc[fill, "o_ws"]; res.loc[fill, "match_pass"] = 3
    print(f"pass 3 (older NY export): {fill.sum()} crossings filled")
    res = res.drop(columns="o_ws")
    res.drop(columns="state").to_parquet("data/processed/plunkett_outlet_drop.parquet", index=False)
    res["has"] = res.drop_ws.notna()
    print("\nshare of Plunkett crossings with an outlet-drop value, by state:")
    print(res.groupby("state").has.agg(["size", "sum", "mean"]).round(3).to_string())
    print(f"\nTOTAL {res.has.sum()} of {len(res)} ({res.has.mean():.3f})")
    print("match distance (m) median %.2f, p95 %.2f; score diff median %.4f" % (
        res.match_m.median(), res.match_m.quantile(.95), res.score_diff.median()))
