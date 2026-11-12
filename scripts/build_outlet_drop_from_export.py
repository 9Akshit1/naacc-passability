"""Join outlet drop, outlet grade and number of structures from NAACC's "detailed" export to Ethan Plunkett's crossings.

Ethan Plunkett (email, 2026-11-09) said the detailed export from the NAACC Data Center has these fields, and that NAACC data
can be used for anything with no restrictions. He also described the layout:
  - a crossing with more than one survey has one or more rows for each survey, told apart by survey id
  - a crossing with more than one structure has several rows with the same survey id
  - for a multi-part crossing the NAACC score is the maximum passability over the structures, so the score follows the most
    passable one. A single drop for the crossing can be the mean or the minimum.

Plunkett's file already names one survey for each crossing (crossing_survey_id), so rows are picked by that id and no "most
recent survey" rule is needed. Rows of that survey are then collapsed to one value per crossing: the lowest outlet drop (the
same rule as scripts/build_outlet_drop_join.py), the lowest-drop structure's grade, and the number of structure rows.

Survey ids repeat across the state databases, so a row is accepted only when its location is within 30 m of Plunkett's point and,
when the export has the aquatic score, the score differs by less than 0.02 (the checks used for the public layer).

Usage (from the repository root):
    python -m scripts.build_outlet_drop_from_export path/to/detailed_export.csv      (or .xlsx)

The export is read by column name. If a name below is missing the script prints the names it did find and stops, so nothing is
guessed. The raw export is not committed. Output: data/processed/plunkett_outlet_drop_export.parquet.
"""
import sys

import numpy as np
import pandas as pd

MAX_M, SCORE_TOL = 30.0, 0.02
NEEDED = {"sid": "Survey_Id", "drop_ws": "Outlet_Drop_To_Water_Surface", "grade": "Outlet_Grade",
          "lat": "GIS_Latitude", "lon": "GIS_Longitude"}
OPTIONAL = {"score": "Aqua_Pass_Score", "n_culverts": "Number_Of_Culverts", "date": "Date_Observed"}


def read_export(path):
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:                    # NAACC's export is Windows-1252, not UTF-8
        return pd.read_csv(path, low_memory=False, encoding="cp1252")


def pick(raw, wanted, required):
    lower = {c.lower().replace(" ", "_"): c for c in raw.columns}
    out, missing = {}, []
    for key, name in wanted.items():
        c = lower.get(name.lower())
        if c is None:
            missing.append(name)
        else:
            out[key] = c
    if missing and required:
        print("These columns are not in the export:", ", ".join(missing))
        print("Columns that are:", ", ".join(map(str, raw.columns)))
        raise SystemExit(1)
    return out


def dist_m(lat1, lon1, lat2, lon2):
    return np.hypot((lat1 - lat2) * 110540.0, (lon1 - lon2) * 111320.0 * np.cos(np.radians(lat1)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    raw = read_export(sys.argv[1])
    cols = pick(raw, NEEDED, True)
    cols.update(pick(raw, OPTIONAL, False))
    print(f"export: {len(raw)} rows, {raw[cols['sid']].nunique()} survey ids")

    e = pd.DataFrame({k: raw[c] for k, c in cols.items()})
    e["sid"] = pd.to_numeric(e.sid, errors="coerce").astype("Int64")
    for k in ("drop_ws", "lat", "lon", "score", "n_culverts"):
        if k in e:
            e[k] = pd.to_numeric(e[k], errors="coerce")
    e["drop_ws"] = e.drop_ws.where(e.drop_ws >= 0)            # -1 is the survey's "not measured" code
    e = e.dropna(subset=["sid", "lat", "lon"])

    d = pd.read_parquet("data/processed/plunkett_features_osm.parquet",
                        columns=["crossing_code", "crossing_survey_id", "state", "aquatic_score", "lat_nad83", "lon_nad83"])
    d["sid"] = d.crossing_survey_id.astype("Int64")
    m = d.merge(e, on="sid", how="left", suffixes=("", "_e"))
    m["match_m"] = dist_m(m.lat_nad83, m.lon_nad83, m.lat, m.lon)
    ok = m.match_m <= MAX_M
    if "score" in m:
        m["score_diff"] = (m.aquatic_score - m.score).abs()
        ok &= m.score_diff < SCORE_TOL
    print(f"Plunkett crossings {len(d)}; with an id in the export {m.groupby('crossing_code').lat.apply(lambda s: s.notna().any()).sum()}; "
          f"rows accepted after the location/score checks {int(ok.sum())}")
    m = m[ok]

    m = m.sort_values(["crossing_code", "drop_ws"])
    g = m.groupby("crossing_code")
    res = pd.DataFrame({"drop_ws": g.drop_ws.min(),
                        "drop_spread_ft": g.drop_ws.agg(lambda s: s.max() - s.min() if s.notna().any() else 0.0),
                        "n_rows": g.size()})
    low = m.dropna(subset=["drop_ws"]).drop_duplicates("crossing_code").set_index("crossing_code")
    res["outlet_grade"] = low["grade"].reindex(res.index) if "grade" in low else np.nan
    res = d[["crossing_code", "state"]].merge(res.reset_index(), on="crossing_code", how="left")
    res["has_drop"] = res.drop_ws.notna()
    print(f"crossings with an outlet drop: {res.has_drop.sum()} of {len(res)} ({res.has_drop.mean():.3f})")
    print(res.groupby("state").has_drop.agg(["size", "sum", "mean"]).round(3).to_string())
    print(f"crossings with several rows: {(res.n_rows > 1).sum()}; pipes differ by more than 0.05 ft: {(res.drop_spread_ft > 0.05).sum()}")

    old = pd.read_parquet("data/processed/plunkett_outlet_drop.parquet")[["crossing_code", "drop_ws"]].rename(columns={"drop_ws": "old"})
    c = res.merge(old, on="crossing_code")
    both = c.drop_ws.notna() & c.old.notna()
    print(f"agreement with the public-layer join where both have a value: {both.sum()} crossings, "
          f"equal {np.isclose(c.drop_ws[both], c.old[both]).mean():.3f}")
    res.drop(columns="state").to_parquet("data/processed/plunkett_outlet_drop_export.parquet", index=False)
    print("saved data/processed/plunkett_outlet_drop_export.parquet")
