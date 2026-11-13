"""Read outlet drop for the older UMass crossings from NAACC's UMass detailed export.

About 5,400 of the 56,525 crossings in Ethan Plunkett's file come from the UMass Stream Continuity project (2005-2017); the
column `database` is "umass" for them. Their scores are the ones stored in the NAACC database, not scores worked out from the
outlet drop (Ethan Plunkett, email 2026-11-12), so the outlet drop is not the same kind of input as it is for the current
NAACC surveys.

The UMass export stores Outlet_Drop as text: whole numbers up to 99, and four ranges, 1-6, 6-12, 12-24 and >24. These are
inches. The export does not say so, but the score of single-structure crossings falls smoothly with the number (0.79 at 0,
0.58 at 1, 0.18 at 12, 0.03 at 24) and the four ranges land where their midpoints would put them. Ethan suggested using the
midpoint of a range. Here:

  whole number n         n inches
  range a-b              midpoint (a+b)/2 inches, and the range ends are kept as drop_lo_ft and drop_hi_ft
  >24                    24 inches, which is the lower end; drop_hi_ft is left empty
  empty or "No data"     no value, and nothing is filled in

Two groups of states are left out, because their numbers do not mean the same thing:
  New Jersey             its drops are not whole numbers (0.0833, 1.0833 and so on) and the score does not move with them, so the
                         unit is not inches and is not known
  New Hampshire, Vermont a drop is recorded only for outlets that have one (none of the recorded values is 0, against 52 to 88
                         percent in the states that are used), so a missing value there means "no drop" more often than "not measured"
The rule in the code is: a state is used when its recorded values include zeros (10 percent or more) and all of them are whole
numbers or ranges. For the other states the drop is left empty and the reason is in the column `skipped`.

A crossing with several structures gets the lowest drop, the same rule as for the current surveys, because the score follows
the most passable structure. Rows are matched to Ethan's file by Crossing_Record_ID = crossing_survey_id. The score in the
export (AquaticScore_NAACC) must equal his aquatic_score, or the script stops.

Usage (from the repository root):
    python -m scripts.build_outlet_drop_umass path/to/umass_detailed_export.csv

Output: data/processed/plunkett_outlet_drop_umass.parquet. The raw export is not committed.
"""
import re
import sys

import numpy as np
import pandas as pd

RANGE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$")
OVER = re.compile(r"^\s*>\s*(\d+(?:\.\d+)?)\s*$")
PERCHED_FT = 0.25


def parse_drop(text):
    """Return (point, low, high) in feet for one Outlet_Drop cell, or (nan, nan, nan)."""
    s = str(text).strip()
    try:
        v = float(s)
        return v / 12, v / 12, v / 12
    except ValueError:
        pass
    m = RANGE.match(s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (a + b) / 24, a / 12, b / 12
    m = OVER.match(s)
    if m:
        a = float(m.group(1)) / 12
        return a, a, np.nan
    return np.nan, np.nan, np.nan


def state_rule(res_state):
    """res_state has columns state, drop_ws, whole (True when the recorded value is a whole number or a range)."""
    rec = res_state.dropna(subset=["drop_ws"])
    zero = rec.groupby("state").drop_ws.apply(lambda s: (s == 0).mean())
    whole = rec.groupby("state").whole.all()
    ok = (zero >= 0.10) & whole
    return ok, zero, whole


def build(export, plunkett):
    umass = plunkett[plunkett.database == "umass"][["crossing_code", "crossing_survey_id", "aquatic_score"]]
    e = export.merge(umass, left_on="Crossing_Record_ID", right_on="crossing_survey_id")
    off = (e.AquaticScore_NAACC - e.aquatic_score).abs()
    if off.max() > 0.002:
        raise SystemExit(f"{int((off > 0.002).sum())} rows have a score that differs from Ethan's file; stopping")
    p = e.Outlet_Drop.map(parse_drop)
    e["pt"], e["lo"], e["hi"] = (p.map(lambda t: t[i]) for i in range(3))
    e["is_range"] = e.Outlet_Drop.astype(str).str.contains(r"[->]")
    num = pd.to_numeric(e.Outlet_Drop, errors="coerce")
    e["whole"] = e.is_range | (num % 1 == 0) | num.isna()
    e = e.dropna(subset=["pt"]).sort_values(["crossing_code", "pt"])
    low = e.drop_duplicates("crossing_code").set_index("crossing_code")
    out = pd.DataFrame({
        "drop_ws": low.pt,
        "drop_lo_ft": low.lo,
        "drop_hi_ft": low.hi,
        "from_range": low.is_range,
        "whole": e.groupby("crossing_code").whole.all(),
        "n_rows": e.groupby("crossing_code").size(),
    })
    out = umass[["crossing_code"]].merge(out.reset_index(), on="crossing_code", how="left")
    out = out.merge(plunkett[["crossing_code", "state"]], on="crossing_code")
    ok, zero, whole = state_rule(out)
    out["skipped"] = np.where(out.state.map(ok).fillna(False), "", "state rule")
    cut = out.skipped != ""
    out.loc[cut, ["drop_ws", "drop_lo_ft", "drop_hi_ft"]] = np.nan
    out["has_drop"] = out.drop_ws.notna()
    out["from_range"] = out.from_range.eq(True)
    # a range whose ends fall on both sides of 3 inches cannot say whether the outlet is perched
    out["straddles_3in"] = out.from_range & out.drop_lo_ft.lt(PERCHED_FT) & out.drop_hi_ft.gt(PERCHED_FT)
    return out.drop(columns="whole"), len(umass), len(e), zero, ok


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    raw = pd.read_csv(sys.argv[1], encoding="cp1252", low_memory=False)
    pl = pd.read_parquet("data/processed/plunkett_features_osm.parquet", columns=["crossing_code", "crossing_survey_id", "aquatic_score", "database", "state"])
    res, n_umass, n_rows, zero, ok = build(raw, pl)
    print(f"UMass crossings in Ethan's file: {n_umass}; export rows used for them: {n_rows}")
    print("share of recorded drops equal to 0, and whether the state is used:")
    print(pd.DataFrame({"zero_share": zero.round(3), "used": ok}).to_string())
    print(f"with an outlet drop after the state rule: {int(res.has_drop.sum())} ({res.has_drop.mean():.1%})")
    print(f"drop given as a range: {int(res.from_range.sum())}; of those, the range includes the 3 inch cut-off: {int(res.straddles_3in.sum())}")
    print(res.groupby("state").has_drop.agg(["size", "sum", "mean"]).round(3).to_string())
    res.drop(columns="state").to_parquet("data/processed/plunkett_outlet_drop_umass.parquet", index=False)
    print("saved data/processed/plunkett_outlet_drop_umass.parquet")
