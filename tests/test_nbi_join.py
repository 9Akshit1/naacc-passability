"""NBI download/join tool, tested on a small made-up state file (no network)."""
import numpy as np
import pandas as pd
import pytest

from src import nbi

COLS = list(nbi._KEEP)


def dms(deg, minutes, seconds, width):
    """NBI position string: degrees, minutes, then seconds with two implied decimals."""
    return f"{deg:0{width}d}{minutes:02d}{int(round(seconds * 100)):04d}"


def state_file(folder, rows, state="NY", yy="25"):
    df = pd.DataFrame("", index=range(len(rows)), columns=COLS)
    for i, r in enumerate(rows):
        for k, v in r.items():
            df.loc[i, k] = v
    df.to_csv(folder / f"{state}{yy}.txt", index=False)


def structure(num, lat, lon, service="5", length="30.5", year="1990", **extra):
    base = {"STRUCTURE_NUMBER_008": num, "LAT_016": lat, "LONG_017": lon, "SERVICE_UND_042B": service,
            "STRUCTURE_LEN_MT_049": length, "YEAR_BUILT_027": year, "STRUCTURE_TYPE_043B": "02"}
    base.update(extra)
    return base


# 42 30 00.00 N, 075 00 00.00 W
LAT0, LON0 = 42.5, -75.0
LAT_S, LON_S = dms(42, 30, 0, 2), dms(75, 0, 0, 3)


def crossings(*offsets_m):
    """Crossings placed due north of the point, ``offsets_m`` metres away."""
    return pd.DataFrame({"cid": [f"c{i}" for i in range(len(offsets_m))],
                         "lat_nad83": [LAT0 + o / 111_190.0 for o in offsets_m],
                         "lon_nad83": [LON0] * len(offsets_m)})


def test_position_decoding(tmp_path):
    state_file(tmp_path, [structure("A", LAT_S, LON_S)])
    s = nbi.load_nbi(["NY"], 2025, tmp_path)
    assert s.loc[0, "lat"] == pytest.approx(42.5, abs=1e-9)
    assert s.loc[0, "lon"] == pytest.approx(-75.0, abs=1e-9)


def test_only_waterway_structures_with_valid_positions_are_kept(tmp_path):
    state_file(tmp_path, [
        structure("keep", LAT_S, LON_S, service="5"),
        structure("highway_only", LAT_S, LON_S, service="1"),
        structure("seconds_77", dms(42, 30, 77.0, 2), LON_S),
        structure("minutes_75", "42750000", LON_S),
    ])
    s = nbi.load_nbi(["NY"], 2025, tmp_path)
    assert list(s["nbi_id"]) == ["keep"]


def test_sentinels_become_missing(tmp_path):
    state_file(tmp_path, [structure("A", LAT_S, LON_S, year="0", length="0")])
    s = nbi.load_nbi(["NY"], 2025, tmp_path)
    assert np.isnan(s.loc[0, "nbi_year_built"]) and np.isnan(s.loc[0, "nbi_length_m"])


def test_match_radius_and_distance(tmp_path):
    state_file(tmp_path, [structure("A", LAT_S, LON_S)])
    s = nbi.load_nbi(["NY"], 2025, tmp_path)
    m = nbi.match(crossings(10), s, 60.0)
    assert m.loc[0, "nbi_matched"] == 1 and m.loc[0, "nbi_dist_m"] == pytest.approx(10, abs=1.0)
    far = nbi.match(crossings(200), s, 60.0)
    assert far.loc[0, "nbi_matched"] == 0 and np.isnan(far.loc[0, "nbi_length_m"])


def test_mutual_nearest_gives_a_structure_to_one_crossing_only(tmp_path):
    state_file(tmp_path, [structure("A", LAT_S, LON_S)])
    s = nbi.load_nbi(["NY"], 2025, tmp_path)
    c = crossings(5, 30)                         # both within 60 m of the one structure
    mutual = nbi.match(c, s, 60.0)
    assert list(mutual["nbi_matched"]) == [1, 0]  # the nearer crossing wins
    one_way = nbi.match(c, s, 60.0, mutual_only=False)
    assert list(one_way["nbi_matched"]) == [1, 1]


def test_cli_join_end_to_end(tmp_path):
    state_file(tmp_path, [structure("A", LAT_S, LON_S, length="42.0"), structure("far", dms(43, 0, 0, 2), LON_S)])
    c = crossings(8, 5000)
    c.to_csv(tmp_path / "x.csv", index=False)
    out = tmp_path / "out.csv"
    nbi.main(["join", "--crossings", str(tmp_path / "x.csv"), "--id-col", "cid", "--dir", str(tmp_path), "--out", str(out)])
    r = pd.read_csv(out)
    assert list(r["cid"]) == ["c0", "c1"]
    assert list(r["nbi_matched"]) == [1, 0]
    assert r.loc[0, "nbi_structure_number"] == "A" and r.loc[0, "nbi_length_m"] == 42.0
    assert pd.isna(r.loc[1, "nbi_structure_number"])


def test_cli_join_reports_a_missing_column_and_missing_files(tmp_path):
    state_file(tmp_path, [structure("A", LAT_S, LON_S)])
    crossings(5).to_csv(tmp_path / "x.csv", index=False)
    with pytest.raises(SystemExit, match="not found"):
        nbi.main(["join", "--crossings", str(tmp_path / "x.csv"), "--dir", str(tmp_path), "--out", str(tmp_path / "o.csv")])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no NBI files"):
        nbi.main(["join", "--crossings", str(tmp_path / "x.csv"), "--id-col", "cid", "--dir", str(empty), "--out", str(tmp_path / "o.csv")])


def test_the_joined_columns_are_the_ones_the_model_used(tmp_path):
    """The CLI must not drift from the feature set behind the published numbers."""
    state_file(tmp_path, [structure("A", LAT_S, LON_S)])
    s = nbi.load_nbi(["NY"], 2025, tmp_path)
    m = nbi.match(crossings(5), s)
    cols = {c for c in m.columns if c.startswith("nbi_")}
    assert {"nbi_matched", "nbi_dist_m", "nbi_length_m", "nbi_max_span_m", "nbi_deck_width_m"} <= cols
    assert not any(c.startswith("src_") for c in m.columns)
