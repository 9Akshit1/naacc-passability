"""Tests for reading outlet drop from the UMass export (scripts/build_outlet_drop_umass.py)."""
import numpy as np
import pandas as pd
import pytest

from scripts.build_outlet_drop_umass import build, parse_drop


def test_whole_numbers_are_inches():
    assert parse_drop("0") == (0.0, 0.0, 0.0)
    assert parse_drop("12") == (1.0, 1.0, 1.0)
    assert parse_drop(3) == (0.25, 0.25, 0.25)


def test_range_uses_the_midpoint_and_keeps_the_ends():
    point, low, high = parse_drop("1-6")
    assert point == pytest.approx(3.5 / 12)
    assert low == pytest.approx(1 / 12)
    assert high == pytest.approx(6 / 12)
    assert parse_drop("12-24")[0] == pytest.approx(1.5)


def test_over_24_is_the_lower_end_with_no_upper_end():
    point, low, high = parse_drop(">24")
    assert point == pytest.approx(2.0)
    assert low == pytest.approx(2.0)
    assert np.isnan(high)


@pytest.mark.parametrize("text", ["No data", "nan", "", None, np.nan])
def test_missing_stays_missing(text):
    assert all(np.isnan(v) for v in parse_drop(text))


def _plunkett(codes, states, scores):
    return pd.DataFrame({
        "crossing_code": codes, "crossing_survey_id": range(1, len(codes) + 1), "aquatic_score": scores,
        "database": "umass", "state": states,
    })


def _export(ids, drops, scores):
    return pd.DataFrame({"Crossing_Record_ID": ids, "Outlet_Drop": drops, "AquaticScore_NAACC": scores})


def test_lowest_drop_wins_for_a_crossing_with_two_structures():
    pl = _plunkett(["a"] * 1 + ["b", "c", "d"], ["MA"] * 4, [0.5, 0.6, 0.7, 0.8])
    ex = _export([1, 1, 2, 3, 4], ["12", "6", "0", "0", "0"], [0.5, 0.5, 0.6, 0.7, 0.8])
    out = build(ex, pl)[0].set_index("crossing_code")
    assert out.loc["a", "drop_ws"] == pytest.approx(0.5)


def test_a_score_that_differs_from_the_plunkett_file_stops_the_build():
    pl = _plunkett(["a"], ["MA"], [0.5])
    ex = _export([1], ["0"], [0.9])
    with pytest.raises(SystemExit):
        build(ex, pl)


def test_state_rule_drops_states_without_recorded_zeros_and_non_integer_states():
    codes = [f"x{i}" for i in range(40)]
    states = ["MA"] * 10 + ["NH"] * 10 + ["NJ"] * 10 + ["CT"] * 10
    drops = (["0"] * 5 + ["6"] * 5) + (["1-6"] * 10) + (["0.0833", "0", "0.5", "1.0833", "0", "0", "0.25", "0", "0.1", "0"]) + (["0"] * 4 + ["9"] * 6)
    pl = _plunkett(codes, states, [0.5] * 40)
    ex = _export(range(1, 41), drops, [0.5] * 40)
    out = build(ex, pl)[0].set_index("crossing_code")
    used = out.groupby("state").has_drop.sum().to_dict()
    assert used == {"CT": 10, "MA": 10, "NH": 0, "NJ": 0}


def test_a_range_that_holds_the_three_inch_cut_off_is_flagged():
    pl = _plunkett(["a", "b", "c", "d"], ["MA"] * 4, [0.5] * 4)
    ex = _export([1, 2, 3, 4], ["1-6", "6-12", "0", "0"], [0.5] * 4)
    out = build(ex, pl)[0].set_index("crossing_code")
    assert out.straddles_3in.to_dict() == {"a": True, "b": False, "c": False, "d": False}
