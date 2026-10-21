"""Integrity tests for the multi-state (Plunkett) pipeline.

These guard the properties the reported numbers depend on: no survey-field leakage into
predictors, group-clean cross-validation, a fixed disjoint locked test set, label-free
NBI matching, and neighbour features that cannot see their own row or watershed.
"""
import numpy as np
import pandas as pd
import pytest

from src import nbi, plunkett_cv as cv, plunkett_data as pdm, plunkett_features as pf

HAVE_DATA = pdm.CACHE.exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="tagged Plunkett table not built (python -m src.plunkett_data)")


def test_nbi_coordinate_parsing():
    lat = nbi._dms_to_deg(pd.Series(["41123456"]), 2).iloc[0]
    lon = -nbi._dms_to_deg(pd.Series(["073450682"]), 3).iloc[0]
    assert lat == pytest.approx(41 + 12 / 60 + 34.56 / 3600)
    assert lon == pytest.approx(-(73 + 45 / 60 + 6.82 / 3600))
    assert np.isnan(nbi._dms_to_deg(pd.Series(["41996000"]), 2).iloc[0])   # minutes >= 60 is invalid


def test_nbi_match_is_mutual_nearest_and_radius_limited():
    cross = pd.DataFrame({"lat_nad83": [42.0, 42.0010, 43.0], "lon_nad83": [-73.0, -73.0, -73.0]})
    s = pd.DataFrame({"lat": [42.0001, 42.0014], "lon": [-73.0, -73.0], "nbi_length_m": [10.0, 20.0],
                      "nbi_id": ["a", "b"]})
    m = nbi.match(cross, s, radius_m=60)
    assert m.loc[0, "nbi_length_m"] == 10.0 and m.loc[1, "nbi_length_m"] == 20.0   # each structure used once
    assert m.loc[2, "nbi_matched"] == 0 and np.isnan(m.loc[2, "nbi_length_m"])     # nothing within 60 m


@needs_data
def test_predictors_exclude_survey_fields():
    cols = set(pdm.predictor_columns())
    assert not cols & set(pdm.SURVEY_ONLY_COLUMNS)
    assert pdm.TARGET not in cols


@needs_data
def test_feature_groups_never_contain_state_or_labels():
    d = pd.read_parquet("data/processed/plunkett_features_nbi.parquet")
    feats = sum(pf.feature_groups(d).values(), [])
    banned = {"state", "huc8", "huc6", "huc4", "src_state", "aquatic_score", "bridge", "crossing_type", "nbi_dist_m"}
    assert not banned & set(feats)
    assert not any(c.startswith("nb_") or c.startswith("aux_") for c in feats)


@needs_data
def test_locked_set_is_deterministic_whole_basins():
    d = pdm.load()
    m1, m2 = cv.locked_test_mask(d), cv.locked_test_mask(d)
    assert m1.equals(m2)
    assert 0.13 <= m1.mean() <= 0.20
    assert d.loc[m1, "state"].nunique() >= cv.LOCK_MIN_STATES
    # whole HUC6 basins: no basin is split between locked and development
    assert not set(d.loc[m1, "huc6"]) & set(d.loc[~m1, "huc6"])


@needs_data
@pytest.mark.parametrize("scheme", ["huc8", "huc6", "state"])
def test_cv_splits_are_group_clean(scheme):
    d = pdm.load(); dev = d[~cv.locked_test_mask(d)]
    key = {"huc8": "huc8", "huc6": "huc6", "state": "state"}[scheme]
    seen_test = set()
    for tr, te in cv.splits(dev, scheme, n_splits=5, seed=0):
        assert not set(dev.loc[tr, key]) & set(dev.loc[te, key])
        assert not set(tr) & set(te)
        seen_test |= set(te)
    if scheme != "state":
        assert seen_test == set(dev.index)       # every row is tested exactly once


def test_neighbour_features_exclude_self_and_same_watershed():
    rng = np.random.default_rng(0)
    n = 60
    ref = pd.DataFrame({"_x": rng.uniform(0, 5000, n), "_y": rng.uniform(0, 5000, n),
                        "aquatic_score": rng.uniform(0, 1, n), "bridge": rng.integers(0, 2, n).astype(bool),
                        "huc8": ["A"] * 30 + ["B"] * 30})
    ref.loc[ref.huc8 == "A", "aquatic_score"] = 1.0       # sentinel: watershed A scores are all 1.0
    ref.loc[ref.huc8 == "B", "aquatic_score"] = 0.0
    out = pf.neighbour_features(ref, ref, exclude_same_group="huc8")
    # rows in A may only see B (0.0) and vice versa -> proves same-watershed and self are excluded
    assert (out.loc[ref.huc8 == "A", "nb_mean_k5"] == 0.0).all()
    assert (out.loc[ref.huc8 == "B", "nb_mean_k5"] == 1.0).all()


def test_metrics_mean_predictor_has_zero_r2():
    y = np.array([0.1, 0.5, 0.9, 0.3])
    assert cv.metrics(y, np.full(4, y.mean()))["r2"] == pytest.approx(0.0)
    assert cv.metrics(y, y)["r2"] == pytest.approx(1.0)


def test_final_ensemble_outputs_are_bounded():
    from src.plunkett_final import FinalEnsemble
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(400, 5)), columns=list("abcde"))
    y = pd.Series(np.clip(0.5 + 0.4 * X["a"], 0, 1))
    p = FinalEnsemble().fit(X, y).predict(X)
    assert p.min() >= 0 and p.max() <= 1


def test_osm_pbf_tag_regex_parses_hstore():
    from src.osm_structures import _TAG
    t = dict(_TAG.findall('"bridge"=>"yes","layer"=>"1","bridge:structure"=>"culvert","name"=>"A \\"quoted\\" road"'))
    assert t["bridge"] == "yes" and t["layer"] == "1" and t["bridge:structure"] == "culvert"
    assert "name" in t


def test_osm_feature_columns_have_no_distances_or_state():
    d = pd.read_parquet("data/processed/plunkett_features_osm.parquet") if HAVE_DATA and \
        __import__("os").path.exists("data/processed/plunkett_features_osm.parquet") else None
    if d is None:
        pytest.skip("OSM feature table not built")
    osm_cols = pf.feature_groups(d)["OSM"]
    assert osm_cols and not any(c.endswith("_dist_m") for c in osm_cols)
    assert not {"state", "huc8"} & set(osm_cols)


@needs_data
def test_group_splitter_seeds_give_different_splits():
    """Regression test for the 2026-10-08 bug: the older splitter returned (almost) the same split for every seed."""
    d = pdm.load(); dev = d[~cv.locked_test_mask(d)]
    def folds(seed):
        f = pd.Series(-1, index=dev.index)
        for k, (_, te) in enumerate(cv.splits(dev, "huc8", n_splits=5, seed=seed)):
            f.loc[te] = k
        return f
    ct = pd.crosstab(folds(0), folds(1))
    assert ct.max(axis=1).sum() / ct.values.sum() < 0.8      # near 1.0 would mean the seeds do not matter
    sizes = [(folds(0) == k).sum() for k in range(5)]
    assert min(sizes) > 0.5 * max(sizes)                      # folds stay roughly balanced
