import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import naacc_score
from src.gis_predictors import (_polygon_area_km2, _haversine_km, _classify_osm_tags,
                                _aggregate_streamcat_record, _clean_vaa_slope, _offset_latlon,
                                _clean_structure_condition)
from src.evaluate import regression_metrics, prioritization_overlap
from src.spatial_cv import huc8_group_kfold, random_kfold, repeated_huc8_group_kfold
from src.uncertainty import conformal_intervals, evaluate_coverage

RAW = ROOT / "data/raw/ny_naacc_surveys.csv"
SUBSET = ROOT / "data/processed/subset_predictors.parquet"


# ---------------------------------------------------------------------------
# NAACC score reconstruction (pure computation - always runs)
# ---------------------------------------------------------------------------


class TestNaaccScoreFormulas:
    def test_outlet_drop_monotone_decreasing(self):
        s = naacc_score.outlet_drop_score(pd.Series([0.0, 0.3, 0.6, 1.0, 3.0]))
        assert s.iloc[0] == pytest.approx(1.0)
        assert (s.diff().dropna() <= 1e-9).all()
        assert s.iloc[-1] < 0.2

    def test_openness_monotone_increasing_bounded(self):
        s = naacc_score.openness_score(pd.Series([0.0, 0.05, 0.16, 0.5, 2.0]))
        assert s.iloc[0] == pytest.approx(0.0)
        assert (s.diff().dropna() >= -1e-9).all()
        assert s.max() <= 1.0

    def test_height_score_bounded(self):
        s = naacc_score.height_score(pd.Series([0.0, 1.0, 3.0, 10.0]))
        assert s.min() >= 0 and s.max() <= 1.0

    def test_weights_sum_to_one(self):
        assert sum(naacc_score.WEIGHTS.values()) == pytest.approx(1.0, abs=1e-6)

    def test_composite_in_range_on_synthetic(self):
        df = pd.DataFrame({
            "Outlet_Openness": [0.16, 0.5], "Outlet_Height": [3.0, 5.0],
            "Outlet_Drop_To_Water_Surface": [0.0, 1.5], "Outlet_Drop_To_Stream_Bottom": [0.0, 2.0],
            "Crossing_Span": ["Moderate", "Spans Full Channel & Banks"],
            "Inlet_Grade": ["At Stream Grade", "Perched"],
            "Internal_Structure": [None, None], "Armoring": [None, "Extensive"],
            "Barrier_Severity": [None, "Severe"], "Scour_Pool": ["None", "Large"],
            "Substrate_Continuous": ["100%", "None"],
            "Structure_Substrate_Matches_Stream": ["Comparable", "None"],
            "Water_Depth_Matches_Stream": ["Yes", "No-Shallower"],
            "Water_Velocity": ["Yes", "No-Faster"], "Outlet_Grade": ["At Stream Grade", "Free Fall"],
        })
        res = naacc_score.compute_passability(df)
        assert res.final.between(0, 1).all()
        assert res.final.iloc[0] > res.final.iloc[1]  # first crossing is clearly better


@pytest.mark.skipif(not RAW.exists(), reason="raw NAACC data not downloaded")
class TestAgainstPublishedScore:
    def test_reconstruction_tracks_published(self):
        from src.data import build_modeling_dataset
        m, _ = build_modeling_dataset()
        v = naacc_score.validate_against_published(m)
        assert v["n"] > 20000
        assert v["pearson_r"] > 0.80         # documented as an approximation, not exact
        assert v["within_0.20"] > 0.90
        assert abs(v["bias"]) < 0.05


@pytest.mark.skipif(not RAW.exists(), reason="raw NAACC data not downloaded")
class TestCleaning:
    def test_modeling_dataset_shape_and_integrity(self):
        from src.data import build_modeling_dataset
        m, rep = build_modeling_dataset()
        assert 20000 < len(m) < 40000
        assert m["crossing_id"].is_unique
        assert m["passability"].between(0, 1).all()
        assert m["huc8_name"].nunique() > 20
        assert m["lat"].between(40, 46).all()
        # every drop step is accounted for
        assert all(b >= a for _, b, a in rep.steps)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_polygon_area_unit_square():
    # ~0.1 deg square near 43N: ~ (0.1*111) x (0.1*111*cos43) km
    ring = [[-76.0, 43.0], [-75.9, 43.0], [-75.9, 43.1], [-76.0, 43.1], [-76.0, 43.0]]
    a = _polygon_area_km2(ring)
    assert 60 < a < 100


def test_haversine_known_distance():
    d = _haversine_km(42.0, -76.0, 43.0, -76.0)
    assert d == pytest.approx(111.2, abs=1.0)


# ---------------------------------------------------------------------------
# OSM road-class crosswalk (pure function, no network - see gis_predictors.py)
# ---------------------------------------------------------------------------


def test_classify_osm_tags_railway_takes_priority():
    assert _classify_osm_tags({"railway": "rail", "highway": "residential"}) == "railroad"


def test_classify_osm_tags_driveway():
    assert _classify_osm_tags({"highway": "service", "service": "driveway"}) == "driveway"


def test_classify_osm_tags_trail():
    assert _classify_osm_tags({"highway": "footway"}) == "trail"
    assert _classify_osm_tags({"highway": "path"}) == "trail"


def test_classify_osm_tags_multilane_from_lanes_tag():
    assert _classify_osm_tags({"highway": "secondary", "lanes": "4"}) == "multilane road (>2 lanes)"


def test_classify_osm_tags_multilane_from_motorway_default():
    assert _classify_osm_tags({"highway": "motorway"}) == "multilane road (>2 lanes)"


def test_classify_osm_tags_unpaved_surface():
    assert _classify_osm_tags({"highway": "unclassified", "surface": "gravel"}) == "unpaved"


def test_classify_osm_tags_paved_surface():
    assert _classify_osm_tags({"highway": "residential", "surface": "asphalt"}) == "paved"


def test_classify_osm_tags_paved_default_when_no_surface_tag():
    assert _classify_osm_tags({"highway": "tertiary"}) == "paved"


def test_classify_osm_tags_unknown_when_no_tags():
    assert _classify_osm_tags({}) == "unknown"


# StreamCat record -> engineered sc_* features (pure function, no network - see gis_predictors.py)


def test_aggregate_streamcat_record_passthrough_metrics():
    rec = {"precip8110ws": 1000.0, "bfiws": 50.0}
    out = _aggregate_streamcat_record(rec)
    assert out["sc_precip_mm"] == 1000.0
    assert out["sc_baseflow_idx"] == 50.0


def test_aggregate_streamcat_record_sums_landcover_subclasses():
    rec = {"pctdecid2019ws": 10.0, "pctconif2019ws": 5.0, "pctmxfst2019ws": 2.0}
    out = _aggregate_streamcat_record(rec)
    assert out["sc_pct_forest"] == pytest.approx(17.0)


def test_aggregate_streamcat_record_missing_subclass_counts_as_zero():
    # only some of the forest sub-classes present - the rest should count as
    # 0%, not make the whole composite NaN.
    rec = {"pctdecid2019ws": 10.0}
    out = _aggregate_streamcat_record(rec)
    assert out["sc_pct_forest"] == pytest.approx(10.0)


def test_aggregate_streamcat_record_all_missing_is_nan():
    out = _aggregate_streamcat_record({})
    assert np.isnan(out["sc_pct_forest"])
    assert out["sc_precip_mm"] is None


# NHDPlus VAA slope sentinel handling (pure function, no network - see gis_predictors.py)


def test_clean_vaa_slope_passes_through_valid_value():
    assert _clean_vaa_slope(0.0234) == pytest.approx(0.0234)


def test_clean_vaa_slope_treats_sentinel_as_missing():
    assert _clean_vaa_slope(-9998.0) is None


def test_clean_vaa_slope_treats_none_and_nan_as_missing():
    assert _clean_vaa_slope(None) is None
    assert _clean_vaa_slope(float("nan")) is None


def test_clean_vaa_slope_zero_is_valid():
    assert _clean_vaa_slope(0.0) == 0.0


# _offset_latlon (pure function, no network - see gis_predictors.py)


def test_offset_latlon_north_moves_lat_only():
    lat, lon = 43.0, -75.0
    olat, olon = _offset_latlon(lat, lon, 0.0, 100.0)
    assert olat > lat
    assert olon == pytest.approx(lon, abs=1e-9)


def test_offset_latlon_east_moves_lon_only():
    lat, lon = 43.0, -75.0
    olat, olon = _offset_latlon(lat, lon, 100.0, 0.0)
    assert olon > lon
    assert olat == pytest.approx(lat, abs=1e-9)


def test_offset_latlon_round_trip_distance_is_accurate():
    """100m offset then haversine back should recover ~100m, within the
    known error of the small-offset spherical approximation (<1%)."""
    lat, lon = 40.0, -74.0
    olat, olon = _offset_latlon(lat, lon, 0.0, 100.0)
    d_km = _haversine_km(lat, lon, olat, olon)
    assert d_km * 1000 == pytest.approx(100.0, rel=0.01)


def test_offset_latlon_longitude_degrees_shrink_at_high_latitude():
    """At a fixed east-west metre offset, the same distance should be a
    bigger longitude-degree change near the pole than near the equator
    (cos(lat) shrinks the denominator) - catches a sign/formula error."""
    _, lon_lo = _offset_latlon(5.0, -75.0, 1000.0, 0.0)
    _, lon_hi = _offset_latlon(70.0, -75.0, 1000.0, 0.0)
    assert (lon_hi - (-75.0)) > (lon_lo - (-75.0))


# _clean_structure_condition (pure function, no network - see gis_predictors.py)


def test_clean_structure_condition_passes_through_valid_value():
    assert _clean_structure_condition(5.5) == pytest.approx(5.5)


def test_clean_structure_condition_treats_zero_sentinel_as_missing():
    assert _clean_structure_condition(0.0) is None


def test_clean_structure_condition_treats_none_and_nan_as_missing():
    assert _clean_structure_condition(None) is None
    assert _clean_structure_condition(float("nan")) is None


def test_clean_structure_condition_one_is_valid():
    assert _clean_structure_condition(1.0) == 1.0


def test_regression_metrics_sane():
    y = np.linspace(0, 1, 200)
    m = regression_metrics(y, y)
    assert m["mae"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    m2 = regression_metrics(y, np.full_like(y, y.mean()))
    assert m2["r2"] == pytest.approx(0.0, abs=1e-9)
    assert m2["pred_sd_ratio"] == pytest.approx(0.0)


def test_spatial_split_holds_groups_together():
    df = pd.DataFrame({"huc8_name": (["A"] * 50 + ["B"] * 50 + ["C"] * 50 + ["D"] * 50),
                       "passability": np.random.rand(200)})
    for tr, te in huc8_group_kfold(df, n_splits=4):
        train_g = set(df.loc[tr, "huc8_name"])
        test_g = set(df.loc[te, "huc8_name"])
        assert train_g.isdisjoint(test_g)


def test_spatial_split_every_row_covered_exactly_once():
    df = pd.DataFrame({"huc8_name": (["A"] * 50 + ["B"] * 50 + ["C"] * 50 + ["D"] * 50),
                       "passability": np.random.rand(200)})
    seen = []
    for _tr, te in huc8_group_kfold(df, n_splits=4, seed=3):
        seen.extend(te.tolist())
    assert sorted(seen) == list(df.index)


def test_spatial_split_seed_changes_the_split():
    df = pd.DataFrame({"huc8_name": (list("ABCDEFGH") * 25), "passability": np.random.rand(200)})
    splits_a = [set(te) for _tr, te in huc8_group_kfold(df, n_splits=3, seed=0)]
    splits_b = [set(te) for _tr, te in huc8_group_kfold(df, n_splits=3, seed=1)]
    assert splits_a != splits_b


def test_repeated_spatial_kfold_yields_n_repeats_of_full_splits():
    df = pd.DataFrame({"huc8_name": (list("ABCDEFGH") * 25), "passability": np.random.rand(200)})
    repeats = list(repeated_huc8_group_kfold(df, n_splits=3, n_repeats=4, seed=0))
    assert len(repeats) == 4
    for splits in repeats:
        seen = []
        for tr, te in splits:
            assert set(df.loc[tr, "huc8_name"]).isdisjoint(df.loc[te, "huc8_name"])
            seen.extend(te.tolist())
        assert sorted(seen) == list(df.index)


def test_conformal_coverage_on_synthetic():
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=(n, 2))
    y = np.clip(0.5 + 0.2 * x[:, 0] + rng.normal(scale=0.1, size=n), 0, 1)
    df = pd.DataFrame({"lat": x[:, 0], "lon": x[:, 1], "road_class": "paved",
                       "log_drainage_area": x[:, 0], "drainage_area_km2": np.abs(x[:, 0]),
                       "basin_span_km": np.abs(x[:, 1]), "elev_m": x[:, 1] * 10,
                       "county_income": 60000, "county_pop_density": 50,
                       "sc_precip_mm": 1000, "sc_runoff_mm": 500, "sc_baseflow_idx": 50,
                       "sc_road_density": 2, "sc_road_stream_crossing_density": 0.02,
                       "sc_dam_density": 0, "sc_pop_density": 30, "sc_bedrock_depth_cm": 120,
                       "sc_soil_perm": 5, "sc_wetness_index": 750, "sc_pct_impervious": 2,
                       "sc_pct_forest": 40, "sc_pct_wetland": 3, "sc_pct_agriculture": 30,
                       "stream_slope": 0.02, "stream_order": 2,
                       "aadt": 500, "log_aadt": np.log10(500),
                       "terrain_slope_100m": 0.05, "terrain_rough_100m": 5,
                       "terrain_slope_500m": 0.03, "terrain_rough_500m": 15,
                       "ecoregion": "Northeastern Highlands",
                       "structure_age_yr": 40, "structure_condition": 5,
                       "structure_length_m": 10, "structure_material": "1 - Concrete",
                       "structure_is_bridge": "bridge",
                       "structure_superstructure_condition": 6,
                       "structure_fhwa_status": "Fair",
                       "structure_opening_width_m": 4.5, "structure_num_spans": 1,
                       "structure_culvert_skew_deg": 0, "structure_streambed_material": "none",
                       "constriction_ratio": 0.8, "precip_24h_10yr_in": 4.0,
                       "sc_local_catchment_area_km2": 5, "sc_road_density_local": 2,
                       "sc_road_stream_crossing_density_local": 0.02,
                       "sc_bedrock_depth_cm_local": 100, "sc_soil_perm_local": 5,
                       "sc_wetness_index_local": 750, "sc_pct_impervious_local": 5,
                       "sc_pct_forest_local": 40, "sc_pct_wetland_local": 3,
                       "sc_pct_agriculture_local": 20,
                       "passability": y})
    from src.models import make_models
    model = make_models()["ridge"]
    idx = df.index.to_numpy()
    iv = conformal_intervals(model, df, idx[:400], idx[400:500], idx[500:], "passability", alpha=0.1)
    cov = evaluate_coverage(iv, df["passability"])["empirical_coverage"]
    assert cov >= 0.80  # ~90% target, allow slack on a small sample


@pytest.mark.skipif(not SUBSET.exists(), reason="subset predictors not built")
def test_subset_predictors_plausible():
    d = pd.read_parquet(SUBSET)
    assert len(d) > 500
    da = d["drainage_area_km2"].dropna()
    # Upper bound covers the Hudson River mainstem / Lake Champlain outlet
    # (Richelieu River) crossings included once the subset expanded past the
    # original 16-HUC8 whitelist (which excluded those mainstem/coastal
    # watersheds) - their real drainage areas run up to ~33,000 km2.
    assert da.between(0.01, 35000).all()
    assert d["county_income"].dropna().between(20000, 250000).all()
    if "elev_m" in d:
        assert d["elev_m"].dropna().between(-10, 2000).all()


@pytest.mark.skipif(not SUBSET.exists(), reason="subset predictors not built")
def test_predict_crossings_end_to_end(monkeypatch):
    """Regression test for a real bug found 2026-09-27: predict_crossings
    crashed on every call (`DataFrame.setdefault()` doesn't exist - that's a
    dict method) and, once that was fixed, would have crashed again inside
    the model's ColumnTransformer because build_predictor_table was only
    fetching drainage/elevation/road_class, not the sc_*/stream_slope/aadt/
    terrain_*/ecoregion/structure_* columns the trained model actually
    needs. Neither bug had any test coverage. This test monkeypatches
    build_predictor_table to return real rows from the already-built subset
    (no live network calls, no per-row rate limiting) so it stays fast, but
    exercises the exact code path (County handling, DataFrame construction,
    model.predict on the full feature set) that broke."""
    import src.predict as predict_mod

    real = pd.read_parquet(SUBSET).head(2).reset_index(drop=True)
    monkeypatch.setattr(predict_mod, "build_predictor_table", lambda df, **kw: real)

    crossings = pd.DataFrame({"crossing_id": real["crossing_id"], "lat": real["lat"], "lon": real["lon"]})
    out = predict_mod.predict_crossings(crossings, model_name="hist_gbm")

    assert len(out) == 2
    assert out["predicted_passability"].between(0, 1).all()
    assert (out["lower_bound"] <= out["predicted_passability"]).all()
    assert (out["upper_bound"] >= out["predicted_passability"]).all()


def test_save_load_model_roundtrip(tmp_path, monkeypatch):
    """A saved model should give identical predictions to the model it was
    saved from, and predict_crossings should pick it up automatically
    instead of retraining when one exists at the default path."""
    import src.predict as predict_mod

    real = pd.read_parquet(SUBSET).head(2).reset_index(drop=True)
    monkeypatch.setattr(predict_mod, "build_predictor_table", lambda df, **kw: real)
    crossings = pd.DataFrame({"crossing_id": real["crossing_id"], "lat": real["lat"], "lon": real["lon"]})

    model, _, calib_res = predict_mod.train_final_model("hist_gbm")
    saved_path = predict_mod.save_final_model(model, calib_res, "hist_gbm", tmp_path / "hist_gbm.joblib")
    assert saved_path.exists()

    loaded = predict_mod.load_final_model("hist_gbm", saved_path)
    np.testing.assert_array_equal(loaded["model"].predict(real), model.predict(real))

    out = predict_mod.predict_crossings(crossings, model_name="hist_gbm", model_path=saved_path)
    assert len(out) == 2

    monkeypatch.setattr(predict_mod, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(predict_mod, "train_final_model",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not retrain")))
    out2 = predict_mod.predict_crossings(crossings, model_name="hist_gbm")
    assert len(out2) == 2
