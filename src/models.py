"""The models compared for the New York version, from a mean-only baseline through linear models to tree ensembles and
gradient boosting. Every model is a scikit-learn Pipeline that fills in missing values, scales the inputs for the linear models
and one-hot encodes road_class.
"""

from __future__ import annotations

import os

# joblib/loky's Windows core-count detection shells out to `wmic`, which
# stalls intermittently on this machine - set this before any sklearn/
# joblib import to skip it (joblib's own suggested fix).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

# joblib's default process backend (loky) is unreliable here above
# n_jobs=1 - full runs hung for hours and left orphaned processes. The
# thread backend is reliable and no slower, since these tree estimators
# release the GIL in their C/Cython loops. Callers wrap their entry point in
# joblib.parallel_backend("threading", n_jobs=_N_JOBS).
_N_JOBS = 6

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor, VotingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:  # noqa
    _HAS_XGB = False


NUMERIC_FEATURES_DEFAULT = [
    "log_drainage_area", "drainage_area_km2", "basin_span_km", "elev_m",
    "county_income", "county_pop_density", "lat", "lon",
    # EPA StreamCat upstream-watershed metrics - see gis_predictors.py::
    # _STREAMCAT_METRICS for the rationale behind each.
    "sc_precip_mm", "sc_runoff_mm", "sc_baseflow_idx", "sc_road_density",
    "sc_road_stream_crossing_density", "sc_dam_density", "sc_pop_density",
    "sc_bedrock_depth_cm", "sc_soil_perm", "sc_wetness_index",
    "sc_pct_impervious", "sc_pct_forest", "sc_pct_wetland", "sc_pct_agriculture",
    # The local-catchment StreamCat measures (sc_*_local) are fetched and cached but left out here. They correlate 0.82 to
    # 0.91 with the watershed-scale versions above and spread the permutation importance across features without
    # changing R2.
    "stream_slope", "stream_order",
    "aadt", "log_aadt",
    "terrain_slope_100m", "terrain_rough_100m",
    "terrain_slope_500m", "terrain_rough_500m",
    # New York DOT structure inventory (fetch_structure_inventory_bulk): age, condition and length of bridges and large
    # culverts, matched within 100 m for about 21% of crossings. NAACC's own score is built from structure properties and
    # not from the landscape.
    "structure_age_yr", "structure_condition", "structure_length_m",
    "structure_superstructure_condition",
    "structure_opening_width_m", "structure_num_spans", "structure_culvert_skew_deg",
    "precip_24h_10yr_in",
    # Tried and left out because they were flat or negative: constriction_ratio, structure_yrs_since_inspection,
    # precip_60m_10yr_in and precip_flashiness_ratio.
]
CATEGORICAL_FEATURES_DEFAULT = [
    "road_class",
    # EPA Level III ecoregion, which lets the model account for regional differences (not a DSL predictor); see
    # gis_predictors.py::fetch_ecoregion_bulk.
    "ecoregion",
    # New York DOT structure categories, "none" where unmatched.
    "structure_material", "structure_is_bridge",
    "structure_fhwa_status",
    "structure_streambed_material",
    # Tried and left out (flat, within noise): structure_type.
]


def _preprocessor(numeric, categorical, scale: bool):
    # add_indicator=True adds one 0/1 column per numeric feature that has missing values, saying whether the value was
    # observed, next to the median-imputed value. On its own this made no difference in repeated spatial CV (N=5), but with
    # the voting ensemble it gave a small gain with less variation between splits (see make_models). Most missing structure
    # fields (about 79% unmatched) are already flagged by structure_is_bridge == "none", but AADT (58% coverage) and
    # terrain (98%) had no such flag.
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]), categorical),
    ])


def make_models(numeric=None, categorical=None, seed: int = 0,
                tuned_params: dict | None = None) -> dict[str, Pipeline]:
    """``tuned_params``: optional ``{model_name: {"est__param": value, ...}}``
    from ``tune_hyperparameters()`` - overrides the hand-picked defaults below
    for whichever model names it contains via ``Pipeline.set_params()``.

    ``voting_ensemble``: a plain equal-weight average
    (``sklearn.ensemble.VotingRegressor``) of the four tuned tree/boosting
    models, built from the same pipeline objects after ``tuned_params`` is
    applied. The ensemble-vs-single-model comparison is in the E0 results of
    model_experiments.py."""
    numeric = numeric or NUMERIC_FEATURES_DEFAULT
    categorical = categorical or CATEGORICAL_FEATURES_DEFAULT
    tuned_params = tuned_params or {}

    def pipe(est, scale=False):
        return Pipeline([("prep", _preprocessor(numeric, categorical, scale)), ("est", est)])

    models = {
        "mean": pipe(DummyRegressor(strategy="mean")),
        "linear": pipe(LinearRegression(), scale=True),
        "ridge": pipe(RidgeCV(alphas=np.logspace(-3, 3, 25)), scale=True),
        "elasticnet": pipe(ElasticNetCV(l1_ratio=[.1, .5, .9, 1.0], n_alphas=50,
                                        random_state=seed, max_iter=5000), scale=True),
        "random_forest": pipe(RandomForestRegressor(
            n_estimators=500, min_samples_leaf=3, n_jobs=_N_JOBS, random_state=seed)),
        "extra_trees": pipe(ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=3, n_jobs=_N_JOBS, random_state=seed)),
        "hist_gbm": pipe(HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.06, max_iter=600, l2_regularization=1.0,
            early_stopping=True, random_state=seed)),
    }
    if _HAS_XGB:
        models["xgboost"] = pipe(XGBRegressor(
            n_estimators=700, max_depth=4, learning_rate=0.03, subsample=0.85,
            colsample_bytree=0.85, reg_lambda=2.0, n_jobs=_N_JOBS, random_state=seed))
    for name, params in tuned_params.items():
        if name in models:
            models[name].set_params(**params)

    ensemble_members = [n for n in ("random_forest", "extra_trees", "hist_gbm", "xgboost") if n in models]
    models["voting_ensemble"] = VotingRegressor(estimators=[(n, models[n]) for n in ensemble_members])
    return models


def catchment_only_model(seed: int = 0) -> Pipeline:
    """passability ~ f(drainage area) only - to quantify the catchment-area problem."""
    return Pipeline([
        ("prep", _preprocessor(["log_drainage_area", "drainage_area_km2"], [], scale=False)),
        ("est", RandomForestRegressor(n_estimators=400, min_samples_leaf=5,
                                      n_jobs=_N_JOBS, random_state=seed)),
    ])


# Hyperparameter tuning: one RandomizedSearchCV per tunable model, scored with HUC8 group CV (never random CV, so tuning
# cannot reward memorising watershed identity). It is one search over the whole dataset and is not nested inside every
# fold of the repeated spatial-CV evaluation, which would multiply the cost by repeats times folds. The tuning seed is
# separate from the reporting seeds, so the reported score is still on combinations of watersheds that tuning did not see.

TUNABLE_PARAM_DISTS = {
    "random_forest": {
        "est__n_estimators": [200, 400, 600, 900],
        "est__max_depth": [None, 6, 10, 16, 24],
        "est__min_samples_leaf": [1, 2, 3, 5, 8, 12],
        "est__max_features": ["sqrt", 0.5, 0.7, 1.0],
    },
    "extra_trees": {
        "est__n_estimators": [200, 400, 600, 900],
        "est__max_depth": [None, 6, 10, 16, 24],
        "est__min_samples_leaf": [1, 2, 3, 5, 8, 12],
        "est__max_features": ["sqrt", 0.5, 0.7, 1.0],
    },
    "hist_gbm": {
        "est__max_depth": [3, 4, 5, 6, None],
        "est__learning_rate": [0.02, 0.04, 0.06, 0.08, 0.12],
        "est__max_iter": [200, 400, 600, 900],
        "est__l2_regularization": [0.0, 0.5, 1.0, 2.0, 4.0],
        "est__min_samples_leaf": [10, 20, 30, 50],
    },
}
if _HAS_XGB:
    TUNABLE_PARAM_DISTS["xgboost"] = {
        "est__n_estimators": [300, 500, 700, 1000],
        "est__max_depth": [3, 4, 5, 6, 8],
        "est__learning_rate": [0.01, 0.03, 0.05, 0.08],
        "est__subsample": [0.7, 0.85, 1.0],
        "est__colsample_bytree": [0.6, 0.75, 0.85, 1.0],
        "est__reg_lambda": [0.5, 1.0, 2.0, 4.0],
    }


def tune_hyperparameters(df, model_names=None, numeric=None, categorical=None,
                         seed: int = 0, n_iter: int = 20, inner_splits: int = 5,
                         group_col: str = "huc8_name") -> dict:
    """One RandomizedSearchCV per model in ``model_names`` (default: every key
    of ``TUNABLE_PARAM_DISTS``), scored by negative MAE under a HUC8-group CV
    built from ``df`` itself (seeded independently of the reporting splits -
    see module docstring above). Returns ``{model_name: best_params}``,
    passable straight to ``make_models(..., tuned_params=...)``."""
    from sklearn.model_selection import RandomizedSearchCV

    from .spatial_cv import huc8_group_kfold

    model_names = model_names or list(TUNABLE_PARAM_DISTS)
    df = df.reset_index(drop=True)
    cv = list(huc8_group_kfold(df, group_col, min(df[group_col].nunique(), inner_splits), seed=seed))
    base_models = make_models(numeric=numeric, categorical=categorical, seed=seed)

    best_params = {}
    for name in model_names:
        if name not in TUNABLE_PARAM_DISTS or name not in base_models:
            continue
        search = RandomizedSearchCV(
            base_models[name], TUNABLE_PARAM_DISTS[name], n_iter=n_iter, cv=cv,
            scoring="neg_mean_absolute_error", random_state=seed, n_jobs=_N_JOBS, refit=False)
        search.fit(df, df["passability"])
        best_params[name] = search.best_params_
    return best_params
