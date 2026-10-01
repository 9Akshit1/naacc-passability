"""Inference pipeline: unsurveyed crossings -> predicted passability + interval.

    python -m src.predict --crossings my_crossings.csv --out predictions.csv

``my_crossings.csv`` needs at least ``crossing_id, lat, lon``. Optional
``County`` (for census join) improves predictions. All other GIS predictors -
drainage area, elevation, road class (from OpenStreetMap), upstream watershed
metrics (from EPA StreamCat), and stream slope/order (NHDPlusV2 VAA) - are
fetched automatically from the same public services used to train the model,
with caching. No field-survey columns are required or used.

The model is trained on the NY inland subset (see METHODOLOGY.md). Predictions
outside that region / regime should be treated as illustrative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from .gis_predictors import build_predictor_table
from .model_experiments import SEED, TARGET, load_subset
from .models import make_models
from .uncertainty import conformal_intervals

MODEL_DIR = Path("results/models")
EXPERIMENT_RESULTS = Path("results/experiment_results.json")


def _load_tuned_params() -> dict:
    """Hyperparameters found by ``model_experiments.py``'s
    ``tune_hyperparameters`` pass, if that's been run - so the deployed
    model matches what the report evaluated, not silently the untuned
    defaults. Missing/unreadable file just means "not tuned yet"."""
    if not EXPERIMENT_RESULTS.exists():
        return {}
    try:
        return json.loads(EXPERIMENT_RESULTS.read_text()).get("tuned_hyperparameters", {})
    except json.JSONDecodeError:
        return {}


def train_final_model(model_name: str = "voting_ensemble"):
    df = load_subset()
    rng = np.random.default_rng(SEED)
    calib = pd.Index(rng.choice(df.index, size=max(120, len(df) // 6), replace=False))
    train = df.index.difference(calib)
    model = make_models(seed=SEED, tuned_params=_load_tuned_params())[model_name]
    model.fit(df.loc[train], df.loc[train, TARGET])
    calib_pred = model.predict(df.loc[calib])
    calib_res = np.abs(df.loc[calib, TARGET].values - calib_pred)
    return model, df, calib_res


def _model_path(model_name: str, path: str | Path | None = None) -> Path:
    return Path(path) if path else MODEL_DIR / f"{model_name}.joblib"


def save_final_model(model, calib_res: np.ndarray, model_name: str,
                     path: str | Path | None = None) -> Path:
    """Fits are otherwise never written to disk - predict_crossings retrains
    on every call by default. This saves the fitted model + its conformal
    calibration residuals as one file, so a later call can load instead of
    retrain."""
    out = _model_path(model_name, path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "calib_res": calib_res, "model_name": model_name}, out)
    return out


def load_final_model(model_name: str = "voting_ensemble", path: str | Path | None = None) -> dict:
    return joblib.load(_model_path(model_name, path))


def _conformal_q(calib_res: np.ndarray, alpha: float = 0.1) -> float:
    n = len(calib_res)
    return float(np.quantile(calib_res, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def predict_crossings(crossings: pd.DataFrame, model_name: str = "voting_ensemble",
                      alpha: float = 0.1, model_path: str | Path | None = None,
                      retrain: bool = False) -> pd.DataFrame:
    for col in ("crossing_id", "lat", "lon"):
        if col not in crossings.columns:
            raise ValueError(f"input needs a '{col}' column")
    crossings = crossings.copy()
    if "County" not in crossings.columns:
        crossings["County"] = pd.NA

    # Must fetch every predictor category the trained model's feature list
    # references (models.py::NUMERIC_FEATURES_DEFAULT/CATEGORICAL_FEATURES_
    # DEFAULT), or the ColumnTransformer raises on a missing column at
    # predict time. do_constriction is left off since constriction_ratio
    # isn't an active model feature (see models.py).
    feat = build_predictor_table(crossings, do_drainage=True, do_elev=True,
                                 do_road_class=True, road_class_mode="per_point",
                                 do_streamcat=True, do_vaa=True, do_aadt=True,
                                 do_terrain=True, do_ecoregion=True, do_structures=True,
                                 do_precip=True,
                                 sleep=0.1)

    path = _model_path(model_name, model_path)
    if retrain:
        model, _, calib_res = train_final_model(model_name)
    elif path.exists():
        saved = load_final_model(model_name, model_path)
        model, calib_res = saved["model"], saved["calib_res"]
    elif model_path:
        raise FileNotFoundError(f"no saved model at {path} - run with --save-model first")
    else:
        model, _, calib_res = train_final_model(model_name)
    q = _conformal_q(calib_res, alpha)
    point = np.clip(model.predict(feat), 0, 1)
    return pd.DataFrame({
        "crossing_id": feat["crossing_id"],
        "lat": feat["lat"], "lon": feat["lon"],
        "drainage_area_km2": feat.get("drainage_area_km2"),
        "elev_m": feat.get("elev_m"),
        "predicted_passability": point.round(4),
        "lower_bound": np.clip(point - q, 0, 1).round(4),
        "upper_bound": np.clip(point + q, 0, 1).round(4),
        "interval_half_width": round(q, 4),
        "model": model_name,
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crossings", help="required unless --save-model is given")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default="voting_ensemble")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--model-path", help="where to save/load the fitted model (default: results/models/<model>.joblib)")
    ap.add_argument("--save-model", action="store_true",
                    help="train once, save it to --model-path, then exit without predicting")
    ap.add_argument("--retrain", action="store_true",
                    help="ignore any saved model file and train fresh")
    a = ap.parse_args()

    if a.save_model:
        model, _, calib_res = train_final_model(a.model)
        out = save_final_model(model, calib_res, a.model, a.model_path)
        print(f"saved {a.model} to {out}")
        return

    if not a.crossings:
        ap.error("--crossings is required unless --save-model is given")
    df = pd.read_csv(a.crossings)
    res = predict_crossings(df, a.model, a.alpha, model_path=a.model_path, retrain=a.retrain)
    res.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(res)} crossings, {1 - a.alpha:.0%} prediction intervals)")
    print(res.head(10).to_string(index=False))


if __name__ == "__main__":
    from .models import _N_JOBS
    # Same reliability fix as model_experiments.py - see models.py's
    # _N_JOBS docstring.
    with joblib.parallel_backend("threading", n_jobs=_N_JOBS):
        main()
