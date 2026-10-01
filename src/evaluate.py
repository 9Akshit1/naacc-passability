"""Metrics for continuous passability prediction.

Reports MAE / RMSE / R2 / Spearman (ranking) plus calibration and
predict-toward-the-mean diagnostics - the last one because the DSL documentation
states its GIS model "generally doesn't predict the more extreme values".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    err = y_pred - y_true
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    rho = spearmanr(y_true, y_pred).statistic if len(y_true) > 2 else np.nan
    return {
        "n": int(m.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "spearman": float(rho),
        "bias": float(np.mean(err)),
        # variance-shrinkage: <1 means predictions are less spread than truth
        "pred_sd_ratio": float(np.std(y_pred) / np.std(y_true)) if np.std(y_true) > 0 else np.nan,
        # performance on the tails (true extremes)
        "mae_low_tail": _tail_mae(y_true, y_pred, "low"),
        "mae_high_tail": _tail_mae(y_true, y_pred, "high"),
    }


def _tail_mae(y_true, y_pred, which: str, q: float = 0.15) -> float:
    if which == "low":
        m = y_true <= np.quantile(y_true, q)
    else:
        m = y_true >= np.quantile(y_true, 1 - q)
    return float(np.mean(np.abs(y_pred[m] - y_true[m]))) if m.any() else np.nan


def calibration_table(y_true, y_pred, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true, float), "p": np.asarray(y_pred, float)})
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(n=("y", "size"), pred_mean=("p", "mean"),
                                             obs_mean=("y", "mean"))
    g["gap"] = g["obs_mean"] - g["pred_mean"]
    return g.reset_index(drop=True)


def cv_predict(model, df: pd.DataFrame, splits, target: str = "passability") -> pd.Series:
    """Out-of-fold predictions over the given splitter."""
    oof = pd.Series(np.nan, index=df.index)
    for tr, te in splits:
        model.fit(df.loc[tr], df.loc[tr, target])
        oof.loc[te] = model.predict(df.loc[te])
    return oof


def prioritization_overlap(pred_a: pd.Series, pred_b: pd.Series, top_frac: float = 0.2) -> dict:
    """How much do two models' 'worst crossings' lists agree? Lower passability =
    higher restoration priority, so we take the bottom `top_frac`."""
    common = pred_a.dropna().index.intersection(pred_b.dropna().index)
    a, b = pred_a.loc[common], pred_b.loc[common]
    k = max(1, int(len(common) * top_frac))
    set_a = set(a.nsmallest(k).index)
    set_b = set(b.nsmallest(k).index)
    jacc = len(set_a & set_b) / len(set_a | set_b)
    return {"k": k, "jaccard": jacc, "changed": k - len(set_a & set_b),
            "spearman_full": float(spearmanr(a, b).statistic)}
