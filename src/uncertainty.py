"""Split-conformal prediction intervals.

Gives a distribution-free interval with (approximately) the requested coverage,
without assuming a probabilistic model. Uses a group-aware calibration split so
the guarantee reflects spatial transfer rather than in-watershed noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def conformal_intervals(model, df: pd.DataFrame, train_idx, calib_idx, test_idx,
                        target: str = "passability", alpha: float = 0.1):
    """Fit on ``train_idx``, calibrate residual quantile on ``calib_idx``,
    return (lower, point, upper, half_width) for ``test_idx`` clipped to [0, 1]."""
    model.fit(df.loc[train_idx], df.loc[train_idx, target])
    calib_pred = model.predict(df.loc[calib_idx])
    calib_res = np.abs(df.loc[calib_idx, target].values - calib_pred)
    n = len(calib_res)
    q = np.quantile(calib_res, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")

    point = model.predict(df.loc[test_idx])
    lo = np.clip(point - q, 0, 1)
    hi = np.clip(point + q, 0, 1)
    return pd.DataFrame({"lower": lo, "point": np.clip(point, 0, 1), "upper": hi,
                         "half_width": q}, index=test_idx)


def evaluate_coverage(intervals: pd.DataFrame, y_true: pd.Series) -> dict:
    y = y_true.loc[intervals.index]
    covered = ((y >= intervals["lower"]) & (y <= intervals["upper"])).mean()
    return {"empirical_coverage": float(covered),
            "mean_interval_width": float((intervals["upper"] - intervals["lower"]).mean())}
