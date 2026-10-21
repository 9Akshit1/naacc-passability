"""Assemble the multi-state feature table: Plunkett predictors + NBI (+ OSM when cached)
+ leakage-safe neighbour-label features.

Feature groups (names used throughout the experiment scripts):
  PLUNKETT  - Plunkett's own 38 predictors (passability_predvars.txt), unchanged
  NBI       - FHWA National Bridge Inventory join (src/nbi.py)
  OSM       - OSM bridge/culvert tags (src/osm_structures.py)
  NEIGHBOUR - label summaries of nearby SURVEYED crossings in OTHER watersheds,
              computed only from the training fold (see neighbour_features)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from . import nbi as nbi_mod
from . import plunkett_data as pdm

NBI_RADIUS_M = 60.0


def build(use_osm: bool = False) -> pd.DataFrame:
    d = pdm.add_engineered(pdm.load())
    n = nbi_mod.load_nbi(sorted(d.state.unique()))
    m = nbi_mod.match(d, n, radius_m=NBI_RADIUS_M)
    d = pd.concat([d, m.drop(columns=["_nearest_dist_m", "_mutual"])], axis=1)
    if use_osm:
        from . import osm_structures as osm
        o = osm.match_crossings(d, sorted(d.state.unique()))
        d = pd.concat([d, o], axis=1)
    x, y = Transformer.from_crs(4326, 5070, always_xy=True).transform(d.lon_nad83.values, d.lat_nad83.values)
    d["_x"], d["_y"] = x, y
    return d


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    nbi_cols = [c for c in df.columns if c.startswith("nbi_") and c != "nbi_dist_m"]
    osm_cols = [c for c in df.columns if c.startswith("osm_") and not c.endswith("_dist_m")]
    return {"PLUNKETT": pdm.predictor_columns(), "NBI": nbi_cols, "OSM": osm_cols}


NEIGHBOUR_COLS = ["nb_mean_k5", "nb_mean_k15", "nb_bridge_share_k15", "nb_dist_nearest_km",
                  "nb_mean_r5km", "nb_n_r5km"]


def neighbour_features(ref: pd.DataFrame, qry: pd.DataFrame, exclude_same_group: str | None = "huc8",
                       kmax: int = 80) -> pd.DataFrame:
    """Label summaries of the nearest ``ref`` crossings for each ``qry`` crossing.

    ``ref`` must be TRAINING rows only (labels allowed); when ``qry`` is part of ``ref``
    (computing features for training rows) neighbours in the same ``exclude_same_group``
    (HUC8) are dropped, so training rows see neighbours at the same kind of distance a
    held-out watershed would - otherwise the model would learn to trust neighbour labels
    far more than it can at test time. A row is never its own neighbour.
    """
    tree = cKDTree(np.c_[ref["_x"], ref["_y"]])
    k = min(kmax, len(ref))
    dist, idx = tree.query(np.c_[qry["_x"], qry["_y"]], k=k)
    rs, rb = ref["aquatic_score"].values, ref["bridge"].astype(float).values
    rg = ref[exclude_same_group].values if exclude_same_group else None
    qg = qry[exclude_same_group].values if exclude_same_group else None
    out = np.full((len(qry), len(NEIGHBOUR_COLS)), np.nan)
    for i in range(len(qry)):
        di, ii = dist[i], idx[i]
        keep = di > 0.5                       # never itself (coincident point)
        if rg is not None:
            keep &= rg[ii] != qg[i]
        di, ii = di[keep], ii[keep]
        if len(ii) == 0:
            continue
        w5 = 1.0 / (di[:5] + 100.0); w15 = 1.0 / (di[:15] + 100.0)
        out[i, 0] = np.average(rs[ii[:5]], weights=w5)
        out[i, 1] = np.average(rs[ii[:15]], weights=w15)
        out[i, 2] = np.average(rb[ii[:15]], weights=w15)
        out[i, 3] = di[0] / 1000.0
        r5 = di <= 5000
        out[i, 5] = r5.sum()
        out[i, 4] = rs[ii[r5]].mean() if r5.any() else np.nan
    return pd.DataFrame(out, index=qry.index, columns=NEIGHBOUR_COLS)


def neighbour_prep(tr: pd.DataFrame, te: pd.DataFrame):
    """cross_val_oof ``prep`` hook: train-fold labels only."""
    tr = tr.join(neighbour_features(tr, tr))
    te = te.join(neighbour_features(tr.drop(columns=NEIGHBOUR_COLS), te, exclude_same_group=None))
    return tr, te


# ---------------------------------------------------------------------------
# Auxiliary-supervision stacking: P(bridge) learned from the survey 'bridge' field
# ---------------------------------------------------------------------------
AUX_COL = "aux_p_bridge"


def aux_bridge_prep_factory(feature_cols, n_inner: int = 5, seed: int = 0):
    """Returns a ``prep(tr, te)`` hook for ``plunkett_cv.cross_val_oof``.

    Inside each training fold: fit a LightGBM classifier on the survey ``bridge`` flag
    (a training-only label), give training rows their inner HUC8-group OUT-OF-FOLD
    probability (so the downstream model never sees a probability fitted on the row's
    own label) and test rows the probability from a classifier fit on the whole training
    fold. Test labels are never touched."""
    import lightgbm as lgb
    from .spatial_cv import huc8_group_kfold

    def mk():
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=30,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=8,
                                  random_state=seed)

    def prep(tr, te):
        tr, te = tr.copy(), te.copy()
        yb = tr["bridge"].astype(int)
        inner = pd.Series(np.nan, index=tr.index)
        for a, b in huc8_group_kfold(tr, group_col="huc8", n_splits=n_inner, seed=seed + 11):
            m = mk().fit(tr.loc[a, feature_cols], yb.loc[a])
            inner.loc[b] = m.predict_proba(tr.loc[b, feature_cols])[:, 1]
        tr[AUX_COL] = inner
        te[AUX_COL] = mk().fit(tr[feature_cols], yb).predict_proba(te[feature_cols])[:, 1]
        return tr, te
    return prep


# ---------------------------------------------------------------------------
# Derived features (functions of Plunkett's predictors and NBI size fields only)
# ---------------------------------------------------------------------------
_WINDOWS = [90, 150, 210, 270, 330, 390, 450, 510, 570, 630]


def add_derived(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return (df with extra columns, {group_name: [new columns]}). No labels, coordinates or state used."""
    d = df.copy()
    eps = 1e-3
    area_km2 = d["d8accum"] * 900 / 1e6          # 30 m cells (matches implied cell size of the best-linked reaches)
    d["log_area_km2"] = np.log10(area_km2)
    scale, constr = [], []
    d["log_stream_power"] = d["log_area_km2"] + np.log10(d["gradient"].clip(lower=0) + eps); scale.append("log_stream_power")
    inc = np.column_stack([d[f"incise1_{w}"] for w in _WINDOWS])
    rng_ = np.column_stack([d[f"elev_range_{w}"] for w in _WINDOWS])
    sd_ = np.column_stack([d[f"elev_sd_{w}"] for w in _WINDOWS])
    d["incise_ratio_90_630"] = d["incise1_90"] / (d["incise1_630"].abs() + eps)
    d["incise_ratio_270_630"] = d["incise1_270"] / (d["incise1_630"].abs() + eps)
    d["range_ratio_90_630"] = d["elev_range_90"] / (d["elev_range_630"] + eps)
    d["sd_ratio_150_450"] = d["elev_sd_150"] / (d["elev_sd_450"] + eps)
    d["incise_mean"] = inc.mean(1); d["incise_max"] = inc.max(1)
    d["incise_argmax_window"] = np.array(_WINDOWS)[inc.argmax(1)]
    d["incise_slope_per_window"] = np.polyfit(np.arange(10), inc.T, 1)[0]
    d["range_slope_per_window"] = np.polyfit(np.arange(10), rng_.T, 1)[0]
    d["sd_mean"] = sd_.mean(1)
    scale += ["incise_ratio_90_630", "incise_ratio_270_630", "range_ratio_90_630", "sd_ratio_150_450", "incise_mean",
              "incise_max", "incise_argmax_window", "incise_slope_per_window", "range_slope_per_window", "sd_mean"]
    # constriction proxy: structure size relative to the channel width expected from catchment area
    # (hydraulic geometry W ~ A^0.4). Missing where no NBI structure matched.
    for src, name in [("nbi_length_m", "constr_len"), ("nbi_max_span_m", "constr_span"), ("nbi_deck_width_m", "constr_deck")]:
        if src in d.columns:
            d[name] = np.log10(d[src].clip(lower=0.1)) - 0.4 * d["log_area_km2"]; constr.append(name)
    return d, {"SCALE": scale, "CONSTRICTION": constr}
