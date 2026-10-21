"""Cross-validation protocol for the multi-state (Plunkett) dataset.

Rules fixed BEFORE any model was fit on this dataset (2026-10-08):

1. Locked test set - ~15% of crossings, chosen as whole HUC6 basins (seeded,
   state-diverse), are set aside. They are never used for fitting, feature
   selection, hyper-parameter tuning, or model choice. They are scored once,
   at the very end, for the final configuration and the baselines.
2. Development set = everything else. All comparisons use grouped CV on it:
   * ``huc8``  - hold out whole HUC8 watersheds (the standard used since 2026-09)
   * ``huc6``  - hold out whole HUC6 basins (bigger blocks => less spatial leakage)
   * ``state`` - leave-one-state-out (the "does it transfer to a new state?" test
     that Plunkett flagged as the thing that degraded in the past)
   * ``random`` - ordinary k-fold, reported only to show the optimism gap
3. Hyper-parameters are tuned only on the development set, inside grouped CV.
4. Reported numbers are pooled out-of-fold R2/MAE, and mean +/- sd over
   repeated, differently-seeded group assignments (never a single split).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold



def random_group_kfold(df: pd.DataFrame, group_col: str, n_splits: int = 5, seed: int = 0):
    """Whole groups held out; the group-to-fold assignment genuinely depends on ``seed``.

    Groups are visited in a seeded RANDOM order and each goes to the currently lightest fold
    (ties broken at random). The older ``spatial_cv.huc8_group_kfold`` sorts groups by size after
    shuffling, so for unique group sizes every seed gave the same split (found 2026-10-08: seeds
    0-4 agreed on 95-100% of rows) - repeated runs with it do NOT measure split-to-split variance.
    """
    groups = df[group_col].astype(str).values
    uniq = np.unique(groups)
    n_splits = min(n_splits, len(uniq))
    sizes = pd.Series(groups).value_counts()
    rng = np.random.default_rng(seed)
    load = np.zeros(n_splits)
    fold_of = {}
    for g in rng.permutation(uniq):
        lightest = np.flatnonzero(load == load.min())
        f = int(rng.choice(lightest))
        fold_of[g] = f
        load[f] += sizes[g]
    fold_row = np.array([fold_of[g] for g in groups])
    for f in range(n_splits):
        yield df.index[fold_row != f], df.index[fold_row == f]

LOCK_SEED = 20261008
LOCK_TARGET_SHARE = 0.15
LOCK_MIN_STATES = 4
LOCK_MIN_GROUP_N = 250


def locked_test_mask(df: pd.DataFrame, seed: int = LOCK_SEED) -> pd.Series:
    """Boolean mask: True = locked test row. Deterministic given ``seed`` and the data."""
    sizes = df["huc6"].value_counts()
    eligible = [g for g in sizes.index if sizes[g] >= LOCK_MIN_GROUP_N]
    total = len(df)
    for k in range(1000):
        rng = np.random.default_rng(seed + k)
        order = list(rng.permutation(eligible))
        chosen, n = [], 0
        for g in order:
            if n >= LOCK_TARGET_SHARE * total:
                break
            # do not let one giant basin blow far past the target
            if n + sizes[g] > (LOCK_TARGET_SHARE + 0.04) * total:
                continue
            chosen.append(g)
            n += sizes[g]
        m = df["huc6"].isin(chosen)
        if (LOCK_TARGET_SHARE - 0.02) <= m.mean() <= (LOCK_TARGET_SHARE + 0.04) and df.loc[m, "state"].nunique() >= LOCK_MIN_STATES:
            return m
    raise RuntimeError("could not build a locked set under the constraints")


def splits(df: pd.DataFrame, scheme: str, n_splits: int = 5, seed: int = 0):
    """Yield (train_idx, test_idx) as positional-label index arrays of ``df``."""
    if scheme == "random":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in kf.split(df):
            yield df.index[tr], df.index[te]
    elif scheme in ("huc8", "huc6", "huc4"):
        yield from random_group_kfold(df, group_col=scheme, n_splits=n_splits, seed=seed)
    elif scheme == "state":
        # leave-one-state-out; a HUC8 straddling a state line stays whole in the TRAIN or
        # TEST side only through the state label, so border watersheds do leak a little -
        # reported as is (the state label is the point-in-polygon state of each crossing).
        for s in sorted(df["state"].unique()):
            te = df.index[df["state"] == s]
            if len(te) < 150:
                continue
            yield df.index.difference(te), te
    else:
        raise ValueError(scheme)


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    res = y - p
    ss_res, ss_tot = float((res ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    from scipy.stats import spearmanr
    return {
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mae": float(np.abs(res).mean()),
        "rmse": float(np.sqrt((res ** 2).mean())),
        "spearman": float(spearmanr(y, p)[0]),
        "n": int(len(y)),
    }


def cross_val_oof(df, feature_cols, make_model, scheme, target="aquatic_score",
                  n_splits=5, seeds=(0, 1, 2), fit_kwargs=None, prep=None):
    """Return list (one per seed) of dicts {oof: Series, metrics: dict, per_fold: [...]}.

    ``make_model()`` -> fresh sklearn-style regressor. ``prep(train_df, test_df)`` may
    return transformed (train_df, test_df) with features computed ONLY from train labels
    (used for neighbour-label features), and must never look at test labels.
    """
    out = []
    # state scheme is deterministic (no seed dependence)
    use_seeds = (0,) if scheme == "state" else seeds
    for sd in use_seeds:
        oof = pd.Series(np.nan, index=df.index)
        folds = []
        for tr, te in splits(df, scheme, n_splits=n_splits, seed=sd):
            tr_df, te_df = df.loc[tr], df.loc[te]
            if prep is not None:
                tr_df, te_df = prep(tr_df, te_df)
            m = make_model()
            m.fit(tr_df[feature_cols], tr_df[target], **(fit_kwargs or {}))
            p = m.predict(te_df[feature_cols])
            oof.loc[te] = p
            fm = metrics(te_df[target], p)
            fm["held_out"] = (te_df["state"].iloc[0] if scheme == "state" else None)
            folds.append(fm)
        ok = oof.notna()
        out.append({"seed": sd, "oof": oof, "metrics": metrics(df.loc[ok, target], oof[ok]), "per_fold": folds})
    return out


def summarize(runs) -> dict:
    r2 = [r["metrics"]["r2"] for r in runs]
    mae = [r["metrics"]["mae"] for r in runs]
    return {"r2_mean": float(np.mean(r2)), "r2_sd": float(np.std(r2)),
            "mae_mean": float(np.mean(mae)), "mae_sd": float(np.std(mae)),
            "spearman_mean": float(np.mean([r["metrics"]["spearman"] for r in runs])),
            "n_repeats": len(runs)}
