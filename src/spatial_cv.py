"""Cross-validation splitters for the New York version.

The question is whether the model can predict crossings in watersheds it has never seen. A random split puts crossings from the
same watershed on both sides, so the test set shares geology, stream size and land use with the training set and the score is too
high. The splitters here are:

- random_kfold: ordinary k-fold on single crossings, for reference only
- huc8_group_kfold: whole HUC8 watersheds held out, with one split
- repeated_huc8_group_kfold: the same, repeated with different random assignments of watersheds to folds, so that the score is
  a mean and a spread over several splits and not whatever one split happened to give
- leave_one_huc8_out: each watershed with enough crossings held out alone, for a per-watershed breakdown

huc8_group_kfold sorts the watersheds by size after shuffling, so its folds barely change with the seed. The multi-state code
uses plunkett_cv.random_group_kfold instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def random_kfold(df: pd.DataFrame, n_splits: int = 5, seed: int = 0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(df):
        yield df.index[tr], df.index[te]


def huc8_group_kfold(df: pd.DataFrame, group_col: str = "huc8_name",
                     n_splits: int | None = None, seed: int = 0):
    """Group k-fold over ``group_col``, with entire groups held out in each fold.

    ``sklearn.model_selection.GroupKFold`` has no ``shuffle`` or ``random_state`` in the scikit-learn version used here and
    always gives the same split, so every "re-run" with it evaluated the same single split. This function shuffles the group
    order with ``seed`` before a greedy size-balanced assignment, which is what ``repeated_huc8_group_kfold`` relies on to
    turn one split into a distribution. The assignment then sorts the groups by size, so the folds still change little
    with the seed (see ``plunkett_cv.random_group_kfold``).
    """
    groups = df[group_col].astype(str).values
    uniq = np.unique(groups)
    n_splits = n_splits or min(len(uniq), 7)
    sizes = pd.Series(groups).value_counts()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(uniq)
    # Greedy size-balanced bin-packing: largest-first (for balance) among the
    # seed-shuffled group order (for variation across seeds/repeats).
    order = sorted(shuffled, key=lambda g: -sizes[g])
    fold_sizes = np.zeros(n_splits, dtype=int)
    group_to_fold = {}
    for g in order:
        f = int(np.argmin(fold_sizes))
        group_to_fold[g] = f
        fold_sizes[f] += sizes[g]
    fold_of_row = np.array([group_to_fold[g] for g in groups])
    for f in range(n_splits):
        te = np.where(fold_of_row == f)[0]
        tr = np.where(fold_of_row != f)[0]
        yield df.index[tr], df.index[te]


def repeated_huc8_group_kfold(df: pd.DataFrame, group_col: str = "huc8_name",
                              n_splits: int | None = None, n_repeats: int = 5,
                              seed: int = 0):
    """``n_repeats`` independent ``huc8_group_kfold`` runs (different seeds),
    each a list of (train_idx, test_idx) pairs - so the caller can evaluate a
    model on every repeat and report the mean/spread of the metric, not a
    single split's number."""
    for r in range(n_repeats):
        yield list(huc8_group_kfold(df, group_col, n_splits, seed=seed + r))


def leave_one_huc8_out(df: pd.DataFrame, group_col: str = "huc8_name", min_test: int = 40):
    for huc, g in df.groupby(group_col):
        if len(g) < min_test:
            continue
        te = g.index
        tr = df.index.difference(te)
        yield huc, tr, te
