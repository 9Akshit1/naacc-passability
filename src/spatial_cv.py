"""Cross-validation splitters.

The research question is *spatial transfer*: can the model predict passability at
crossings in watersheds it never trained on? A random split leaks watershed-level
structure (geology, stream size regime, land use) into the test set and gives an
over-optimistic score. We therefore compare:

* ``random_kfold``            - ordinary k-fold on individual crossings (reference only)
* ``huc8_group_kfold``        - entire HUC8 watersheds held out (one split)
* ``repeated_huc8_group_kfold`` - the same, repeated with different random
  watershed-to-fold assignments, so the headline score is a mean +/- spread
  across several splits rather than whatever one split happened to draw
* ``leave_one_huc8_out``      - every watershed with enough crossings held
  out individually, for a per-watershed breakdown
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
    """Group k-fold over ``group_col``, entire groups held out per fold.

    Unlike ``sklearn.model_selection.GroupKFold`` (which has no ``shuffle``/
    ``random_state`` in this project's sklearn version and is fully
    deterministic given a group column - every "re-run" was silently
    evaluating the exact same single split), this shuffles the group order
    by ``seed`` before a greedy size-balanced assignment, so different
    seeds give genuinely different held-out-watershed combinations. That
    variation is what ``repeated_huc8_group_kfold`` uses to turn a single
    lucky/unlucky split into a distribution.
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
