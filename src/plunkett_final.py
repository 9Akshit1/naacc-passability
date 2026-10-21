"""Frozen final-model recipe (chosen on the DEVELOPMENT set only, 2026-10-08).

Equal-weight average of three members:
  * LightGBM, L2 objective, extra-trees splits (best regression config of the 36-config search)
  * LightGBM, cross-entropy objective (bounded 0-1 output; best xent config of the same search)
  * CatBoost depth 6
The search was scored by HUC8-group CV with seed 100 on the development set; see
results/plunkett/tune.jsonl. Nothing here was altered after looking at the locked test set.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
from catboost import CatBoostRegressor

LGBM_REG = dict(num_leaves=15, min_child_samples=100, n_estimators=600, learning_rate=0.02, colsample_bytree=0.8,
                subsample=0.6, reg_lambda=0, objective="regression", extra_trees=True, max_bin=255)
LGBM_XENT = dict(num_leaves=7, min_child_samples=100, n_estimators=1000, learning_rate=0.03, colsample_bytree=0.8,
                 subsample=1.0, reg_lambda=30, objective="cross_entropy", extra_trees=False, max_bin=255)


def _lgbm(p, seed=0):
    return lgb.LGBMRegressor(subsample_freq=1, verbose=-1, n_jobs=10, random_state=seed, **p)


def _cat(seed=0):
    return CatBoostRegressor(iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=5, verbose=0,
                             thread_count=10, random_seed=seed)


class FinalEnsemble:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, y):
        self.members_ = [_lgbm(LGBM_REG, self.seed).fit(X, y), _lgbm(LGBM_XENT, self.seed).fit(X, y),
                         _cat(self.seed).fit(X, y)]
        self.columns_ = list(X.columns)
        return self

    def predict(self, X):
        X = X[self.columns_]
        return np.clip(np.mean([m.predict(X) for m in self.members_], axis=0), 0.0, 1.0)
