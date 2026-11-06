"""How should the volunteer yes/no answer be used when volunteers can be wrong? It uses development rows only and never reads the test
basins.

The rows are development crossings with a recorded outlet drop, HUC8 grouped 5-fold CV, seed 0, and map inputs without DSL rasters.
The model sees a volunteer answer a. The true value is q, which is 1 when the outlet drop is more than 0.25 ft. The answers are
simulated from assumed error rates, because volunteer accuracy has not been measured:
  sym   each answer is flipped with probability e
  miss  a truly perched outlet is called "no" with probability e, and a non-perched one is called "yes" with probability e/6
        (e = 0.3 gives 30% misses and 5% false alarms)

The grid trains the model on answers simulated at rate e_train and scores it on answers simulated at rate e_test. The test noise is
drawn once and is the same for every e_train. e_train = 0 is a model trained on clean surveyor answers, as in the first test on the
set-aside basins.
The measures are R2, Spearman, and capture, which is the share of the worst 20% of crossings (lowest true score) that the model also
puts in its worst 20%.
It also checks whether two layer rows with the same survey id can disagree on outlet drop.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.plunkett_regions_eval import load
from src import plunkett_cv as cv

SEED = 0


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def capture(y, p, frac=0.2):
    k = int(len(y) * frac)
    return len(set(np.argsort(y)[:k]) & set(np.argsort(p)[:k])) / k


def answers(q, rng, kind, e):
    if kind == "sym":
        return np.where(rng.random(len(q)) < e, 1 - q, q)
    if kind == "miss":                                  # a truly perched outlet is called "no" with probability e, a non-perched one "yes" with e/6
        u = rng.random(len(q))                          # (e = 0.3 gives the 30% misses / 5% false alarms named in the header)
        return np.where(q == 1, (u > e).astype(float), (u < e / 6).astype(float))
    raise ValueError(kind)


mk = lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=10)

if __name__ == "__main__":
    # join audit: duplicate layer rows with the same survey id
    n = pd.read_parquet("data/raw/naacc_public_layer.parquet")
    g = pd.DataFrame({"sid": n.Survey_Id, "a": pd.to_numeric(n.Outlet_D_1, errors="coerce").where(lambda s: s >= 0)})
    g = g[g.sid.duplicated(keep=False) & g.a.notna()]
    sp = g.groupby("sid").a.agg(lambda s: s.max() - s.min())
    print(f"layer survey ids repeated with a recorded drop: {len(sp)}; share where the repeated rows disagree by > 0.05 ft: {(sp > 0.05).mean():.4f}", flush=True)

    dev, t3, nodsl = load()
    x = dev[dev.q_perched.notna()].reset_index(drop=True)
    q, y = x.q_perched.values, x.aquatic_score.values
    folds = [(np.asarray(a), np.asarray(b)) for a, b in cv.random_group_kfold(x, "huc8", 5, SEED)]
    print(f"rows {len(x)}, perched share {q.mean():.3f}", flush=True)
    feats = nodsl
    X = x[feats].copy()
    rates = (0.0, 0.1, 0.2, 0.3)
    for kind in ("sym", "miss"):
        rng = np.random.default_rng(1)
        a_test = {e: answers(q, rng, kind, e) for e in rates}                       # test noise, fixed
        a_train = {e: answers(q, np.random.default_rng(10 + int(e * 100)), kind, e) for e in rates}   # train noise, fixed
        print(f"\n=== error kind: {kind} ({'random flips at rate e' if kind == 'sym' else 'e = miss rate 0/10/20/30%, false-alarm rate e/6 = 0/1.7/3.3/5%'})", flush=True)
        oof = {(et, es): np.full(len(x), np.nan) for et in rates for es in rates}
        base = np.full(len(x), np.nan)
        for tr, te in folds:
            m0 = mk().fit(X.iloc[tr], y[tr]); base[te] = m0.predict(X.iloc[te])
            for et in rates:
                Xt = X.iloc[tr].assign(a=a_train[et][tr])
                m = mk().fit(Xt, y[tr])
                for es in rates:
                    oof[(et, es)][te] = m.predict(X.iloc[te].assign(a=a_test[es][te]))
        print(f" map only (no answer): R2={r2(y, base):.4f} Spearman={spearmanr(y, base)[0]:.3f} capture={capture(y, base):.3f}")
        print(" rows = noise the model was TRAINED on, columns = noise in the TEST answers:   R2 (Spearman, capture of worst 20%)")
        print("   train\\test " + "".join(f"{es:>26.1f}" for es in rates))
        for et in rates:
            print(f"   {et:>9.1f} " + "".join(f"  {r2(y, oof[(et, es)]):6.3f} ({spearmanr(y, oof[(et, es)])[0]:.2f},{capture(y, oof[(et, es)]):.2f})   " for es in rates), flush=True)
    print("done")
