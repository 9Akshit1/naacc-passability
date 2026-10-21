"""Paired cluster-bootstrap (by HUC8) comparison of ALL locked-set predictions already produced.
The random-forest baseline is refit with its original fixed seed/params (deterministic) because its
predictions were not saved; no new model, tier or choice is introduced here."""
import json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src import plunkett_cv as cv, plunkett_features as pf
d = pd.read_parquet("data/processed/plunkett_features_osm.parquet"); lock = cv.locked_test_mask(d)
dev, test = d[~lock], d[lock]; g = pf.feature_groups(d)
rf = RandomForestRegressor(n_estimators=300, max_features=0.33, min_samples_leaf=3, n_jobs=10, random_state=0)
p_rf = rf.fit(dev[g["PLUNKETT"]], dev.aquatic_score).predict(test[g["PLUNKETT"]])
P = {"RF (Plunkett-style)": p_rf}
for k, n in [("T1", "T1 ensemble"), ("T2", "T2 ensemble +NBI"), ("T3", "T3 ensemble +NBI+OSM")]:
    sv = pd.read_csv(f"results/plunkett/locked_predictions_{k}.csv").set_index("crossing_code")["pred"]
    P[n] = sv.reindex(test["crossing_code"]).values
y = test.aquatic_score.values; grp = test.huc8.values; ug = np.unique(grp); idx = {u: np.where(grp == u)[0] for u in ug}
r2 = lambda yy, pp: 1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
rng = np.random.default_rng(0); B = {n: [] for n in P}
for _ in range(2000):
    s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
    for n in P: B[n].append(r2(y[s], P[n][s]))
out = {"r2": {n: float(r2(y, P[n])) for n in P}, "r2_ci95": {n: [float(np.percentile(B[n], 2.5)), float(np.percentile(B[n], 97.5))] for n in P}, "deltas": {}}
names = list(P)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        dl = np.array(B[b]) - np.array(B[a]); out["deltas"][f"{b} - {a}"] = {"mean": float(dl.mean()), "ci95": [float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))], "p_gt0": float((dl > 0).mean())}
json.dump(out, open("results/plunkett/locked_paired_deltas.json", "w"), indent=1)
for n in P: print(f"{n:24s} R2={out['r2'][n]:.4f}  95%CI {out['r2_ci95'][n][0]:.3f}-{out['r2_ci95'][n][1]:.3f}")
for k, v in out["deltas"].items(): print(f"  delta {k:48s} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  P(>0)={v['p_gt0']:.3f}")
