"""ONE-TIME evaluation on the locked test set (whole HUC6 basins never used for anything).

    python -m scripts.plunkett_locked_eval T1 T2          # tiers to score now
A tier that already has an entry in results/plunkett/locked_results.json is REFUSED (so the
locked set cannot be iterated against). Final recipe = src/plunkett_final.FinalEnsemble.
Also scores, for reference, a Plunkett-style random forest on his predictors.
Uncertainty: 90% intervals = prediction +- the 90th percentile of |dev out-of-fold residual|
(HUC8-group CV OOF, so the width reflects spatial transfer), global and by prediction quintile.
"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src import plunkett_cv as cv, plunkett_features as pf
from src.plunkett_final import FinalEnsemble

OUT = "results/plunkett/locked_results.json"
done = json.load(open(OUT)) if os.path.exists(OUT) else {}
want = [a for a in sys.argv[1:] if not a.startswith("--")]
src_path = "data/processed/plunkett_features_osm.parquet" if "T3" in want else "data/processed/plunkett_features_nbi.parquet"
d = pd.read_parquet(src_path)
lock = cv.locked_test_mask(d)
dev, test = d[~lock], d[lock]
g = pf.feature_groups(d)
g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]
tiers = {"T1": ("T1_plunkett", g["PLUNKETT"]), "T2": ("T2_plunkett+nbi", g["PLUNKETT"] + g["NBI"]),
         "T3": ("T3_plunkett+nbi+osm", g["PLUNKETT"] + g["NBI"] + g["OSM"])}
y = test["aquatic_score"]
preds = {}

def record(name, p, oof_path=None):
    m = cv.metrics(y, p)
    m["pred_sd_over_obs_sd"] = float(np.std(p) / np.std(y))
    m["calib_slope_obs_on_pred"] = float(np.polyfit(p, y, 1)[0])
    if oof_path and os.path.exists(oof_path):
        res = (dev["aquatic_score"] - pd.read_parquet(oof_path)["oof"].reindex(dev.index)).abs().dropna()
        q = float(np.quantile(res, 0.9)); lo, hi = np.clip(p - q, 0, 1), np.clip(p + q, 0, 1)
        m["conformal90_global"] = {"half_width": q, "coverage": float(((y >= lo) & (y <= hi)).mean())}
        dev_pred = pd.read_parquet(oof_path)["oof"].reindex(dev.index); ok = dev_pred.notna()
        edges = np.quantile(dev_pred[ok], [0, .2, .4, .6, .8, 1]); edges[0], edges[-1] = -1, 2
        cb = np.digitize(p, edges[1:-1]); db = np.digitize(dev_pred[ok], edges[1:-1])
        qs = [float(np.quantile(res.reindex(dev_pred[ok].index)[db == b], 0.9)) for b in range(5)]
        lo, hi = np.clip(p - np.array(qs)[cb], 0, 1), np.clip(p + np.array(qs)[cb], 0, 1)
        m["conformal90_by_pred_quintile"] = {"half_widths": qs, "coverage": float(((y >= lo) & (y <= hi)).mean())}
    done[name] = m; preds[name] = np.asarray(p)
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if not isinstance(v, dict)}, flush=True)

if "T1" in want and "T1_rf_plunkett_style" not in done:
    rf = RandomForestRegressor(n_estimators=300, max_features=0.33, min_samples_leaf=3, n_jobs=10, random_state=0)
    record("T1_rf_plunkett_style", rf.fit(dev[g["PLUNKETT"]], dev.aquatic_score).predict(test[g["PLUNKETT"]]))
if "T1" in want or "T2" in want or "T3" in want:
    record("T0_dev_mean", np.full(len(test), dev.aquatic_score.mean())) if "T0_dev_mean" not in done else None
for k in want:
    name, feats = tiers[k]
    if name in done:
        print(f"REFUSED: {name} already scored on the locked set"); continue
    p = FinalEnsemble().fit(dev[feats], dev.aquatic_score).predict(test[feats])
    record(name, p, f"results/plunkett/oof_{name}_ensemble.parquet")
    test.assign(pred=p)[["crossing_code", "state", "huc8", "aquatic_score", "pred"]].to_csv(f"results/plunkett/locked_predictions_{k}.csv", index=False)
    # per-state breakdown
    done[name]["per_state"] = {s: cv.metrics(y[test.state == s], p[(test.state == s).values]) for s in test.state.unique() if (test.state == s).sum() >= 100}

# tiers scored in an earlier call: reload their saved locked predictions so paired deltas can still be computed
for k, (nm, _) in tiers.items():
    f = f"results/plunkett/locked_predictions_{k}.csv"
    if nm not in preds and os.path.exists(f):
        sv = pd.read_csv(f).set_index("crossing_code")["pred"]
        preds[nm] = sv.reindex(test["crossing_code"]).values

# cluster bootstrap (by HUC8) of R2 and paired differences, among tiers scored in THIS call
names = [n for n in preds if n != "T0_dev_mean"]
if len(names) >= 1:
    rng = np.random.default_rng(0); groups = test.huc8.values; ug = np.unique(groups)
    idx = {u: np.where(groups == u)[0] for u in ug}
    def r2(yy, pp): return 1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
    boots = {n: [] for n in names}
    for _ in range(2000):
        s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        for n in names: boots[n].append(r2(y.values[s], preds[n][s]))
    for n in names:
        if n not in done: continue
        done[n]["r2_ci95_cluster_bootstrap"] = [float(np.percentile(boots[n], 2.5)), float(np.percentile(boots[n], 97.5))]
    for a in names:
        for b in names:
            if a < b:
                dlt = np.array(boots[b]) - np.array(boots[a])
                done[f"delta_r2[{b}-{a}]"] = {"mean": float(dlt.mean()), "ci95": [float(np.percentile(dlt, 2.5)), float(np.percentile(dlt, 97.5))]}
done["_meta"] = {"n_locked": int(len(test)), "n_dev": int(len(dev)), "n_locked_huc8": int(test.huc8.nunique()), "locked_states": test.state.value_counts().to_dict()}
json.dump(done, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in done.items() if k.startswith("delta") or k.endswith("ci95_cluster_bootstrap")}, indent=1)[:1500])
