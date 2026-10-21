"""Emulate the documented DSL pipeline: Random Forest + ADDITIVE STEPWISE variable selection that maximises the
out-of-bag R2 (DSL doc, section 2.2 step 2). Variables are added as groups (the scale-dependent variables come as a set
of window sizes). Fitted on the development set only; then (a) its OOB R2 (the metric DSL quotes), (b) HUC8-group CV R2
of the selected forest (honest spatial metric), (c) the locked-test R2 (one look; a NEW reference model, scored once).
Reduced forest size (100 trees, 40% row subsample) so the ~55 selection fits are tractable - a limitation."""
import json, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src import plunkett_cv as cv, plunkett_features as pf
d = pd.read_parquet("data/processed/plunkett_features_osm.parquet"); lock = cv.locked_test_mask(d); dev, test = d[~lock], d[lock]
P = pf.feature_groups(d)["PLUNKETT"]
groups = {"d8accum": ["d8accum"], "gradient": ["gradient"], "dem": ["dem"], "adt": ["adt"], "class": ["class"], "income": ["income"],
          "elev_range": [c for c in P if c.startswith("elev_range")], "incise1": [c for c in P if c.startswith("incise1")],
          "elev_sd": [c for c in P if c.startswith("elev_sd")], "bdev": [c for c in P if c.startswith("bdev")]}
assert sorted(sum(groups.values(), [])) == sorted(P)
rf = lambda n=100: RandomForestRegressor(n_estimators=n, max_features=0.33, min_samples_leaf=1, max_samples=0.4, oob_score=True, n_jobs=10, random_state=0)
def oob(cols):
    m = rf().fit(dev[cols], dev.aquatic_score); return m.oob_score_
chosen, cols, best, log = [], [], -1, []
while True:
    cand = {}
    for g, c in groups.items():
        if g in chosen: continue
        t = time.time(); cand[g] = oob(cols + c)
    if not cand: break
    g = max(cand, key=cand.get)
    log.append({"added": g, "oob_r2": cand[g], "all_candidates": cand}); print(f"step {len(chosen)+1}: add {g:10s} OOB R2={cand[g]:.4f}", flush=True)
    if cand[g] < best + 0.001: log[-1]["stopped_no_gain"] = True; break
    chosen.append(g); cols += groups[g]; best = cand[g]
print("selected groups:", chosen, " n columns:", len(cols), " OOB R2:", round(best, 4))
final = RandomForestRegressor(n_estimators=300, max_features=0.33, min_samples_leaf=1, oob_score=True, n_jobs=10, random_state=0).fit(dev[cols], dev.aquatic_score)
mk = lambda: RandomForestRegressor(n_estimators=150, max_features=0.33, min_samples_leaf=1, max_samples=0.5, n_jobs=10, random_state=0)
sp = cv.summarize(cv.cross_val_oof(dev, cols, mk, "huc8", seeds=(0, 1)))
out = {"selected_groups": chosen, "n_columns": len(cols), "oob_r2_300_trees": float(final.oob_score_), "huc8_cv_r2": sp["r2_mean"],
       "locked_r2": cv.metrics(test.aquatic_score, final.predict(test[cols]))["r2"], "log": log}
json.dump(out, open("results/plunkett/pipeline_emulation.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "log"})
