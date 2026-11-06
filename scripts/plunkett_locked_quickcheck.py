"""Score the outlet-drop test on the set-aside test basins. This is the second scoring, made after the first one was found to
keep an arbitrary pipe for crossings with several pipes.

The two models are:
    T5Q_v2  portable map inputs plus the yes/no answer q_perched
    T5M_v2  the same map inputs without q_perched, trained and scored on the same rows

The map inputs are Plunkett's road and income columns, the bridge inventory, OpenStreetMap, StreamCat, NHDPlus and the snap
distances, with no DSL raster inputs. q_perched is 1 when the surveyed outlet drop to the water surface is more than 0.25 ft.
Only crossings with a recorded drop are used, for training and for scoring. Crossings whose pipes disagree on the drop have no
value (see scripts/build_outlet_drop_join.py). One model is used for all states, the model is src/plunkett_final.FinalEnsemble,
and an entry that is already in locked_results.json is refused.

The first scoring (tiers T5Q_quickcheck_nodsl and T5M_same_rows_map_only, predictions in locked_predictions_T5.csv) is still in
locked_results.json. The basins have now been used twice for this test.

As a sensitivity check on the same fitted model, with no refit, the yes/no in the test rows is flipped at random with probability
e of 10%, 20% and 30%, with 20 draws each. This shows an assumed volunteer error and is not a measurement.
"""
import json
import os

import numpy as np
import pandas as pd

from scripts.plunkett_regions_eval import load
from src import plunkett_cv as cv
from src.plunkett_final import FinalEnsemble

OUT = "results/plunkett/locked_results.json"
done = json.load(open(OUT)) if os.path.exists(OUT) else {}
if "T5Q_v2_quickcheck_nodsl" in done or "T5M_v2_same_rows_map_only" in done:
    raise SystemExit("REFUSED: T5 v2 tiers already scored")

d, t3, nodsl = load(full=True)
lock = cv.locked_test_mask(d).values
has = d.q_perched.notna().values
dev, test = d[~lock & has].reset_index(drop=True), d[lock & has].reset_index(drop=True)
print(f"train rows {len(dev)} (states {dev.state.value_counts().to_dict()})\nlocked rows with a recorded drop {len(test)} "
      f"of {lock.sum()} (states {test.state.value_counts().to_dict()}), perched share {test.q_perched.mean():.3f}", flush=True)
y = test.aquatic_score.values


def r2(yy, pp):
    return 1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()


pM = FinalEnsemble().fit(dev[nodsl], dev.aquatic_score).predict(test[nodsl])
mQ = FinalEnsemble().fit(dev[nodsl + ["q_perched"]], dev.aquatic_score)
pQ = mQ.predict(test[nodsl + ["q_perched"]])

rng = np.random.default_rng(0)
ug = np.unique(test.huc8.values)
idx = {u: np.where(test.huc8.values == u)[0] for u in ug}
boot = [np.concatenate([idx[u] for u in rng.choice(ug, len(ug))]) for _ in range(2000)]
res = {}
for name, p in (("T5M_v2_same_rows_map_only", pM), ("T5Q_v2_quickcheck_nodsl", pQ)):
    m = cv.metrics(test.aquatic_score, p)
    lo, hi = np.quantile([r2(y[s], p[s]) for s in boot], [.025, .975])
    m.update(n_train=int(len(dev)), n_test=int(len(test)), r2_ci95=[float(lo), float(hi)])
    res[name] = m
dl = np.array([r2(y[s], pQ[s]) - r2(y[s], pM[s]) for s in boot])
res["T5Q_v2_quickcheck_nodsl"]["delta_r2_vs_T5M"] = {"point": float(r2(y, pQ) - r2(y, pM)), "ci95": [float(np.quantile(dl, .025)), float(np.quantile(dl, .975))]}
sens = {}
for e in (0.1, 0.2, 0.3):
    vals = []
    for k in range(20):
        r = np.random.default_rng(100 + k)
        q = np.where(r.random(len(test)) < e, 1 - test.q_perched.values, test.q_perched.values)
        vals.append(r2(y, mQ.predict(test[nodsl + ["q_perched"]].assign(q_perched=q))))
    sens[str(e)] = [float(np.mean(vals)), float(np.std(vals))]
res["T5Q_v2_quickcheck_nodsl"]["answer_error_sensitivity_test_only"] = sens
by_state = {s: {"n": int(m.sum()), "T5M": float(r2(y[m], pM[m])), "T5Q": float(r2(y[m], pQ[m]))}
            for s in sorted(test.state.unique()) for m in [(test.state == s).values] if m.sum() >= 100}
res["T5Q_v2_quickcheck_nodsl"]["by_state_n_ge_100"] = by_state
for k, v in res.items():
    print(k, json.dumps(v, indent=1, default=float), flush=True)
done.update(res)
json.dump(done, open(OUT, "w"), indent=1)
test.assign(pred_map_only=pM, pred_quickcheck=pQ)[["crossing_code", "state", "huc8", "aquatic_score", "q_perched", "pred_map_only", "pred_quickcheck"]].to_csv(
    "results/plunkett/locked_predictions_T5v2.csv", index=False)
print("done")
