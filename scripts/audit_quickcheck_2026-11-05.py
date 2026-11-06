"""Independent checks on the outlet-drop test described in reports/OUTLET_DROP.md. It only reads files and writes nothing except its own log.

 1  Recompute the test-basin scores from the saved prediction files with scikit-learn, and compare them with locked_results.json.
 2  Check that the test crossings are the same as in the main test, that no crossing, HUC6 basin or HUC8 watershed is on both
    sides of the split, and that no crossing code is repeated.
 3  Look at the outlet-drop join: why survey-id matches were rejected, whether crossings without a drop differ from the rest,
    and whether the accepted values look reasonable.
 4  List every model input, flag names that look like survey fields, and flag any input whose correlation with the score or
    with the yes/no answer is above 0.9.
 5  Show how much of the gain is arithmetic: the yes/no alone, the exact drop, the map, the map with the yes/no, and the map
    with a shuffled yes/no as a control.
"""
import json
import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from scripts.plunkett_regions_eval import load
from src import plunkett_cv as cv

R = "results/plunkett/"
print("=== 1. locked numbers recomputed independently", flush=True)
lr = json.load(open(R + "locked_results.json"))
print("keys in locked_results.json:", list(lr))
for f in ("T3", "T4", "T4P", "T5"):
    try:
        p = pd.read_csv(R + f"locked_predictions_{f}.csv")
    except FileNotFoundError:
        print(" missing", f); continue
    if f == "T5":
        for col in ("pred_map_only", "pred_quickcheck"):
            y, q = p.aquatic_score.values, p[col].values
            print(f" T5 {col:16s} n={len(p)} R2(sklearn)={r2_score(y, q):.4f} MAE={np.abs(y - q).mean():.4f} Spearman={spearmanr(y, q)[0]:.4f}  "
                  f"pred range {q.min():.3f}..{q.max():.3f} NaN {np.isnan(q).sum()}")
    else:
        y, q = p.aquatic_score.values, p.pred.values
        print(f" {f:4s} n={len(p)} R2(sklearn)={r2_score(y, q):.4f} MAE={np.abs(y - q).mean():.4f} NaN {np.isnan(q).sum()}")
for k, v in lr.items():
    if isinstance(v, dict) and "r2" in v:
        print(f"  json {k:28s} r2={v['r2']:.4f} n={v.get('n')}")

print("\n=== 2. locked / train separation", flush=True)
d, t3, nodsl = load(full=True)
lock = cv.locked_test_mask(d).values
t3codes = set(pd.read_csv(R + "locked_predictions_T3.csv").crossing_code)
mask_codes = set(d.crossing_code[lock])
print(f" locked by mask {lock.sum()}, in T3 predictions file {len(t3codes)}, identical sets: {t3codes == mask_codes}")
print(" duplicate crossing codes in whole table:", int(d.crossing_code.duplicated().sum()))
has = d.q_perched.notna().values
tr, te = d[~lock & has], d[lock & has]
print(f" train {len(tr)}, test {len(te)}; shared crossing codes: {len(set(tr.crossing_code) & set(te.crossing_code))}; "
      f"shared HUC6: {sorted(set(tr.huc6) & set(te.huc6))}; shared HUC8: {sorted(set(tr.huc8) & set(te.huc8))}")
t5 = pd.read_csv(R + "locked_predictions_T5.csv")
print(f" T5 rows are exactly the locked rows with a drop: {set(t5.crossing_code) == set(te.crossing_code)}")
print(" locked HUC6 basins:", sorted(d.huc6[lock].astype(str).unique()))

print("\n=== 3. join audit", flush=True)
j = pd.read_parquet("data/processed/plunkett_outlet_drop.parquet")
print(" accepted by pass:", j.match_pass.value_counts(dropna=False).to_dict())
print(" drop_ws summary (ft):", j.drop_ws.describe().round(3).to_dict())
print(f" share > 0.25 ft: {(j.drop_ws > 0.25).sum() / j.drop_ws.notna().sum():.3f}; share exactly 0: {(j.drop_ws == 0).sum() / j.drop_ws.notna().sum():.3f}")
n = pd.read_parquet("data/raw/naacc_public_layer.parquet")
nn = pd.DataFrame({"sid": n.Survey_Id.astype("Int64"), "lat": n["_lat"], "lon": n["_lon"], "sc": pd.to_numeric(n.Aqua_Pass_, errors="coerce")})
dd = d[["crossing_code", "crossing_survey_id", "aquatic_score", "lat_nad83", "lon_nad83", "state"]].copy()
dd["sid"] = dd.crossing_survey_id.astype("Int64")
m = dd.merge(nn, on="sid", suffixes=("", "_l"))
m["km"] = np.hypot((m.lat_nad83 - m.lat) * 110.54, (m.lon_nad83 - m.lon) * 111.32 * np.cos(np.radians(m.lat_nad83)))
m["sd"] = (m.aquatic_score - m.sc).abs()
print(f" id matches {len(m)}: within 30 m {int((m.km <= .03).sum())}, score diff < 0.02 {int((m.sd < .02).sum())}, both {int(((m.km <= .03) & (m.sd < .02)).sum())}")
print(f" rejected only by score (location ok): {int(((m.km <= .03) & (m.sd >= .02)).sum())}; only by location: {int(((m.km > .03) & (m.sd < .02)).sum())}; both: {int(((m.km > .03) & (m.sd >= .02)).sum())}")
rs = m[(m.km <= .03) & (m.sd >= .02)]
print(f"   of those score-rejected, median score diff {rs.sd.median():.3f}, mean Plunkett score {rs.aquatic_score.mean():.3f} vs layer {rs.sc.mean():.3f}")
print(" aquatic_score: crossings WITH recorded drop vs WITHOUT (selection check)")
print(d.groupby(has).aquatic_score.describe().round(3).to_string())
print(" share with drop by state (all rows):", d.groupby("state").apply(lambda g: round(g.q_perched.notna().mean(), 2)).to_dict())

print("\n=== 4. inputs", flush=True)
print(f" {len(nodsl)} portable inputs. non-statcat/vaa names:")
other = [c for c in nodsl if not c.startswith(("sc_", "vaa_"))]
print("  ", other)
sus = [c for c in nodsl if re.search(r"score|passab|outlet|inlet|perch|drop|barrier|aop|survey|structure|material|slope_perc|culvert", c, re.I)]
print(" survey-like names among inputs:", sus)
dev = d[~lock].reset_index(drop=True)
num = dev[nodsl].apply(pd.to_numeric, errors="coerce")
c_y = num.corrwith(dev.aquatic_score).abs().sort_values(ascending=False)
c_q = num.corrwith(dev.q_perched).abs().sort_values(ascending=False)
print(" top |corr| with aquatic_score:", c_y.head(5).round(3).to_dict())
print(" top |corr| with q_perched:   ", c_q.head(5).round(3).to_dict())
print(" any input identical to label?", [c for c in nodsl if num[c].notna().sum() > 100 and (num[c] == dev.aquatic_score).mean() > .9])

print("\n=== 5. is the gain real? development rows with a recorded drop, HUC8-grouped CV, seeds 0,1", flush=True)
x = dev[dev.q_perched.notna()].reset_index(drop=True)
x["drop_c"] = x.drop_ws
rng = np.random.default_rng(0)
x["q_shuf"] = x.q_perched.values[rng.permutation(len(x))]
x["q_shuf_state"] = x.groupby("state").q_perched.transform(lambda s: s.values[rng.permutation(len(s))])
mk = lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.7, reg_lambda=2, verbose=-1, n_jobs=10)
for name, f in [("yes/no alone", ["q_perched"]), ("continuous drop alone (ft)", ["drop_c"]), ("map only", nodsl),
                ("map + yes/no", nodsl + ["q_perched"]), ("map + SHUFFLED yes/no (control)", nodsl + ["q_shuf"]),
                ("map + yes/no shuffled within state (control)", nodsl + ["q_shuf_state"]), ("map + continuous drop", nodsl + ["drop_c"])]:
    s = cv.summarize(cv.cross_val_oof(x, f, mk, "huc8", seeds=(0, 1)))
    print(f" {name:48s} R2={s['r2_mean']:.4f} (sd {s['r2_sd']:.4f})  MAE={s['mae_mean']:.4f}", flush=True)
print(" label mean by yes/no:", x.groupby("q_perched").aquatic_score.agg(["size", "mean", "std"]).round(3).to_dict("index"))
print("done")
