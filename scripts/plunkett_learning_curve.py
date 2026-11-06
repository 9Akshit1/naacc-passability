"""Would more labelled crossings help? A learning curve on development rows only.

It uses whole-HUC8 5-fold CV. The held-out fold is the same for every fraction and only the training part is thinned at random, to
25, 50, 75 and 100%. It uses T3 inputs, the fixed LightGBM regression model and seeds 0 to 2. If R2 flattens out by 75 to 100%,
more rows with the same inputs will not help much and the missing information is in the inputs and not in the number of rows.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from scripts.plunkett_streamcat_eval import reg
from src import plunkett_cv as cv, plunkett_features as pf

FRACS = (0.25, 0.5, 0.75, 1.0)
SEEDS = (0, 1, 2)

if __name__ == "__main__":
    d = pd.read_parquet("data/processed/plunkett_features_osm.parquet")
    d, _ = pf.add_derived(d)
    g = pf.feature_groups(d)
    g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]
    base = g["PLUNKETT"] + g["NBI"] + g["OSM"]
    dev = d[~cv.locked_test_mask(d)].reset_index(drop=True)
    y = dev["aquatic_score"]
    print(f"development rows {len(dev)}", flush=True)
    res = {f: [] for f in FRACS}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        oof = {f: pd.Series(np.nan, index=dev.index) for f in FRACS}
        for tr, te in cv.random_group_kfold(dev, "huc8", 5, seed):
            order = rng.permutation(np.asarray(tr))
            for f in FRACS:
                sub = order[: int(len(order) * f)]
                oof[f].loc[te] = reg(seed).fit(dev.loc[sub, base], y.loc[sub]).predict(dev.loc[te, base])
        for f in FRACS:
            res[f].append(r2_score(y, oof[f]))
        print(f"seed {seed}: " + "  ".join(f"{int(f * 100)}%={res[f][-1]:.4f}" for f in FRACS), flush=True)
    for f in FRACS:
        print(f"  {int(f * 100):3d}% of training rows (~{int(len(dev) * 0.8 * f)}): R2={np.mean(res[f]):.4f} (sd {np.std(res[f]):.4f})")
    print("done", flush=True)
