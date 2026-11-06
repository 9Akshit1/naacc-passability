"""Do the extra labelled crossings from SARP's compilation help, and does a model carry over to states with no NAACC training data?

The inputs are public-data features only, because the extra points have no DSL rasters, bridge inventory or OpenStreetMap features:
snap_m, snap2_m, order_line, ftype_line, StreamCat (ws and cat) and NHDPlus value added attributes.
A is the Plunkett development rows, labelled with the NAACC aquatic score. B is the extra SARP rows, labelled with SARP_Score, which
follows the NAACC score with Spearman 0.98 on 52,000 shared crossings. The Plunkett test basins are neither trained on nor scored
here, and B excludes them too. It uses the fixed LightGBM regression model and seeds 0 to 2.

  E1  train on A only and score B, (a) in the 13 NAACC states and (b) in other states, to see transfer to unseen rows and regions
  E2  pooled HUC8 grouped 5-fold CV on A and B: a model trained on the A part only against one trained on the A and B parts, scored on
      the same held-out A, B (13 states) and B (other) rows, to see whether extra rows help
  E3  leave one state out for the 6 biggest non-NAACC states: train on A only, and on A plus B in the other states, and score the
      held-out state
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from scripts.plunkett_streamcat_eval import reg
from src import plunkett_cv as cv, plunkett_features as pf, streamcat_feats as sf

SEEDS = (0, 1, 2)
STATES13 = {"Connecticut", "Delaware", "Massachusetts", "Maryland", "Maine", "New Hampshire", "New Jersey", "New York", "Pennsylvania",
            "Rhode Island", "Virginia", "Vermont", "West Virginia"}
BASE_SNAP = ["snap_m", "snap2_m", "order_line", "ftype_line"]


def ms(x):
    return f"{np.mean(x):.4f}"


if __name__ == "__main__":
    d = pd.read_parquet("data/processed/plunkett_features_osm.parquet")
    d, _ = pf.add_derived(d)
    d = pd.concat([d, sf.build(d).drop(columns="comid")], axis=1)
    A = d[~cv.locked_test_mask(d)].reset_index(drop=True)
    B = pd.read_parquet("data/interim/sarp_new_features.parquet")
    P = BASE_SNAP + [c for c in B.columns if c.startswith(("sc_", "vaa_"))]
    P = [c for c in P if c in A.columns]
    print(f"A rows {len(A)}, B rows {len(B)}, public features {len(P)}", flush=True)
    A = pd.DataFrame({"y": A["aquatic_score"].values, "huc8": A["huc8"].astype(str).values, "region": "A", "state": A["state"].values}).join(A[P])
    B = B.assign(y=B["SARP_Score"], huc8=B["HUC8"].astype(str), state=B["State"],
                 region=np.where(B["State"].isin(STATES13), "B13", "Bother"))[["y", "huc8", "region", "state"] + P]
    print("B by region:", B.region.value_counts().to_dict(), "| label mean A %.3f, B13 %.3f, Bother %.3f" % (
        A.y.mean(), B[B.region == "B13"].y.mean(), B[B.region == "Bother"].y.mean()), flush=True)

    def report(name, y, p):
        print(f"  {name:48s} n={len(y):6d}  R2={r2_score(y, p):.4f}  Spearman={spearmanr(y, p)[0]:.3f}", flush=True)

    print("\nE1: train on A only, score B", flush=True)
    preds = [reg(s).fit(A[P], A.y).predict(B[P]) for s in SEEDS]
    p = np.mean(preds, axis=0)
    for r in ("B13", "Bother"):
        m = (B.region == r).values
        report(f"A -> {r}", B.y[m], p[m])
    for st in B[B.region == "Bother"].state.value_counts().head(8).index:
        m = (B.state == st).values
        report(f"A -> {st}", B.y[m], p[m])

    print("\nE2: pooled grouped CV, A-train only versus A-train + B-train", flush=True)
    AB = pd.concat([A, B], ignore_index=True)
    acc = {k: [] for k in ("A|A", "A|B13", "A|Bo", "AB|A", "AB|B13", "AB|Bo")}
    for seed in SEEDS:
        oofA = pd.Series(np.nan, index=AB.index); oofAB = pd.Series(np.nan, index=AB.index)
        for tr, te in cv.random_group_kfold(AB, "huc8", 5, seed):
            trA = [i for i in tr if AB.region.iat[i] == "A"]
            oofA.loc[te] = reg(seed).fit(AB.loc[trA, P], AB.y.loc[trA]).predict(AB.loc[te, P])
            oofAB.loc[te] = reg(seed).fit(AB.loc[tr, P], AB.y.loc[tr]).predict(AB.loc[te, P])
        for tag, oof in (("A", oofA), ("AB", oofAB)):
            for key, mk in (("A", "A"), ("B13", "B13"), ("Bo", "Bother")):
                m = (AB.region == mk).values
                acc[f"{tag}|{key}"].append(r2_score(AB.y[m], oof[m]))
        print(f"  seed {seed}: " + "  ".join(f"{k}={v[-1]:.4f}" for k, v in acc.items()), flush=True)
    for key in ("A", "B13", "Bo"):
        a, ab = acc[f"A|{key}"], acc[f"AB|{key}"]
        print(f"  scored on {key:4s}: trained on A only R2={ms(a)}   trained on A+B R2={ms(ab)}   change {np.mean(ab) - np.mean(a):+.4f}")

    print("\nE3: leave one state out (non-NAACC states)", flush=True)
    for st in B[B.region == "Bother"].state.value_counts().head(6).index:
        te = (AB.state == st).values
        r_a, r_ab = [], []
        for s in SEEDS:
            r_a.append(r2_score(AB.y[te], reg(s).fit(AB.loc[AB.region == "A", P], AB.y[AB.region == "A"]).predict(AB.loc[te, P])))
            trn = ~te                      # A plus every B row outside the held-out state
            r_ab.append(r2_score(AB.y[te], reg(s).fit(AB.loc[trn, P], AB.y[trn]).predict(AB.loc[te, P])))
        print(f"  {st:16s} n={te.sum():5d}  A only R2={ms(r_a)}   A + other-state B R2={ms(r_ab)}   change {np.mean(r_ab) - np.mean(r_a):+.4f}", flush=True)
    print("done", flush=True)
