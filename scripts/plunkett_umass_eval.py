"""Do the older UMass crossings change the outlet-drop results? Development rows only; the set-aside basins are not read.

    python -m scripts.plunkett_umass_eval

Part 1  Does a flag for "older UMass crossing" help the map-only model? The flag is 1 for the 5,403 UMass crossings and 0 for
        the rest. A new crossing would always get 0.
Part 2  The outlet-drop yes/no (is the drop more than 0.25 ft?) for UMass crossings comes from the midpoint of the range
        (scripts/build_outlet_drop_umass.py). A range of 1-6 inches holds the 3 inch cut-off, so it is run four ways: as it
        falls with the midpoint, always "no", always "yes", and left out. Then
          a  model trained on current-survey rows only, scored on the UMass rows
          b  model trained on current plus UMass rows with the older-crossing flag, scored on both
          c  the same rows with the map inputs only

Splits hold out whole HUC8 watersheds, 5 folds, and every row is predicted by a model that never saw its watershed. Differences
are given with a 95% interval from a bootstrap over watersheds.
"""
import numpy as np
import pandas as pd

from scripts.plunkett_regions_eval import SEED, fitpred, load, r2
from src import plunkett_cv as cv

N_BOOT = 300
AMBIG = ["mid", "no", "yes", "left_out"]


def boot_diff(y, p1, p0, groups, n=N_BOOT):
    """Difference in R2 (p1 minus p0) with a 95% interval, resampling watersheds."""
    rng = np.random.default_rng(SEED)
    codes, idx = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(idx == k) for k in range(len(codes))]
    d = []
    for _ in range(n):
        pick = np.concatenate([members[k] for k in rng.integers(0, len(codes), len(codes))])
        d.append(r2(y[pick], p1[pick]) - r2(y[pick], p0[pick]))
    return r2(y, p1) - r2(y, p0), np.percentile(d, 2.5), np.percentile(d, 97.5)


def line(label, y, p, groups=None, base=None):
    s = f"  {label:58s} R2 {r2(y, p):.4f}  (n={len(y)})"
    if base is not None:
        d, lo, hi = boot_diff(y, p, base, groups)
        s += f"   vs reference {d:+.4f} ({lo:+.4f} .. {hi:+.4f})"
    print(s, flush=True)


def folds(x):
    return [(np.asarray(a), np.asarray(b)) for a, b in cv.random_group_kfold(x, "huc8", 5, SEED)]


def oof(x, feats, rows_train, fold_list):
    """Out-of-fold predictions for every row of x, training only on rows_train inside each training fold."""
    out = np.full(len(x), np.nan)
    for tr, te in fold_list:
        tr = tr[rows_train[tr]]
        out[te] = fitpred(x, tr, te, feats)
    return out


if __name__ == "__main__":
    dev, t3, nodsl = load(drop_source="export")
    um = pd.read_parquet("data/processed/plunkett_outlet_drop_umass.parquet")
    um = um.rename(columns={"drop_ws": "um_drop"})
    x = dev.merge(
        um[["crossing_code", "um_drop", "from_range", "straddles_3in"]], on="crossing_code", how="left").reset_index(drop=True)
    x["legacy"] = (x.database == "umass").astype(float)
    x["from_range"] = x.from_range.eq(True)
    x["straddles_3in"] = x.straddles_3in.eq(True)
    y = x.aquatic_score.values
    g = x.huc8.values
    f = folds(x)
    print(f"development rows {len(x)}; older UMass crossings {int(x.legacy.sum())}; "
          f"with a usable UMass drop {int(x.um_drop.notna().sum())} (ranges {int((x.from_range & x.um_drop.notna()).sum())}, "
          f"of which hold the 3 inch cut-off {int(x.straddles_3in.sum())})", flush=True)
    print(f"current-survey rows with an outlet drop {int(((x.legacy == 0) & x.drop_ws.notna()).sum())}", flush=True)
    umr, nar = (x.legacy == 1).values, (x.legacy == 0).values

    print("\nPART 1  map-only model, all development rows", flush=True)
    allrows = np.ones(len(x), bool)
    p_plain = oof(x, nodsl, allrows, f)
    p_flag = oof(x, nodsl + ["legacy"], allrows, f)
    for name, m in (("all rows", allrows), ("current-survey rows", nar), ("older UMass rows", umr)):
        line(f"without flag, {name}", y[m], p_plain[m])
        line(f"with flag,    {name}", y[m], p_flag[m], g[m], base=p_plain[m])

    print("\nPART 2  outlet-drop yes/no with UMass drops (drop_ws = current surveys, um_drop = UMass midpoint)", flush=True)
    cur = (nar & x.drop_ws.notna().values)
    for v in AMBIG:
        q = np.where(umr, np.where(x.um_drop.notna(), (x.um_drop > 0.25).astype(float), np.nan), x.q_perched)
        if v == "no":
            q = np.where(umr & x.straddles_3in.values, 0.0, q)
        if v == "yes":
            q = np.where(umr & x.straddles_3in.values, 1.0, q)
        if v == "left_out":
            q = np.where(umr & x.straddles_3in.values, np.nan, q)
        xv = x.assign(q_perched=q)
        has = xv.q_perched.notna().values
        use_um = umr & has
        use_cur = cur
        print(f"\n[1-6 range handled as: {v}]  UMass rows with the yes/no {int(use_um.sum())}, perched share {np.nanmean(q[use_um]):.3f}", flush=True)
        rows_a = use_cur
        pa = oof(xv, nodsl + ["q_perched"], rows_a, f)
        rows_b = use_cur | use_um
        pb = oof(xv, nodsl + ["q_perched", "legacy"], rows_b, f)
        pc = oof(xv, nodsl + ["legacy"], rows_b, f)
        for name, m in (("older UMass rows", use_um), ("current-survey rows", use_cur)):
            line(f"c  map only (trained on both),            {name}", y[m], pc[m])
            line(f"a  +yes/no, trained on current only,      {name}", y[m], pa[m], g[m], base=pc[m])
            line(f"b  +yes/no, trained on both with flag,    {name}", y[m], pb[m], g[m], base=pc[m])
        by = pd.DataFrame({"state": xv.state[use_um].values, "y": y[use_um], "pc": pc[use_um], "pb": pb[use_um]})
        rows = [f"{s} n={len(d)} {r2(d.y, d.pc):.2f}->{r2(d.y, d.pb):.2f}" for s, d in by.groupby("state") if len(d) >= 40]
        print("  older UMass rows by state, map only -> with yes/no (b): " + "; ".join(rows), flush=True)
    print("done", flush=True)
