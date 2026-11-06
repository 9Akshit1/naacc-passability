"""Do the scores of surveyed crossings upstream and downstream on the same stream help to fill a gap?

This is the common real case of an unsurveyed crossing in a watershed that already has surveyed crossings. Whole-watershed
holdouts say nothing about it. Here crossings are held out at random, one fifth at a time, and the model may use the scores of
surveyed crossings in the training parts only, so a row never sees its own score.

The neighbour inputs compared (T3 inputs plus):
  radius  the earlier summaries of the nearest 5 and 15 crossings (src/plunkett_features.neighbour_features)
  stream  the next surveyed crossing downstream and upstream on the same NHDPlus level path, with its score and its distance along
          the stream, plus the mean score of surveyed crossings on the same stream segment and level path
  both    radius and stream together
Crossings are linked with NLDI COMIDs (scripts/fetch_comids.py), and a crossing more than 100 m from a mapped stream gets no
stream features. Only development rows are used, with the fixed LightGBM regression model and seeds 0 to 2. A random holdout is
optimistic for real gaps, so the results are upper bounds.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src import plunkett_cv as cv, plunkett_features as pf, streamcat_feats as sf
from src.plunkett_final import LGBM_REG

SEEDS = (0, 1, 2)
STREAM_COLS = ["ss_down_score", "ss_down_km", "ss_up_score", "ss_up_km", "ss_reach_mean", "ss_reach_n", "ss_path_mean", "ss_path_n"]
MAX_SNAP = 100.0


def reg(seed):
    return lgb.LGBMRegressor(subsample_freq=1, verbose=-1, n_jobs=10, random_state=seed, **LGBM_REG)


def stream_features(ref: pd.DataFrame, qry: pd.DataFrame) -> pd.DataFrame:
    """Label summaries of ref crossings on the same stream as each qry crossing. ref = training rows only."""
    out = np.full((len(qry), len(STREAM_COLS)), np.nan)
    r = ref[(ref.snap_m <= MAX_SNAP) & ref.path.notna()]
    by_path = {k: g for k, g in r.groupby("path")}
    by_reach = {k: g for k, g in r.groupby("comid")}
    for i, row in enumerate(qry.itertuples()):
        if not (row.snap_m <= MAX_SNAP) or pd.isna(row.path):
            continue
        sid = row.crossing_survey_id
        g = by_path.get(row.path)
        if g is not None:
            g = g[g.crossing_survey_id != sid]
            if len(g):
                up_stream_km = g.pathlen.values - row.pathlen          # > 0: the other crossing is further upstream
                s = g.aquatic_score.values
                dn = up_stream_km < 0
                up = ~dn
                if dn.any():
                    j = np.argmax(np.where(dn, up_stream_km, -np.inf))
                    out[i, 0], out[i, 1] = s[j], -up_stream_km[j]
                if up.any():
                    j = np.argmin(np.where(up, up_stream_km, np.inf))
                    out[i, 2], out[i, 3] = s[j], up_stream_km[j]
                out[i, 6], out[i, 7] = s.mean(), len(s)
        g = by_reach.get(row.comid)
        if g is not None:
            g = g[g.crossing_survey_id != sid]
            if len(g):
                out[i, 4], out[i, 5] = g.aquatic_score.mean(), len(g)
    return pd.DataFrame(out, index=qry.index, columns=STREAM_COLS)


if __name__ == "__main__":
    d = pd.read_parquet("data/processed/plunkett_features_osm.parquet")
    d, _ = pf.add_derived(d)
    g = pf.feature_groups(d)
    g["OSM"] = [c for c in g["OSM"] if c != "osm_road_culvert_flag"]
    base = g["PLUNKETT"] + g["NBI"] + g["OSM"]
    ex = sf.build(d)[["comid", "snap_m"]]
    vaa = pd.read_parquet(sf.VAA).drop_duplicates("comid").set_index("comid")[["levelpathi", "pathlength"]]
    ex["path"] = ex["comid"].map(vaa["levelpathi"])
    ex["pathlen"] = ex["comid"].map(vaa["pathlength"])
    d = pd.concat([d, ex], axis=1)
    dev = d[~cv.locked_test_mask(d)].copy().reset_index(drop=True)
    y = dev["aquatic_score"]
    print(f"development rows {len(dev)}; on a stream within {MAX_SNAP:.0f} m: {(dev.snap_m <= MAX_SNAP).mean():.3f}", flush=True)

    res = {k: [] for k in ("no neighbour scores", "radius neighbours", "stream neighbours", "radius + stream")}
    cover = []
    for seed in SEEDS:
        oof = {k: pd.Series(np.nan, index=dev.index) for k in res}
        for tr, te in cv.random_group_kfold(dev.assign(_row=np.arange(len(dev)).astype(str)), "_row", 5, seed):   # random rows
            train, test = dev.loc[tr].copy(), dev.loc[te].copy()
            rad_tr = pf.neighbour_features(train, train, exclude_same_group=None)
            rad_te = pf.neighbour_features(train, test, exclude_same_group=None)
            st_tr = stream_features(train, train)
            st_te = stream_features(train, test)
            cover.append(st_te["ss_path_n"].notna().mean())
            sets = {"no neighbour scores": (train[base], test[base]),
                    "radius neighbours": (train[base].join(rad_tr), test[base].join(rad_te)),
                    "stream neighbours": (train[base].join(st_tr), test[base].join(st_te)),
                    "radius + stream": (train[base].join(rad_tr).join(st_tr), test[base].join(rad_te).join(st_te))}
            for k, (a, b) in sets.items():
                oof[k].loc[te] = reg(seed).fit(a, y.loc[tr]).predict(b)
        for k in res:
            res[k].append(r2_score(y, oof[k]))
        print(f"seed {seed}: " + "  ".join(f"{k}={v[-1]:.4f}" for k, v in res.items()), flush=True)
    print(f"\nheld-out crossings with at least one surveyed crossing on the same stream path: {np.mean(cover):.3f}")
    ref = np.mean(res["no neighbour scores"])
    for k, v in res.items():
        print(f"  {k:24s} R2={np.mean(v):.4f}  ({np.mean(v) - ref:+.4f})  seeds {[round(x, 4) for x in v]}")
    print("done", flush=True)
