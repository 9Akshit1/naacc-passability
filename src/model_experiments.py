"""Run the modelling experiments and write results/ + reports/03_model_comparison.md.

All on the NY inland subset (see METHODOLOGY.md for scope). Experiments:

* E0  baselines + model ladder, random k-fold vs HUC8 group k-fold, REPEATED
      (N_REPEATS different random watershed-to-fold assignments, mean +/- std
      reported - not a single split's number; see 2026-09-22 note below)
* E1  catchment-area only (RF on drainage area) - how far does the "big stream
      -> bridge -> passable" heuristic get you?
* E2  full model with catchment removed
* E3  residual model: f_catchment(x) + f_environment(residual)
* E4  permutation importance (spatial CV)
* E5  conformal prediction intervals (group-aware calibration)
* E6  calibration + predict-toward-the-mean check vs the DSL-reported pathology
* E7  prioritisation: does the environment model reorder the "worst crossings"?
* E8  leave-one-HUC8-out: per-watershed breakdown of the full model

2026-09-22: two methodology fixes, both aimed at "is the headline number
trustworthy," not at the predictors themselves.
1. Repeated spatial CV (E0) - previously a SINGLE ``GroupKFold`` split (worse:
   sklearn's ``GroupKFold`` in this project's sklearn version has no
   ``random_state`` at all, so every "re-run" was silently the exact same
   split). Given how much the headline R2 swung run to run across this
   project's history on genuinely different data, one split was never enough
   to trust on its own.
2. Hyperparameter tuning (``models.py::tune_hyperparameters``) - every tree/
   boosting model previously ran on hand-picked defaults; there was no
   ``GridSearchCV``/``RandomizedSearchCV`` anywhere in this codebase before
   this date.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from .evaluate import (calibration_table, cv_predict, prioritization_overlap,
                       regression_metrics)
from .models import catchment_only_model, make_models, tune_hyperparameters
from .spatial_cv import huc8_group_kfold, leave_one_huc8_out, random_kfold, repeated_huc8_group_kfold
from .uncertainty import conformal_intervals, evaluate_coverage

SUBSET = Path("data/processed/subset_predictors.parquet")
RESULTS = Path("results")
TABLES = RESULTS / "tables"
FIG = RESULTS / "figures"
REP = Path("reports")
TARGET = "passability"
SEED = 20260910
TUNING_SEED = SEED + 10_000  # deliberately different from the reporting splits' seeds
N_REPEATS = 5                # independent spatial-CV splits averaged for the headline E0 number
DSL_BASELINE_R2 = 0.31  # Plunkett et al. 2022, OOB R2 of the DSL GIS model


def load_subset() -> pd.DataFrame:
    df = pd.read_parquet(SUBSET).reset_index(drop=True)
    df = df[df[TARGET].between(0, 1)].copy()
    # drop rows with no drainage area AND no elevation (nothing to model on)
    df = df[df[["drainage_area_km2", "elev_m"]].notna().any(axis=1)].reset_index(drop=True)
    df["road_class"] = df["road_class"].fillna("unknown").astype(str)
    return df


def _splits(df, kind, seed=SEED):
    return list(random_kfold(df, 5, seed) if kind == "random"
               else huc8_group_kfold(df, "huc8_name", min(df["huc8_name"].nunique(), 7), seed=seed))


def _repeated_metrics(model, df, kind, target=TARGET, n_repeats=N_REPEATS, seed=SEED):
    """Fit/evaluate ``model`` across ``n_repeats`` independent splits (spatial:
    different watershed-to-fold assignments; random: different row shuffles)
    and return per-metric mean + std across repeats, plus the raw per-repeat
    values (so the report can show the spread, not just a point estimate)."""
    per_repeat = []
    for r in range(n_repeats):
        splits = (list(repeated_huc8_group_kfold(df, "huc8_name",
                                                  min(df["huc8_name"].nunique(), 7),
                                                  1, seed=seed + r))[0]
                 if kind == "spatial" else _splits(df, "random", seed=seed + r))
        oof = cv_predict(clone(model), df, splits, target)
        per_repeat.append(regression_metrics(df[target], oof))
    agg = {}
    for k in per_repeat[0]:
        vals = [p[k] for p in per_repeat if np.isfinite(p[k])]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
        agg[f"{k}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
    agg["_per_repeat"] = per_repeat
    return agg


def run() -> dict:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_subset()
    out: dict = {"n_crossings": len(df), "n_huc8": int(df["huc8_name"].nunique()),
                 "huc8_counts": df["huc8_name"].value_counts().to_dict(),
                 "target_mean": float(df[TARGET].mean()), "target_sd": float(df[TARGET].std())}

    # One search pass per tunable model, scored under its own HUC8-group CV
    # (seeded independently of the reporting splits below). A wider search
    # (n_iter 40, inner_splits 5) was tried and discarded: it stabilized which
    # model wins the secondary leave-one-HUC8-out diagnostic but didn't move
    # the actual deployed ensemble's accuracy, at ~5x the runtime. See
    # METHODOLOGY.md for the numbers.
    print("tuning hyperparameters...", flush=True)
    tuned_params = tune_hyperparameters(df, seed=TUNING_SEED, n_iter=15, inner_splits=4)
    out["tuned_hyperparameters"] = tuned_params
    print(f"  tuned: { {k: v for k, v in tuned_params.items()} }", flush=True)

    def models_(numeric=None, categorical=None, seed=SEED):
        return make_models(numeric=numeric, categorical=categorical, seed=seed, tuned_params=tuned_params)

    # ---- E0: model ladder, random vs spatial, REPEATED --------------
    ladder = {}
    for kind in ("random", "spatial"):
        for name, model in models_().items():
            ladder[(kind, name)] = _repeated_metrics(model, df, kind, n_repeats=N_REPEATS, seed=SEED)
    ladder_df = pd.DataFrame({f"{k[0]}/{k[1]}": {kk: vv for kk, vv in v.items() if kk != "_per_repeat"}
                              for k, v in ladder.items()}).T
    ladder_df.to_csv(TABLES / "e0_model_ladder.csv")
    out["e0_model_ladder"] = ladder_df.round(4).to_dict("index")

    # voting_ensemble (2026-09-27) is included in the E0 ladder above - that's
    # only N_REPEATS=5 fits, cheap - but is deliberately EXCLUDED from the
    # model used for E1-E8 below. Those loops refit the model 1500+ times
    # (E4: 44 features x 5 repeats x 7 folds; E8: 44 watersheds), and each
    # ensemble fit spawns 4 sub-models that each already request n_jobs=-1 -
    # a live test confirmed this combination hangs for 15+ hours on this
    # Windows machine (joblib/loky repeatedly tearing down and respawning
    # process pools - "leaked folder objects"/memmapping FileNotFoundErrors
    # in the log, not a clean finish) before eventually crashing from
    # resource exhaustion, never reaching E8. best_spatial (used below) is
    # the best NON-ensemble model instead - a documented compute/feasibility
    # tradeoff, not a claim the ensemble's diagnostics aren't worth having.
    best_ladder_model = min(
        [n for (k, n) in ladder if k == "spatial" and n not in ("mean",)],
        key=lambda n: ladder[("spatial", n)]["mae"])
    out["best_ladder_model"] = best_ladder_model  # true best by E0 MAE, may be the ensemble
    best_spatial = min(
        [n for (k, n) in ladder if k == "spatial" and n not in ("mean", "voting_ensemble")],
        key=lambda n: ladder[("spatial", n)]["mae"])
    out["best_spatial_model"] = best_spatial  # used for E1-E8 below (feasibility, see note above)

    # ---- E1: catchment-only ---------------------------------------
    splits_sp = _splits(df, "spatial")
    oof_catch = cv_predict(catchment_only_model(SEED), df, list(splits_sp), TARGET)
    out["e1_catchment_only_spatial"] = regression_metrics(df[TARGET], oof_catch)

    # ---- E2: full model minus catchment --------------------------
    from .models import NUMERIC_FEATURES_DEFAULT
    no_catch_numeric = [c for c in NUMERIC_FEATURES_DEFAULT
                        if c not in ("drainage_area_km2", "log_drainage_area", "basin_span_km")]
    m_no_catch = models_(numeric=no_catch_numeric)[best_spatial]
    oof_no_catch = cv_predict(clone(m_no_catch), df, list(splits_sp), TARGET)
    out["e2_no_catchment_spatial"] = regression_metrics(df[TARGET], oof_no_catch)

    # ---- E3: residual model -------------------------------------
    resid = df[TARGET] - oof_catch
    df_r = df.assign(_resid=resid)
    m_env = models_(numeric=no_catch_numeric)[best_spatial]
    oof_resid = cv_predict(clone(m_env), df_r, list(splits_sp), "_resid")
    oof_stack = (oof_catch + oof_resid).clip(0, 1)
    out["e3_residual_stack_spatial"] = regression_metrics(df[TARGET], oof_stack)

    full_model = models_()[best_spatial]
    oof_full = cv_predict(clone(full_model), df, list(splits_sp), TARGET)
    out["e0_full_best_spatial"] = regression_metrics(df[TARGET], oof_full)

    # ---- E4: permutation importance (spatial) -------------------
    out["e4_perm_importance"] = _perm_importance(clone(full_model), df, list(splits_sp), n_repeat=5)

    # ---- E8: leave-one-HUC8-out per-watershed breakdown ------------
    print("E8: leave-one-HUC8-out...", flush=True)
    e8 = {}
    for huc, tr, te in leave_one_huc8_out(df, "huc8_name", min_test=40):
        m = clone(full_model)
        m.fit(df.loc[tr], df.loc[tr, TARGET])
        pred = pd.Series(m.predict(df.loc[te]), index=te)
        e8[huc] = regression_metrics(df.loc[te, TARGET], pred)
    out["e8_leave_one_huc8_out"] = e8
    pd.DataFrame(e8).T.to_csv(TABLES / "e8_leave_one_huc8_out.csv")

    # ---- E5: conformal intervals -------------------------------
    # Pick a well-sized watershed (same >=40-crossing threshold as E8) so the
    # held-out test set is large enough for coverage to be a meaningful
    # number, rather than blindly the alphabetically-last of 55 watersheds
    # of wildly varying size (some have single-digit crossing counts).
    sized = df["huc8_name"].value_counts()
    hucs = sorted(sized[sized >= 40].index)
    test_huc = hucs[-1]
    te = df.index[df["huc8_name"] == test_huc]
    rest = df.index.difference(te)
    rng = np.random.default_rng(SEED)
    calib = pd.Index(rng.choice(rest, size=max(60, len(rest) // 5), replace=False))
    train = rest.difference(calib)
    iv = conformal_intervals(clone(full_model), df, train, calib, te, TARGET, alpha=0.1)
    out["e5_conformal"] = {"held_out_huc8": test_huc, "target_coverage": 0.90,
                           **evaluate_coverage(iv, df[TARGET])}
    iv.join(df[[TARGET, "huc8_name"]]).to_csv(TABLES / "e5_conformal_intervals.csv")

    # ---- E6: calibration + toward-the-mean ---------------------
    cal = calibration_table(df[TARGET], oof_full)
    cal.to_csv(TABLES / "e6_calibration.csv", index=False)
    out["e6_calibration_max_gap"] = float(cal["gap"].abs().max())
    out["e6_pred_sd_ratio_full"] = out["e0_full_best_spatial"]["pred_sd_ratio"]
    out["e6_note"] = ("pred_sd_ratio < 1 reproduces the DSL-reported pathology "
                      "(model predicts toward the mean, missing extremes).")

    # ---- E7: prioritisation reordering ------------------------
    out["e7_catchment_vs_full_priority"] = prioritization_overlap(oof_catch, oof_full, 0.2)
    out["e7_full_vs_nocatch_priority"] = prioritization_overlap(oof_full, oof_no_catch, 0.2)

    _figures(df, oof_full, oof_catch, cal, iv)
    _write_report(out)
    (RESULTS / "experiment_results.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def _perm_importance(model, df, splits, n_repeat=8):
    from .models import CATEGORICAL_FEATURES_DEFAULT, NUMERIC_FEATURES_DEFAULT
    feats = NUMERIC_FEATURES_DEFAULT + CATEGORICAL_FEATURES_DEFAULT
    rng = np.random.default_rng(SEED)
    base = cv_predict(clone(model), df, list(splits), TARGET)
    base_mae = regression_metrics(df[TARGET], base)["mae"]
    imp = {}
    for f in feats:
        if f not in df.columns:
            continue
        deltas = []
        for _ in range(n_repeat):
            d2 = df.copy()
            d2[f] = rng.permutation(d2[f].values)
            p = cv_predict(clone(model), d2, list(splits), TARGET)
            deltas.append(regression_metrics(df[TARGET], p)["mae"] - base_mae)
        imp[f] = {"mae_increase": float(np.mean(deltas)), "sd": float(np.std(deltas))}
    return dict(sorted(imp.items(), key=lambda kv: -kv[1]["mae_increase"]))


def _figures(df, oof_full, oof_catch, cal, iv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, (p, t) in zip(ax, [(oof_catch, "catchment-only"), (oof_full, "full model")]):
        a.scatter(df[TARGET], p, s=6, alpha=0.4)
        a.plot([0, 1], [0, 1], "k--", lw=1)
        a.set_xlabel("observed passability"); a.set_ylabel("predicted (spatial CV)")
        a.set_title(t); a.set_xlim(0, 1); a.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(FIG / "pred_vs_obs.png", dpi=110); plt.close(fig)

    fig, a = plt.subplots(figsize=(6, 5))
    a.plot(cal["pred_mean"], cal["obs_mean"], "o-")
    a.plot([0, 1], [0, 1], "k--", lw=1)
    a.set_xlabel("mean predicted"); a.set_ylabel("mean observed"); a.set_title("Calibration (spatial CV)")
    fig.tight_layout(); fig.savefig(FIG / "calibration.png", dpi=110); plt.close(fig)

    ivs = iv.join(df[TARGET]).sort_values("point").reset_index(drop=True)
    fig, a = plt.subplots(figsize=(9, 4))
    a.fill_between(range(len(ivs)), ivs["lower"], ivs["upper"], alpha=0.3, label="90% interval")
    a.plot(range(len(ivs)), ivs["point"], lw=1, label="prediction")
    a.scatter(range(len(ivs)), ivs[TARGET], s=8, color="k", label="observed")
    a.set_title(f"Conformal intervals, held-out HUC8"); a.legend()
    fig.tight_layout(); fig.savefig(FIG / "conformal_intervals.png", dpi=110); plt.close(fig)


def _write_report(o: dict):
    L = ["# NAACC passability - model comparison (NY statewide subset)\n"]
    L.append(f"Subset: **{o['n_crossings']} crossings** across **{o['n_huc8']} HUC8 watersheds** "
             f"(NY, capped per watershed - see `src/build_subset.py`). "
             f"Target mean {o['target_mean']:.3f}, sd {o['target_sd']:.3f}.\n")
    L.append("> Scope: a prototype-scale reproduction of the *structure* of the DSL / Plunkett "
             "GIS-prediction approach with a reduced predictor set. Not the full DSL feature "
             "engineering. See `reports/METHODOLOGY.md`.\n")
    L.append(f"DSL-reported baseline (full feature set, 13-state, OOB): **R2 = {DSL_BASELINE_R2}**, "
             "and it \"generally doesn't predict the more extreme values\".\n")

    L.append("## Hyperparameter tuning\n")
    L.append("One `RandomizedSearchCV` pass per tunable model (RF/ExtraTrees/HistGBM/XGBoost), "
             "scored by negative MAE under its own HUC8-group CV split (seeded independently of "
             "every split below). Not nested inside each individual reporting fold - see "
             "`models.py::tune_hyperparameters` for the honesty note on rigor vs. compute cost.\n")
    for name, params in o.get("tuned_hyperparameters", {}).items():
        L.append(f"- **{name}**: `{params}`")
    L.append("")

    L.append(f"## E0 - model ladder: random vs. spatial (HUC8) cross-validation, "
             f"{N_REPEATS}x repeated\n")
    L.append(f"Each `spatial`/`random` row is the mean over {N_REPEATS} independent splits "
             "(different random watershed-to-fold or row-to-fold assignments) - not a single "
             "split's number. `_std` columns show the spread across those repeats.\n")
    lad = pd.DataFrame(o["e0_model_ladder"]).T
    cols = ["n", "mae", "mae_std", "rmse", "r2", "r2_std", "spearman", "pred_sd_ratio"]
    lad = lad[[c for c in cols if c in lad.columns]]
    L.append(lad.round(3).to_markdown())
    L.append("\nThe gap between `random/*` and `spatial/*` rows is the over-optimism of ordinary "
             "cross-validation for this problem.\n")
    L.append(f"Best spatial model (by mean MAE across repeats): **{o['best_spatial_model']}**.\n")

    def row(d):
        return (f"MAE {d['mae']:.3f} | RMSE {d['rmse']:.3f} | R2 {d['r2']:.3f} | "
                f"Spearman {d['spearman']:.3f} | pred/obs sd {d['pred_sd_ratio']:.2f}")
    L.append("## E1-E3 - the catchment-area investigation (spatial CV)\n")
    L.append(f"- **catchment-area only** (RF on drainage area): {row(o['e1_catchment_only_spatial'])}")
    L.append(f"- **full best model**: {row(o['e0_full_best_spatial'])}")
    L.append(f"- **full model, catchment removed**: {row(o['e2_no_catchment_spatial'])}")
    L.append(f"- **residual stack** f_catchment + f_environment: {row(o['e3_residual_stack_spatial'])}\n")

    L.append("## E4 - permutation importance (spatial CV, MAE increase when shuffled)\n")
    pi = pd.DataFrame(o["e4_perm_importance"]).T
    L.append(pi.round(4).to_markdown())
    L.append("")

    L.append("## E5 - conformal prediction intervals\n")
    e5 = o["e5_conformal"]
    L.append(f"- held-out watershed: **{e5['held_out_huc8']}**, target coverage 90%")
    L.append(f"- empirical coverage: **{e5['empirical_coverage']:.1%}**, "
             f"mean interval width **{e5['mean_interval_width']:.2f}**\n")

    L.append("## E6 - calibration & predict-toward-the-mean\n")
    L.append(f"- max calibration gap (observed - predicted, decile bins): {o['e6_calibration_max_gap']:.3f}")
    L.append(f"- prediction / observation SD ratio (full model): **{o['e6_pred_sd_ratio_full']:.2f}** "
             f"- {'reproduces' if o['e6_pred_sd_ratio_full'] < 0.9 else 'does not reproduce'} "
             "the DSL 'toward the mean' pathology")
    L.append(f"- tail MAE (true low / high 15%): "
             f"{o['e0_full_best_spatial']['mae_low_tail']:.3f} / "
             f"{o['e0_full_best_spatial']['mae_high_tail']:.3f} vs overall "
             f"{o['e0_full_best_spatial']['mae']:.3f}\n")

    L.append("## E7 - does the environment model change which crossings are prioritised?\n")
    p1 = o["e7_catchment_vs_full_priority"]
    L.append(f"- bottom-20% ('worst crossings') list, catchment-only vs full model: "
             f"Jaccard **{p1['jaccard']:.2f}**, {p1['changed']} of {p1['k']} crossings differ, "
             f"rank correlation {p1['spearman_full']:.2f}\n")

    L.append("## E8 - leave-one-HUC8-out: per-watershed breakdown\n")
    L.append("The full model, trained on every *other* watershed, evaluated on each watershed "
             "with >=40 crossings held out entirely on its own (not averaged into a pooled fold "
             "with others) - reveals whether performance is broad-based or driven by a few easy "
             "watersheds.\n")
    e8 = pd.DataFrame(o["e8_leave_one_huc8_out"]).T[["n", "mae", "r2", "spearman"]]
    e8 = e8.sort_values("r2", ascending=False)
    L.append(e8.round(3).to_markdown())
    n_pos = int((e8["r2"] > 0).sum())
    L.append(f"\n**{n_pos} of {len(e8)}** individually-held-out watersheds have positive R2.\n")

    L.append("## Figures\n")
    for fn in ["pred_vs_obs.png", "calibration.png", "conformal_intervals.png"]:
        L.append(f"![{fn}](../results/figures/{fn})\n")

    REP.mkdir(parents=True, exist_ok=True)
    (REP / "03_model_comparison.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    import joblib

    from .models import _N_JOBS
    # threading backend, not the joblib/sklearn default (process-based loky)
    # - see models.py's _N_JOBS docstring for the live-diagnosed reason.
    with joblib.parallel_backend("threading", n_jobs=_N_JOBS):
        r = run()
    print(json.dumps({k: v for k, v in r.items() if not isinstance(v, dict)}, indent=2, default=str))
