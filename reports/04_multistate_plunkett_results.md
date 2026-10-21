> A shorter, current summary is in `reports/RESULTS.md`. The full-difference point estimate is +0.034 (0.320 minus 0.286); +0.035 below is the bootstrap mean.

# 04 - Multi-state revision using the UMass DSL (Plunkett) dataset

*Written 2026-10-08, in response to Ethan Plunkett's email of the same day. Every number below is
produced by a script in `scripts/` or `src/` and stored under `results/plunkett/`.*

> **Audit 2026-10-08:** after this report was first written every calculation was re-checked (`reports/05_audit_2026-10-08.md`).
> Corrections: the CV "seeds" produced near-identical splits (conclusions unchanged; honest split-to-split SD is ~0.002-0.003),
> the catchment-area wording was tightened, and several weight-share statements in the WRITEUP/METHODOLOGY were fixed.
> Locked-test numbers are unaffected.

> **Update 2026-10-09:** an emulation of the documented DSL pipeline (RF + stepwise OOB selection) scores 0.280 on the same
> locked basins (our T3 model 0.320). What drives the model, the oracle analysis and further attempts are in
> `reports/06_improvement_attempts_2026-10-09.md`.

## TL;DR

| Question | Answer (evidence) |
|---|---|
| Does the model improve with Plunkett's data and network-consistent predictors? | **Yes, modestly.** On 8,995 locked-test crossings (7 whole basins, never used in development) a Plunkett-style random forest on his 39 predictors scores **R² 0.286**; the new ensemble on the same predictors **0.299**; adding the national bridge inventory (NBI) **0.315**; adding OpenStreetMap bridge/culvert tags **0.320**. Full model vs the RF baseline: **+0.035 (95% CI +0.027 to +0.045**, cluster bootstrap by HUC8). |
| Is that the "substantial improvement" Ethan asked for? | **No.** It is a real but small gain (about +12% relative). We do not claim more. |
| Was catchment area really unimportant? | **No - our earlier catchment area was a poor proxy.** Our NLDI-derived area correlated only rho = 0.28 with his `d8accum` on 16,155 shared crossings; swapping it into the old NY pipeline raises R² 0.289 -> 0.327. An independent referee agrees: field-measured bankfull width tracks `d8accum` at rho = 0.67 but the old area at only 0.34 (n = 1,840). `d8accum` is the single most important predictor (+0.158 R² when shuffled). |
| Is `d8accum` a proxy for "is it a bridge?" | **Largely, yes** - alone it predicts bridge/non-bridge with AUC 0.865, and alone it scores R² 0.217. Within bridges or within culverts it explains little (R² 0.055 / 0.049). |
| Does pooling states hurt (Ethan's past experience)? | **Not here.** Per-state R² averaged over 8 states: pooled 0.281 > never-saw-the-state 0.263 > trained-on-that-state-only 0.231. Pooling beat state-only in all 8 states. Caveat: all 13 states are Northeast/Mid-Atlantic. |
| Is the NYSDOT-style state bias a concern now? | The new structure feature comes from the **FHWA National Bridge Inventory** - one federal dataset with identical coding in every state - so no state-specific source is used. |

## 1. Data

* **Plunkett file** (`NAACC/FTP/web/Oct/passability_data.parquet`, from his zip): 56,525 NAACC surveys in 13
  states (CT DE MA MD ME NH NJ NY PA RI VA VT WV), on the DSL stream/road networks; target `aquatic_score`;
  the 39 predictors in his `passability_predvars.txt`. No duplicate crossing codes. Verified against our
  independent NY pipeline: on the 21,440 shared NY crossings scores agree (r = 0.995; 98.4% within 0.01).
* **Added geography (used only for validation grouping, never as inputs):** state (Census polygons), HUC8/6/4
  (USGS WBD HUC8 polygons, point-in-polygon).
* **Not used as predictors (survey fields):** `bridge`, `crossing_type`, `terrestrial_passage_score`,
  `tsurveyed`, `roadtile`, `moved`, `date_observed`, `database`, coordinates. A test enforces this.
* **FHWA NBI 2025** (13 state files; waterway-service structures only, 68,102): mutual-nearest match to a
  crossing within 60 m, label-free (16.4% of crossings matched; the radius barely matters: R² 0.312/0.313/0.313/0.313
  at 30/60/100/150 m).
* **OSM** bridge/culvert ways, all 13 states (12 via the Overpass API, PA via the Geofabrik extract because Overpass kept timing out; the PBF parser was checked against Overpass on Rhode Island: 100% per-crossing agreement for bridges, 99.8% for culverts). Matched to the nearest way within 25 m.

## 2. Validation protocol (fixed before any model was fit on this dataset)

1. **Locked test set**: whole HUC6 basins (seeded, >=4 states) totalling 8,995 crossings (15.9%): PA 3,739,
   NY 2,368, VA 1,444, MD 1,245, NJ 128, DE 40, WV 31; 40 HUC8s. Definition: `results/plunkett/locked_test_definition.json`.
   Scored **once per tier** (`scripts/plunkett_locked_eval.py` refuses a second look).
2. **Development set** (47,530): all comparisons, tuning and model choice. Schemes: random k-fold (reference only),
   HUC8-group, HUC6-group, leave-one-state-out. 3 differently seeded group assignments (2 for HUC6).
3. Hyper-parameters tuned on the development set with HUC8 folds at **seed 100**, disjoint from reporting seeds.

## 3. Development-set results (R², pooled out-of-fold)

| Tier / model | random | HUC8 | HUC6 | leave-one-state-out |
|---|---|---|---|---|
| T1 Plunkett predictors, LightGBM default | 0.318 | 0.289 | 0.282 | 0.265 |
| T1 tuned LightGBM | 0.304 | 0.292 | 0.287 | 0.275 |
| T1 ensemble (final recipe) | - | 0.293 | 0.288 | 0.272 |
| T2 +NBI, LightGBM default | 0.340 | 0.314 | 0.307 | 0.289 |
| T2 tuned LightGBM | 0.329 | 0.317 | 0.313 | 0.301 |
| T2 CatBoost | 0.339 | 0.313 | 0.308 | 0.292 |
| T2 ensemble | - | 0.318 | 0.313 | 0.297 |
| T3 +NBI +OSM, LightGBM default | 0.347 | 0.323 | 0.317 | 0.299 |
| T3 tuned LightGBM | 0.338 | 0.327 | 0.322 | 0.310 |
| T3 cross-entropy LightGBM | 0.343 | 0.326 | 0.319 | 0.300 |
| T3 CatBoost | 0.347 | 0.323 | 0.317 | 0.301 |
| **T3 ensemble (final recipe)** | - | **0.328** | **0.322** | **0.306** |

* Random CV overstates spatial performance by about 0.03 (T1: 0.318 vs 0.289) - the optimism a
  random split / out-of-bag estimate builds in.
* Model class is not the bottleneck: LightGBM variants, CatBoost and the ensemble all land within 0.005.
  36-config tuning moved R² by +0.003; the best settings are *more regularised* and score lower under random CV.
* Plunkett-style random forest on his predictors (200 trees on dev, 300 on the locked fit; mtry 0.33, min leaf 3): 0.279 (HUC8 dev).

## 4. Locked-test results (one look each)

| Model | R² | 95% CI (HUC8-cluster bootstrap) | MAE | Spearman |
|---|---|---|---|---|
| mean of development data | 0.000 | - | 0.258 | - |
| T1 Plunkett-style random forest | 0.286 | 0.255-0.307 | 0.208 | 0.558 |
| T1 final ensemble, his predictors | 0.299 | 0.269-0.321 | 0.205 | 0.573 |
| T2 final ensemble, + NBI | 0.315 | 0.285-0.338 | 0.200 | 0.588 |
| **T3 final ensemble, + NBI + OSM** | **0.320** | 0.294-0.339 | 0.199 | 0.595 |

Paired differences (`results/plunkett/locked_paired_deltas.json`): T1 ens - RF = +0.013 (0.005 to 0.022);
T2 - T1 = +0.016 (0.009 to 0.024); T2 - RF = +0.029 (0.024 to 0.037); **T3 - RF = +0.035 (0.027 to 0.045)**;
**T3 - T2 = +0.005 (-0.002 to +0.012; P(>0) = 0.93)**. So the NBI gain is clearly real on the locked basins; the
extra OSM gain (+0.010 on dev) is smaller there and **not statistically separable from zero** with only 40 locked
watersheds. Dev (T2 0.318, T3 0.328) and locked (0.315, 0.320) agree to within 0.01 - no sign of overfitting to dev.

* Per state on the locked basins (T3 / T2): PA 0.334 / 0.339, VA 0.314 / 0.294, NY 0.282 / 0.279, MD 0.240 / 0.213, NJ -0.011 / -0.115 (n=128 only).
* Calibration slope of observed-on-predicted = 1.00; predictions are still shrunk toward the mean
  (SD(pred)/SD(obs) = 0.57) - the limitation DSL also notes.
* 90% intervals (T3: +-0.424 from the 90th percentile of dev HUC8-out-of-fold |residual|) cover **89.2%** of locked
  crossings (target 90%; the earlier NY-only model covered 86.1%). They are **wide**: a single crossing's score
  is not pinned down; the model is suited to ranking and area-level aggregation, not site-level judgement.

## 5. Catchment area: Ethan's question, answered

Same crossings (14,061 development-set NY crossings present in both datasets), same old feature set and
HistGradientBoosting pipeline, same HUC8 folds; only the catchment columns differ
(`scripts/plunkett_old_pipeline_d8accum.py`):

| Old NY pipeline with ... | R² (HUC8) |
|---|---|
| its old NLDI drainage area | 0.289 |
| **no** catchment area at all | 0.290 |
| `d8accum` area instead | **0.327** |
| both | 0.329 |

Removing the old area altogether changed nothing (0.290), **but that is not proof it was useless**: the old feature
set still contained another size proxy (`basin_span_km`). The reliable evidence is independent of the model:

| Spearman with field-measured **bankfull width** (n = 1,840 crossings; 683 high-confidence) | `d8accum` | old NLDI area |
|---|---|---|
| all | **0.667** | 0.343 |
| high-confidence widths only | **0.669** | 0.398 |
| width 5-15 ft only | **0.315** | 0.108 |
| width > 15 ft only | 0.528 | 0.480 |

(Bankfull width comes from the public NAACC layer, `scripts/download_naacc_public_layer.py`; it is a field
measurement independent of both area estimates and of the score.) A second, weaker test using NBI structure length
(only larger streams, n = 2,721) is close: 0.63 vs 0.60 - the old area was acceptable on large streams and poor on the
small ones where most culverts are. Univariate Spearman with the score: old area 0.19, `d8accum` 0.52. So the earlier
"catchment area isn't important" was an artefact of a poor area estimate, as Ethan suspected, not a property of
catchment area.

## 5b. Why the model works the way it does

Grouped permutation importance (dev set, held-out HUC8 folds, drop in R²):

| Feature group | R² drop |
|---|---|
| catchment area `d8accum` (1 feature) | 0.158 |
| NBI structure **size** (length, span, widths, clearances; 8) | 0.059 |
| gradient + elevation (2) | 0.054 |
| 30 terrain-window features | 0.036 |
| traffic + road class | 0.009 |
| development (`bdev`) | 0.009 |
| county income | 0.007 |
| NBI owner/function class/ADT/lanes | 0.003 |
| NBI condition/scour/waterway-adequacy | 0.002 |
| NBI matched flag + kind/type | 0.001 |
| NBI year built / reconstructed | 0.001 |
| NBI skew | 0.0005 |

Only the **size** of the structure carries NBI's information; age and condition (which looked important in
the earlier NYSDOT work) add nothing here. Predictor groups ablated one at a time (LightGBM, HUC8): dropping
catchment area costs 0.114 (0.289 -> 0.175); dropping any other single group costs <= 0.011.

Where the signal is: bridges average 0.91, non-bridges 0.55. Most of the model's skill is telling bridge-like
from culvert-like crossings. **Among non-bridges all 39 predictors explain only 15.5% of variance**
(among bridges 6.0%). Culvert passability depends on field-measured outlet drops, substrate, etc.

## 6. Negative / null results (so they are not re-tried)

| Idea | Result (dev, HUC8 unless noted) |
|---|---|
| Neighbour-label features (scores of nearby surveyed crossings in *other* watersheds, computed from the training fold only) | 0.3124 -> 0.3133 (no gain); leave-one-state-out **worse** 0.289 -> 0.279 |
| Auxiliary supervision: out-of-fold P(bridge) from the survey `bridge` flag stacked in | 0.3124 -> 0.3131 (no gain) |
| Hyper-parameter search (36 configs) | +0.003 |
| Model class (CatBoost, objective choices, ensembling) | within 0.005 |
| NBI match radius 30-150 m | no change |
| log catchment area + stream power (T3 base 0.3227) | 0.3223 HUC8 / 0.3003 LOSO (base 0.3008) - no change |
| 10 terrain-scale features (window ratios, trends, argmax) | 0.3226 / 0.3016 - no change |
| Constriction proxy (log structure length/span/deck width minus 0.4 log catchment area) | 0.3224 / 0.2989 - no change |
| Scale + constriction together | 0.3220 / 0.3016 - no change |
| Down-weighting over-sampled watersheds (1/sqrt(n_HUC8)) | 0.3208 / 0.2952 - slightly **worse** |
| *Observation, not an idea:* NJ is hard in every regime | R² 0.08-0.12 on dev, -0.115 on the 128 locked NJ crossings; cause not established |

## 7. OSM bridge/culvert tags (T3)

OSM `highway`+`bridge`, `highway`+`tunnel=culvert` and `waterway`+`tunnel=culvert` ways, nearest within 25 m.
All 13 states are now included. The share of crossings flagged as a road bridge is 5-43% by state and is **not**
uniformly complete. Overall it flags 55% of surveyed bridges (precision 79%: 79% of OSM-flagged crossings are
surveyed bridges), but the share varies by state: PA 45%, NJ 47%, ME 47%, NY 56%, VA 60%, MA 63%, CT 65%, MD 73%,
VT 85% (small states omitted). Mapping completeness is therefore a state-level confounder. It still helped when
whole states were held out.

| Dev set, HUC8 / HUC6 / leave-one-state-out | T2 | T3 | gain |
|---|---|---|---|
| default LightGBM | 0.3135 / 0.3068 / 0.2887 | 0.3225 / 0.3169 / 0.2991 | +0.009 / +0.010 / +0.010 |
| final ensemble | 0.3183 / 0.3130 / 0.2965 | 0.3278 / 0.3221 / 0.3063 | +0.010 / +0.009 / +0.010 |

(An early 11-state development check, before NY/PA were available, gave +0.013 over NBI.) On the locked basins the
gain is +0.005 (CI -0.002 to +0.012), so I would describe OSM as a *probable* small improvement, not an established one.

## 8. Limitations - read before quoting any number

1. **Modest gain.** +0.029 R² over a Plunkett-style RF on the locked basins. Ethan's bar ("a more substantial bump")
   is not met. Within non-bridges the model explains ~16% of variance; roughly half of the NAACC score weight
   comes from field-observed items not available to any GIS model (see WRITEUP.md).
2. **DSL's quoted R² 0.31 is not comparable**: out-of-bag, with predictors chosen by stepwise selection on that
   same OOB score, and trained on a different vintage. Under the same spatial protocol his *model type* scores
   0.279 (dev) / 0.286 (locked).
3. **Northeast / Mid-Atlantic only.** "Pooling states helps" is shown for these 13 states, whose
   networks are all DSL-derived. Transfer to other regions is untested; `d8accum`, the terrain windows and
   `bdev` exist only where DSL built the network. NBI and OSM are national.
4. **Spatial leakage can't be fully excluded.** HUC6 and leave-one-state-out give lower scores than HUC8
   (0.313/0.297 vs 0.318) - the ordering one would expect - and the locked basins agree with dev, but adjacent
   basins remain correlated.
5. **Older NY-only pipeline defect.** Its drainage area (NLDI) was frequently attached to the wrong stream
   reach (on the 16,155 shared crossings the old area is a median 2.1x the `d8accum` area, and more than 19x for the top quarter; the implied cell size is only ~30 m for the best 5%). Its **StreamCat
   metrics are joined through the same COMID**, so their "noise-level importance" findings should not be read
   as evidence about those variables. NYSDOT/terrain/AADT joins were by coordinates and are not affected.
6. **NBI** covers only public-road structures longer than 6.1 m (16.4% of crossings matched); "no match" is
   itself informative and handled as missing. **OSM completeness differs by state** (see section 7), so the OSM
   tier carries a state-level bias risk that the leave-one-state-out gain (+0.010) only partly addresses.
7. **Not adopted, not field-validated, not deployed.** Prototype/research stage.

## 9. Reproduce

```
python scripts/download_multistate_inputs.py        # needs NAACC/FTP/web/Oct/passability_data.parquet
python -m src.plunkett_data                         # tags state + HUC8/6/4
python - <<'EOF'
from src import plunkett_features as pf; pf.build().to_parquet("data/processed/plunkett_features_nbi.parquet")
EOF
python -m scripts.plunkett_ablation                 # catchment/feature-group ablations
python -m scripts.plunkett_bridge_proxy
python -m scripts.plunkett_state_transfer
python -m scripts.plunkett_tune 36
python -m scripts.plunkett_dev_comparison
python -m scripts.plunkett_group_importance
python -m scripts.plunkett_old_pipeline_d8accum
python -m scripts.build_osm_features                # needs OSM for all 13 states (PA: Geofabrik pbf in data/raw/osm/pbf)
python -m scripts.plunkett_feature_ideas            # rejected derived features / weighting
python -m scripts.validate_osm_pbf                  # PBF parser vs Overpass (RI)
python -m scripts.plunkett_locked_eval T1 T2 T3     # one look per tier
python -m scripts.plunkett_locked_deltas            # paired bootstrap across tiers
python -m pytest tests/test_plunkett.py
```
