# 06 - Is our model worse than the published one? What drives it? What else was tried? (2026-10-09)

All numbers come from scripts in `scripts/` and files in `results/plunkett/`. Development-set figures use HUC8-group CV
(random splits); the only locked-test number added here is the emulated-DSL reference (one look).

## 1. Plunkett's own accuracy and data recency

* **His data is the newest available.** `passability_data.zip` was last modified 2026-10-08 14:21 GMT (the day before this
  work); its newest survey is 2026-10-06; 20,377 of 56,525 surveys are from 2021 onward. His site does not list a directory.
* **His only published accuracy is R2 = 0.31** (out-of-bag, DSL documentation dated 2022-09-22). His 2026-10-08 email gives
  no R2 for the pipeline re-run and I found no newer published figure. I do not know his current number.
* **Fair comparison by emulation** (`scripts/plunkett_pipeline_emulation.py`). The documented pipeline - random forest with
  additive stepwise variable selection on out-of-bag R2 - was re-created on his current data and fitted on our development
  set. It selected `d8accum, elev_range, bdev, gradient, adt, class, income, dem` (19 columns); **OOB R2 0.301** (close to
  the published 0.31, which suggests the emulation is faithful); honest HUC8-CV R2 **0.272**; **locked-basin R2 0.280**.
  Caveats: my forest sizes and settings, a reduced search budget, and I cannot run his actual deployed model.

| Locked basins (8,995 crossings, scored once each) | R2 |
|---|---|
| Emulated documented DSL pipeline (RF + stepwise OOB) | 0.280 |
| Random forest on all 39 of his predictors | 0.286 |
| Our ensemble, same 39 predictors (T1) | 0.299 |
| + national bridge inventory (T2) | 0.315 |
| + OpenStreetMap tags (T3) | 0.320 |

**So no: on identical held-out basins our model is not worse than an emulation of the published method; it is +0.040 above
it (T3).** The emulation's out-of-bag R2 (0.301) overstates its spatial skill (0.280) by about 0.02, the optimism of OOB.

## 2. How much does catchment area drive the model now? (T3 features, development set)

TreeSHAP share of total |contribution|: **catchment area `d8accum` 30.2%**, gradient + elevation 15.9%, terrain windows
13.0%, NBI structure size 12.7%, OSM tags 9.4%, traffic + road class 5.4%, other NBI fields 5.3%, county income 4.2%,
development 3.9%.

| Retrained model | R2 |
|---|---|
| all T3 features | 0.322 |
| without `d8accum` | 0.299 |
| his predictors minus `d8accum` (no NBI/OSM) | 0.173 |
| `d8accum` alone | 0.217 |
| `d8accum` + NBI + OSM only | 0.252 |
| NBI + OSM only (no DSL predictors at all) | 0.198 |
| all features but the 30 terrain windows | 0.311 |

`d8accum` is the largest single input but **not a single point of failure**: with the national bridge/culvert evidence the
model loses only 0.023 without it (versus 0.114 in his predictors-only model), because those layers carry much of the same
bridge-versus-culvert signal. It is a physically sensible predictor (bigger streams, bigger structures; checked against
field-measured bankfull width), not a leak. A model using only nationally available layers already reaches 0.198.

## 3. What limits accuracy: the oracle test

If a field property were known exactly (added to T3 as an input; NY development crossings, 14,061; base R2 0.342):

| Property known exactly | R2 gain |
|---|---|
| **outlet drop** | **+0.479 (R2 0.82)** |
| structure openness | +0.068 |
| constriction class | +0.050 |
| structure length | +0.046 |
| crossing type | +0.044 |
| road fill height | +0.040 |
| structure material | +0.035 |
| inlet/outlet height | +0.031 |
| inlet grade | +0.028 |
| all size fields | +0.091 |
| type + all size fields | +0.097 |

The outlet-drop component carries 16.1% of the score weight and the final score is capped by it. **Nearly all of the
unexplained variance is whether the outlet is perched, a property no public GIS layer measures.** From GIS alone, perched
outlets (> 0.25 ft, 28% of NY surveys) are classified at AUC 0.76. This is an upper-bound analysis, not a model.

## 4. Things tried to get a substantially better model

| Attempt | Result |
|---|---|
| NBI national bridge inventory | +0.025 dev (real) |
| OSM bridge/culvert tags | +0.010 dev, +0.005 locked (CI includes 0) |
| Clean-link StreamCat/NHDPlus attributes (NY, only crossings whose old reach link agrees with `d8accum`, 48%) | +0.007 (all linked) / **+0.014 (verified subset)**. Real but not scaled: a proper re-link of 56k crossings needs rate-limited APIs for an estimated +0.005 to +0.01 |
| Fine-scale (1 m-class lidar DEM) ground profile around each crossing (49 samples x 6,000 NY crossings; embankment, stream-axis slope, step below crossing) | **no gain**: score +0.0014 (sd 0.0037); outlet-drop R2 0.165 -> 0.175; perched AUC 0.762 -> 0.762; the "embankment" proxy correlates 0.03 with surveyed fill height, so it does not measure fill |
| State-aware tier (state as an input; for states with surveys) | +0.0068 (5/5 splits), mostly NJ 0.127 -> 0.176; labelling-practice differences are real but small |
| Neighbour-label features, aux bridge classifier, derived terrain/constriction features, weighting, tuning, model families | all <= +0.004 (report 04, section 6) |
| Census block-group income/housing age | **not run**: the Census API now requires a registered key; a keyless route (ACS geodatabases) is large for a small expected gain |

## 5. Honest conclusion

The model is better than an emulation of the published method (+0.04 R2 on held-out basins) and it is calibrated, but it is
**not substantially better in absolute terms**: R2 is about 0.32, and the oracle test shows the dominant remaining signal
(culvert outlet perching) is invisible to the layers available. The remaining realistic levers are (a) estimating
perching from imagery or survey-assisted data, (b) the clean StreamCat/NHDPlus re-link (about +0.005 to +0.01), and
(c) more labelled NAACC data. None was achievable here without data we do not have.
