# Results

Every number here comes from a script in `scripts/` and a file in `results/plunkett/`, and the headline numbers
are re-derived by `scripts/verify_headline_numbers.py`. Written 20 October 2026.

## Data

Plunkett's file: 56,525 NAACC surveys on the DSL stream and road networks in 13 states (CT DE MA MD ME NH NJ NY PA
RI VA VT WV). Target is `aquatic_score`; inputs are the 39 columns listed in his `passability_predvars.txt`
(catchment area `d8accum`, gradient, elevation, terrain windows, development, traffic, road class, county income).
No survey field is an input; a test and a check in the verify script enforce that. State and HUC8/6/4 were added from
public boundaries and are used only to group crossings for validation.

Added inputs: the FHWA National Bridge Inventory 2025 (see `docs/NBI_JOIN.md`) and OpenStreetMap bridge and culvert
tags (nearest way within 25 m). Both are public and exist for unsurveyed crossings. OSM mapping is more complete in
some states than others (it flags 45% of surveyed bridges in PA and 85% in VT).

## How it was validated

Before any model was fitted, 7 whole HUC6 basins (8,995 crossings, 15.9%, 40 HUC8s) were set aside as a locked test set
(`results/plunkett/locked_test_definition.json`). The closest locked crossing is 319 m from a development crossing and the
median is 22 km. Everything else (47,530 crossings) was used for comparison and tuning, with folds that hold out whole
HUC8 watersheds, whole HUC6 basins, or whole states. Each tier was scored on the locked basins once; the script
refuses a second look.

## Locked-basin results

| Model | R2 | 95% interval | MAE |
|---|---|---|---|
| Training mean | 0.000 | | 0.258 |
| Random forest, 39 predictors | 0.286 | 0.255 to 0.307 | 0.208 |
| Ensemble, 39 predictors | 0.299 | 0.269 to 0.321 | 0.205 |
| plus NBI | 0.315 | 0.285 to 0.338 | 0.200 |
| plus NBI and OSM | 0.320 | 0.294 to 0.339 | 0.199 |

Intervals resample HUC8 watersheds. The ensemble is an equal average of two LightGBM models and one CatBoost model
(`src/plunkett_final.py`), chosen on the development set only.

Differences (paired, same resamples): full model minus random forest +0.034 (0.026 to 0.045); NBI step +0.016 (0.009 to
0.024); OSM step +0.005 (-0.002 to 0.012). The OSM step is not separable from zero on 40 watersheds, although it was
+0.010 on the development folds, so treat it as probable and small.

By state on the locked basins (full model): PA 0.334, VA 0.314, NY 0.282, MD 0.240, NJ -0.011 (128 crossings). NJ was poor
in every setup we tried (about 0.1 on development folds) and we do not know why.

Development folds (HUC8, three differently seeded splits, ensemble): 0.292 with his predictors, 0.317 with NBI, 0.326 with OSM, 0.306 for the
full model when whole states are held out. Random k-fold gives about 0.03 more than HUC8 folds, which is the optimism
of random splits.

## What the model relies on

- Catchment area is the largest single input (30% of attributions). Without it the full model drops from 0.322 to 0.299.
  Alone it scores 0.217. It acts mostly as an indicator of bridge versus culvert (AUC 0.865 for bridge).
- NBI: most of the gain is the yes/no "a large structure is here" flag; length and span add a little; condition, age
  and owner add nothing measurable (`docs/NBI_JOIN.md`).
- Bridges average 0.91 and non-bridges 0.55. Among non-bridges all 39 predictors explain about 16% of the variance.
- Predictions are shrunk toward the mean (spread of predictions is 57% of the spread of observations), calibration
  slope is 1.00, and by-quintile means are within 0.02 of observed. Bridges are under-predicted (0.84 against 0.93) and
  non-bridges over-predicted (0.58 against 0.54); within bridges the model has no ranking skill.
- 90% intervals (prediction +-0.424, the 90th percentile of out-of-fold error on the development set) cover 89.2% of
  locked crossings. They are 0.73 wide on average, so a single crossing's score is not pinned down. The model is
  useful for ranking and for area totals, not for judging one site.

## Why the ceiling is where it is

Adding a perfectly known field property to the model, one at a time (New York development crossings, base R2 0.342):
outlet drop raises R2 to 0.82; every other property adds 0.07 or less. From map data alone, perched outlets (drop over
0.25 ft, 28% of surveys) are identified with AUC 0.76. Note that the NAACC score is partly built from the outlet drop, so
this is an upper-bound check on where the missing information sits, not a modelling result.

A cheap field answer, "is the outlet perched, yes or no", lifts R2 from 0.287 to 0.711 on the locked New York basins
(2,094 crossings), but falls to about 0.47 if 20% of the answers are wrong. That error rate is an assumption, not a
measurement. Because the score depends on outlet drop, part of this jump is arithmetic. It is a different product
from the map-only model and must not be quoted as its accuracy.

## Tried and not adopted

All on the development set unless noted; details in `reports/04` to `reports/06`.

| Idea | Outcome |
|---|---|
| Hyperparameter search (36 configurations) | +0.003 |
| Other model families, objectives, ensembles | within 0.005 |
| Scores of nearby surveyed crossings as inputs (other watersheds) | none; worse leaving one state out |
| Same, inside a surveyed area (random holdout, `scripts/plunkett_infill_mode.py`) | 0.340 to 0.355, an upper bound for gap filling |
| Extra terrain, stream-power and constriction features | none |
| 1 m lidar profile at 6,000 New York crossings | none |
| State as an input | +0.007, mostly NJ |
| Weighting over-sampled watersheds | slightly worse |
| NBI match radius 30 to 150 m; mutual versus one-way matching | none |

Not done: re-linking StreamCat/NHDPlus attributes to the right stream segments for every crossing (about +0.005 to
+0.014 on a New York subset, needs rate-limited APIs), and census block-group inputs (the API now needs a key).

## Comparison with the DSL model

DSL's published 0.31 is an out-of-bag estimate from 2022 that also used that estimate to choose predictors. We rebuilt
the documented procedure (random forest, stepwise selection on out-of-bag R2) on his current file. It reaches 0.301 out
of bag, close to the published figure, 0.272 under HUC8 folds and 0.280 on the locked basins. So the fair comparison on
held-out basins is 0.280 against 0.320, not 0.31 against 0.32. This is our reading of the documentation, not his deployed
model, and we do not know his current accuracy.

## Limits

1. A gain of 0.034 R2 is modest. It is not the substantial improvement asked for and we do not describe it as one.
2. All scores describe surveyed crossings. Survey effort is chosen by programmes and volunteers, so unsurveyed
   crossings may differ in ways these data cannot show. This is the largest untested assumption.
3. Only Northeast and Mid-Atlantic states, all on DSL-built networks. Transfer elsewhere is untested.
4. Spatial leakage cannot be excluded entirely, because neighbouring basins are correlated. Locked-basin scores
   agree with development scores to within 0.01 and show no trend with distance from the training data.
5. Forty locked watersheds limit what can be separated statistically.
6. The locked basins were reused for three extra questions after the main scoring: the DSL emulation (0.280), the
   New York quick-check test (0.711) and the bias audit. None of these changed a model.
7. NBI and OSM are present-day snapshots against surveys from 2003 to 2026, and OSM completeness varies by state.
8. Not adopted, not field-tested, not deployed.
