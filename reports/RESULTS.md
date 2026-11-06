# Results

Every number in this file comes from a script in `scripts/` and a file in `results/plunkett/`. The headline numbers are also recomputed from the raw data by `scripts/verify_headline_numbers.py`, which does not share code with the modelling scripts. How the tests were set up is described in [METHODS.md](METHODS.md).

## The data

The data is Ethan Plunkett's file of 56,525 NAACC surveys in 13 states (CT, DE, MA, MD, ME, NH, NJ, NY, PA, RI, VA, VT, WV). Each row is one crossing, matched to the DSL stream and road networks. The value being predicted is the NAACC aquatic passability score, called `aquatic_score` in the file. The inputs are the 39 columns listed in his `passability_predvars.txt`: drainage area (`d8accum`), stream gradient, elevation, terrain measures, development, traffic, road class and county income. None of these is a field survey measurement, and a test in `tests/` plus a check in `verify_headline_numbers.py` confirm that. State and HUC8, HUC6 and HUC4 watershed codes were added from public boundary files and are used only to group crossings when splitting the data.

Two public datasets were added as extra inputs. The first is the FHWA National Bridge Inventory 2025 (see [../docs/NBI_JOIN.md](../docs/NBI_JOIN.md)). The second is OpenStreetMap bridge and culvert tags, taken from the nearest mapped way within 25 m. Both exist for crossings nobody has surveyed. OpenStreetMap is mapped more completely in some states than in others: it marks 45% of the surveyed bridges in Pennsylvania and 85% of those in Vermont.

## How the models were tested

Before any model was built, 7 whole HUC6 river basins were set aside as a test set. They hold 8,995 crossings (15.9% of the file) in 40 HUC8 watersheds. The closest test crossing is 319 m from a crossing used for development, and the median distance is 22 km. The other 47,530 crossings were used to compare and tune models, always with folds that hold out whole watersheds, whole basins or whole states, so that a model is never scored on a stream it has already seen. Each model was scored on the set-aside basins once, and the scoring script refuses to score the same model twice.

## Scores on the set-aside basins

| Model | R² | 95% interval | MAE |
|---|---|---|---|
| Guess the training average | 0.000 | | 0.258 |
| Random forest, 39 predictors | 0.286 | 0.255 to 0.307 | 0.208 |
| Ensemble, 39 predictors | 0.299 | 0.269 to 0.321 | 0.205 |
| Ensemble plus bridge inventory | 0.315 | 0.285 to 0.338 | 0.200 |
| Ensemble plus bridge inventory and OpenStreetMap | 0.320 | 0.294 to 0.339 | 0.199 |

The intervals come from resampling HUC8 watersheds. The ensemble is an equal average of two LightGBM models and one CatBoost model (`src/plunkett_final.py`), with settings chosen on the development crossings only.

Comparing the models on the same resamples, the full model beats the random forest by 0.034 (0.026 to 0.045). The bridge inventory step adds 0.016 (0.009 to 0.024) and the OpenStreetMap step adds 0.005 (-0.002 to 0.012). With only 40 test watersheds the OpenStreetMap gain cannot be told apart from zero, although it was 0.010 on the development folds, so it is probably real and small.

By state, the full model scores 0.334 in Pennsylvania, 0.314 in Virginia, 0.282 in New York, 0.240 in Maryland and -0.011 in New Jersey (128 crossings). New Jersey was poorly predicted in every setup we tried, at about 0.1 on the development folds, and we do not know why.

On the development folds, where each fold holds out whole HUC8 watersheds and three different random splits were used, the ensemble scores 0.292 with Plunkett's predictors, 0.317 with the bridge inventory added and 0.326 with OpenStreetMap added. When whole states are held out, the full model scores 0.306. Ordinary random k-fold splits give about 0.03 more than watershed splits, which shows how much a random split flatters a model when nearby crossings land on both sides.

## What the model depends on

Drainage area is the largest single input, with 30% of the model's attributions. Without it the full model drops from 0.322 to 0.299, and by itself it scores 0.217. Most of its value is that it tells bridges from culverts: it separates bridges with an AUC of 0.865.

Most of the gain from the bridge inventory comes from the plain fact that a large structure exists at the crossing. Length and span add a little. Condition ratings, age and owner add nothing that can be measured.

Bridges average 0.91 in the NAACC score and other crossings average 0.55. Among the crossings that are not bridges, all 39 predictors together explain only about 16% of the variation.

The predictions are pulled toward the average. The spread of the predictions is 57% of the spread of the real scores. The calibration slope is 1.00 and the average prediction in each fifth of the data is within 0.02 of the observed average. Bridges are under-predicted (0.84 predicted against 0.93 observed) and other crossings are over-predicted (0.58 against 0.54), and among bridges alone the model cannot rank one above another.

A 90% prediction interval of plus or minus 0.424 around each prediction (the 90th percentile of the out-of-fold error on the development set) covers 89.2% of the set-aside crossings. The intervals are 0.73 wide on average, which means the score of a single crossing is not pinned down. The model is useful for ranking many crossings and for totals over an area, and it is not reliable for judging one site.

## Why accuracy stops where it does

The NAACC score is partly built from the drop between the culvert outlet and the water below it. To see how much each field measurement matters, each one was given to the model as a perfectly known input, one at a time, using the New York development crossings (base R² 0.342). Knowing the outlet drop raised R² to 0.82. No other single measurement raised it by more than 0.07. From map data alone, outlets with a drop over 0.25 ft (28% of surveyed crossings) are found with an AUC of 0.76. Because the score is partly computed from the outlet drop, this test shows where the missing information sits. It does not show that a model could reach 0.82.

A cheap field answer to the question "is the outlet perched, yes or no" raised R² from 0.287 to 0.711 on the set-aside New York basins (2,094 crossings), but only to about 0.47 when 20% of the answers were wrong. The 20% error rate was assumed and has not been measured. Part of that jump is arithmetic, since the score uses the outlet drop, so it describes a different product from the map-only model and should not be quoted as the map-only model's accuracy. The extension of this test to all 13 states is in [OUTLET_DROP.md](OUTLET_DROP.md).

## Ideas that did not help

These were tested on the development set unless noted.

| Idea | Result |
|---|---|
| Hyperparameter search over 36 settings | +0.003 |
| Other model types, loss functions and ensembles | within 0.005 |
| Scores of nearby surveyed crossings as inputs (other watersheds) | no gain, and worse when a whole state is held out |
| The same, inside a watershed that is partly surveyed (random holdout) | 0.340 to 0.355, the most that filling gaps could give |
| Extra terrain, stream power and constriction measures | no gain |
| A 1 m lidar elevation profile at 6,000 New York crossings | no gain |
| State as an input | +0.007, mostly from New Jersey |
| Giving less weight to over-sampled watersheds | slightly worse |
| Bridge match radius from 30 to 150 m, mutual versus one-way matching | no difference |
| Huber, Fair and L1 losses, neighbour-crossing context, keeping only the top features | between +0.001 and -0.034 |
| OpenStreetMap road and waterway context (surface, waterway kind, skew angle) in a Pennsylvania and Rhode Island pilot | 0.328 to 0.327 |
| A classifier for a yes/no barrier label instead of the regression | AUC 0.770 against 0.768 |
| EPA StreamCat (62 catchment measures) and NHDPlus attributes joined to the nearest stream line | +0.010 on development, +0.0055 on the set-aside basins (-0.003 to +0.015), not adopted |
| Scores of surveyed crossings on the same stream | +0.005 |
| Predicting extra targets and stacking, survey year, Connecticut and Vermont culvert inventories | no gain |
| More training rows (25% up to 100% of the file) | +0.008 for four times the rows |
| 54,000 extra NAACC-protocol crossings from the SARP national compilation | the model cannot predict them (R² about 0 in the same states), and adding them lowered the score on Plunkett's rows by 0.025, so they were not used |

Census block-group inputs were not tried because the Census API now needs a key.

## Scoring places nobody has surveyed

With none of Plunkett's raster inputs, and only road, income, bridge inventory, OpenStreetMap, StreamCat and NHDPlus data, which are all public, the model scores 0.278 on the set-aside basins, compared with 0.320 for the full model and 0.280 for our rebuild of the DSL method. The public catchment data replaces his drainage area almost completely, and the loss comes from his terrain inputs. A pipeline that finds candidate crossings from road and stream-line intersections has not been built.

## Comparison with the DSL model

DSL reports an R² of 0.31 for its model. That figure is an out-of-bag estimate from 2022, and the same estimate was used to choose the predictors. We rebuilt the procedure described in the DSL documentation (a random forest with stepwise predictor selection on out-of-bag R²) on Plunkett's current file. It reaches 0.301 out of bag, which is close to the published figure, but 0.272 on held-out HUC8 watersheds and 0.280 on the set-aside basins. The fair comparison on held-out basins is therefore 0.280 against 0.320, and not 0.31 against 0.32. This is our reading of the documentation and not DSL's deployed model, and we do not know its current accuracy.

## Limits

A gain of 0.034 in R² is modest. It is not a large improvement and should not be described as one.

All of the scores describe surveyed crossings. Programs and volunteers choose which crossings to survey, so crossings nobody has surveyed may differ in ways this data cannot show. This is the largest assumption in the project that has not been tested.

Only Northeast and Mid-Atlantic states were used, and all of their networks were built by DSL, so how well the model carries over to other regions is unknown. The model has only been shown to work on crossings that DSL could match to a road and stream intersection within 30 m.

Nearby basins are correlated, so some leakage between development and test data cannot be ruled out completely. The set-aside scores agree with the development scores to within 0.01 and do not get better or worse with distance from the training data. Forty test watersheds also limit how small a difference can be detected.

The set-aside basins were used again for four further questions after the main scoring: the rebuild of the DSL method (0.280), the New York one-question test (0.711), a test for bias, and two tests using only public data. None of these changed a model, but the basins are not as untouched as they were at the start.

The bridge inventory and OpenStreetMap are snapshots from today, while the surveys date from 2003 to 2026, and OpenStreetMap is more complete in some states than in others.

The model has not been field tested or deployed, and neither NAACC nor DSL uses it.
