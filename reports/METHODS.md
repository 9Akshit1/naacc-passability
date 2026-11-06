# Methods

This file explains how the data was prepared, how the models were tested, and what the earlier New York version did. The results are in [RESULTS.md](RESULTS.md).

## The problem

NAACC has field surveys for tens of thousands of road-stream crossings. From each survey it computes an aquatic passability score between 0 and 1, using a published formula that combines about 13 survey measurements such as the outlet drop, the width of the opening compared with the stream, the water depth in the culvert and whether the culvert has a natural stream bed inside. Most crossings have never been surveyed. The Designing Sustainable Landscapes (DSL) project at UMass Amherst predicts the score for every crossing from map layers with a random forest and reports an out-of-bag R² of 0.31. The DSL documentation notes that the model does not predict the extreme values and tends to predict values near the average. This project builds a model of the same kind, tests it more strictly, and checks whether two public datasets improve it.

## Data for the multi-state version

The multi-state version uses Ethan Plunkett's file of 56,525 surveyed crossings in 13 states. Each crossing has the NAACC score and 39 map-based predictors computed for DSL's own stream and road networks. The 39 predictors are listed in his `passability_predvars.txt`. Two things were added to the file:

- State and the HUC8, HUC6 and HUC4 watershed codes, found by locating each crossing in Census and USGS boundary files. These are used only to decide which crossings go in which fold.
- The extra input columns described below.

No input column comes from the NAACC survey. The survey also records things like road type and structure size, and those are the kind of measurement the model is meant to do without, because they cannot be known for a crossing nobody has visited.

## Extra inputs

**FHWA National Bridge Inventory.** This is a federal list of public-road bridges and of culverts longer than 6.1 m (20 ft), with the same fields in every state. We kept structures whose item 42B (what runs under the structure) says a waterway is involved. Each crossing is matched to its nearest such structure when the two are within 60 m and each is the other's nearest neighbour. The radius was picked from the distribution of match distances and not from model scores. A crossing with no match is given `nbi_matched = 0` and missing values, which the models read as "probably not a large public-road structure". [../docs/NBI_JOIN.md](../docs/NBI_JOIN.md) describes the join in detail.

**OpenStreetMap.** For each crossing we take the nearest way tagged as a bridge or culvert within 25 m, and the tags on it. The code is in `src/osm_structures.py`.

Both inputs are public and exist for crossings that were never surveyed, so they could be used when the model is applied for real. Coordinates are used to join them and survey scores never are.

## How the data was split

Random splits give flattering results here, because crossings on the same stream or in the same watershed resemble each other and one of them ends up in training while its neighbour is in the test. All splits in this project therefore keep whole groups together.

First, 7 whole HUC6 basins were set aside as the test set before any model was built. They hold 8,995 crossings in 40 HUC8 watersheds. The definition is saved in `results/plunkett/locked_test_definition.json`. The code and the file names call this set the locked set. No watershed appears in both the test set and the development set, and the nearest test crossing is 319 m from a development crossing.

The remaining 47,530 crossings were used for development. Models were compared with 5-fold cross-validation in which every fold holds out entire HUC8 watersheds (`src/plunkett_cv.py`), using three different random assignments of watersheds to folds. Some comparisons also hold out entire HUC6 basins or entire states.

A model that is finished is scored on the set-aside basins once. `scripts/plunkett_locked_eval.py` records each score in `results/plunkett/locked_results.json` and refuses to run again for a model that is already there.

## The final model

The final model (`src/plunkett_final.py`) is an equal average of three models: a LightGBM regression with extra-trees splits, a LightGBM model with a cross-entropy objective, which keeps its output between 0 and 1, and a CatBoost model of depth 6. The settings for the two LightGBM models were chosen from a search of 36 configurations scored on the development folds with a separate random seed. Predictions are clipped to the range 0 to 1.

The baseline for comparison is a random forest on the same 39 predictors. Prediction intervals are plus or minus the 90th percentile of the absolute out-of-fold error on the development set, which is 0.424.

## Statistical tests

Intervals on the test set come from a bootstrap that resamples whole HUC8 watersheds many times, so the uncertainty reflects the small number of watersheds and not only the number of crossings. Differences between two models are computed on the same resamples, so a 95% interval for the difference that excludes zero is evidence of a real difference.

## Checks on the results

`scripts/verify_headline_numbers.py` recomputes the main results from the raw file and the saved predictions, with code that does not use the modelling code. It checks the number of crossings, that no predictor is a survey field, that the test set is the recorded 7 basins, the test scores, the random forest baseline, the paired intervals, the bridge inventory join and the interval coverage. Mistakes that were found in the work are listed in [CHECKS.md](CHECKS.md).

## The first version, for New York only

The first version used 44,871 New York surveys from the state's public NAACC layer, reduced to 28,141 unique crossings and then to a modelling subset of 21,111 crossings with at most 1,000 per watershed. It did not have DSL's raster layers, so it approximated them: terrain from USGS 3DEP elevation points around each crossing, drainage area from the USGS NLDI basin polygon, stream gradient from NHDPlus attributes, traffic from the New York transportation department, and structure age and condition from that department's bridge and large-culvert list. On held-out watersheds this version scored an R² of about 0.27.

Two parts of it were later found to be wrong. The drainage area came from the nearest stream line and was attached to the wrong stream for many crossings, and the cross-validation seeds did not change the splits. Both are described in [CHECKS.md](CHECKS.md). The code for this version is still in `src/` (`data.py`, `gis_predictors.py`, `build_subset.py`, `models.py`, `model_experiments.py`, `predict.py`) and `naacc_score.py` rebuilds the 2016 NAACC scoring formula from survey measurements. The rebuilt formula matches the published score with a correlation of 0.89 and an average error of 0.055. The New York files are described in [../DATA_DICTIONARY.md](../DATA_DICTIONARY.md).

## References

- Plunkett EB, McGarigal K, Compton BW, Jackson SD, DeLuca WV, Grand J. 2022. Designing Sustainable Landscapes: Aquatic Barriers data product. University of Massachusetts, Amherst. A copy of the documentation is in `docs/DSL_documentation_abarriers.pdf`.
- North Atlantic Aquatic Connectivity Collaborative. Aquatic Connectivity Scoring Systems for Non-tidal Crossings, revised 2016-06-16. A copy is in `docs/`.
- Mulvihill CI, Baldigo BP, Miller SJ, DeKoskie D, DuBois J. 2009. Bankfull discharge and channel characteristics of streams in New York State. USGS Scientific Investigations Report 2009-5144.
