# Checks, corrections and open questions

This file lists the mistakes that were found in this project and fixed, the checks that were repeated with separate code, and the questions that are still open.

## Mistakes that were found and fixed

1. The cross-validation seed did not change the splits. The first splitting function (`spatial_cv.huc8_group_kfold`) shuffled the watersheds and then sorted them by size, so every seed produced nearly the same folds. The claim for the New York version that five independent splits gave a spread of 0.002 therefore measured nothing. The multi-state code uses `plunkett_cv.random_group_kfold`, whose folds do depend on the seed, and a test checks that. The real spread between splits is 0.002 to 0.003, and no conclusion changed.

2. The first drainage area was attached to the wrong stream for many crossings. It came from the USGS NLDI service, which snaps a point to the nearest stream line. Its rank correlation with DSL's `d8accum` is only 0.28, and against field-measured bankfull width it is 0.34, compared with 0.67 for `d8accum` (1,840 crossings). The earlier statement that drainage area is not important is withdrawn. The StreamCat measures had been joined through the same stream link, so what was said about them is also unreliable.

3. The file `docs/naacc_scoring.pdf` was a copy of a web page and not the scoring document. The real document is `docs/naacc_scoring_systems_2016-06-16.html`. All 13 weights and tables in it were checked against `src/naacc_score.py`.

4. An earlier write-up said the five score components that come from field observation make up "just over half" of the weight. They make up 43.8%, or 50.7% when armoring and internal structures are included.

5. An earlier write-up argued that an R² near 0.27 to 0.31 is a ceiling for any model that uses only map data, because about half of the score comes from things only a visit can show. That argument was too strong and was withdrawn. The published score can be predicted with an R² of 0.35 on the New York subset, while a reconstruction from its separate components reaches only 0.22, so the map data captures rules that the component formula does not.

6. An earlier note said NAACC's rule for crossings with several structures is not public. It is public: the structure with the highest openness sets the value for the crossing. The reconstruction in `src/naacc_score.py` is 35% less accurate on those crossings. The multi-state work does not use it, because it uses the published score.

7. The gain of the full model over the random forest was first reported as +0.035. That was the mean of the bootstrap. The difference of the two test scores is 0.320 minus 0.286, which is +0.034, and the interval is unchanged.

8. The number of surveys that differ between Plunkett's file and the public NAACC layer had been counted per row, and the layer has one row per structure. Counted per survey the numbers are smaller, as listed below.

## Checks repeated with separate code

`scripts/verify_headline_numbers.py` recomputes these from the raw file and the saved predictions, and all of them pass:

- The file has 56,525 surveys with unique crossing codes and scores between 0 and 1, and none of the 39 predictors is a survey field.
- The test set is the 7 recorded basins, no HUC8 watershed is on both sides of the split, and the nearest test crossing is 319 m from a development crossing.
- The saved test predictions reproduce the R² values 0.2991, 0.3154 and 0.3202 and the MAE values 0.2046, 0.2000 and 0.1988, and their targets match the raw file.
- Refitting the random forest gives 0.2862, as stored, and the paired bootstrap gives +0.034 (0.026 to 0.045).
- Rebuilding the bridge inventory join from the raw FHWA files gives the same 68,102 structures and the same 9,248 matched crossings. The bridge and OpenStreetMap joins use coordinates only.
- The calibration slope is 1.00 and the 90% intervals cover 89.2% of the test crossings.

## Further checks on the bridge inventory and neighbouring surveys

A single 0/1 flag for "matched" raises development R² from 0.287 to 0.307. Adding length and span brings it to 0.311, and all 24 bridge fields give 0.313 (`scripts/plunkett_nbi_minimal.py`). Matching only when two structures are mutual nearest neighbours gives 0.3126, and matching one way gives 0.3124 (`scripts/plunkett_nbi_oneway.py`). Using the scores of neighbouring surveyed crossings inside a partly surveyed area raises R² from 0.340 to 0.355 under a random split, so there is little to gain from it (`scripts/plunkett_infill_mode.py`).

## Scores that differ from the public NAACC layer

For the 27,451 surveys that are in both files and have a public score, 99.6% agree to within 0.001. For 97 surveys the two scores differ by at least 0.001, for 41 by more than 0.05, and for 5 by more than 0.2. All 41 have the same observation date in both files, and Plunkett's score is lower for 32 of them. They are concentrated in surveys from 2015 (23 of the 41, although 2015 holds 16% of the shared surveys) and 2019 (11), and crossings with several culverts are twice as common among them (22% against 11%). The public layer's last-edit dates come after the survey dates, so edits by coordinators and different handling of crossings with several structures are both plausible. The cause has not been established (`scripts/plunkett_public_layer_diff.py`).

## Questions that are still open

- Crossings that have been surveyed are not the same as the crossings the model would be applied to.
- The cell size of `d8accum` is not stated in the file. We assumed 30 m, and no result depends on it.
- We do not know what `tsurveyed = 2` means (1,546 rows), or how one survey was chosen for each crossing.
- We do not know why New Jersey is poorly predicted in every setup.
- It is not clear that OpenStreetMap adds anything beyond the bridge inventory. The gain is +0.005 and its interval includes zero.
- We do not know the current accuracy or validation method of the DSL model.
- The error rate of volunteers answering a yes/no question about a perched outlet has not been measured.
