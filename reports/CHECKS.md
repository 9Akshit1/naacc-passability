# Checks, corrections and open questions

A running record of what was wrong, what was re-verified, and what is still uncertain. Dated 20 October 2026. The
earlier audit (`05_audit_2026-10-08.md`) has the working for the first items.

## Errors found and fixed

1. Cross-validation seeds did not change the split. The older `spatial_cv.huc8_group_kfold` shuffled groups and then
   sorted them by size, so every seed gave nearly the same folds. The "five independent splits, sd 0.002" quoted for the
   New York work measured nothing. The multi-state code uses `plunkett_cv.random_group_kfold`, which depends on the
   seed (tested). The real split-to-split spread is 0.002 to 0.003. Conclusions did not change.
2. The first catchment area (NLDI) was linked to the wrong stream reach for many crossings. Its rank correlation with
   DSL's `d8accum` is 0.28, and against field-measured bankfull width it is 0.34 against 0.67 for `d8accum` (n = 1,840).
   The earlier statement that catchment area is unimportant is withdrawn. The StreamCat metrics were joined through the
   same link, so what was said about them is unreliable.
3. The file `docs/naacc_scoring.pdf` was a web page, not the scoring document. The real document is
   `docs/naacc_scoring_systems_2016-06-16.html`; all 13 weights and tables were checked against `src/naacc_score.py`.
4. The old write-up called the five field-observed score components "just over half" of the weight. They are 43.8%
   (50.7% with armoring and internal structures).
5. The old write-up argued that R2 near 0.27 to 0.31 is a ceiling for GIS-only models because half of the score is
   unobservable. That argument was too strong. The published score is predictable at 0.35 on the New York subset,
   while a reconstruction from components reaches only 0.22, so GIS captures rules that the component formula does not.
   The write-up was replaced.
6. The scoring reconstruction said NAACC's rule for crossings with several structures is not public. It is: the
   structure with the highest openness sets the crossing's value. The reconstruction is 35% less accurate on those
   crossings. It is not used in the multi-state work, which takes NAACC's published score.
7. The headline gain was written as +0.035 in earlier messages. That is the mean of the bootstrap; the difference of the
   two locked scores is 0.320 minus 0.286 = +0.034. The interval is unchanged.
8. The count of surveys that differ between Plunkett's file and the public layer was taken per row, and the layer has
   one row per structure. Counted per survey the numbers are smaller (see below).

## Re-verified on 20 October 2026 with separate code (`scripts/verify_headline_numbers.py`)

- 56,525 surveys, unique crossing codes, scores between 0 and 1, and none of the 39 predictors is a survey field.
- The locked set is the 7 recorded basins, no HUC8 lies on both sides, and the closest locked crossing to development
  data is 319 m away.
- The saved locked predictions reproduce R2 0.2991, 0.3154, 0.3202 and MAE 0.2046, 0.2000, 0.1988, and their targets
  match the raw file.
- Refitting the random-forest baseline gives 0.2862, as stored. The paired bootstrap gives +0.034 (0.026 to 0.045).
- The NBI join rebuilt from the raw FHWA files gives the same 68,102 structures and the same 9,248 matches. The NBI
  and OSM joins use coordinates only, never a score.
- Calibration slope is 1.00 and the 90% interval covers 89.2%.

## New checks this round

- Smallest useful NBI join: a matched flag takes development R2 from 0.287 to 0.307; length and span bring it to
  0.311; all 24 fields give 0.313 (`scripts/plunkett_nbi_minimal.py`).
- Mutual-nearest against one-way matching: 0.3126 against 0.3124 (`scripts/plunkett_nbi_oneway.py`).
- Neighbouring survey scores inside a surveyed area: 0.340 to 0.355 under random holdout, so little to gain
  (`scripts/plunkett_infill_mode.py`).

## Scores that differ from the public NAACC layer

For the 27,451 surveys in both files that have a public score, 99.6% agree to within 0.001. 97 differ by at least
0.001, 41 by more than 0.05 and 5 by more than 0.2. All 41 have the same observation date in both files. Plunkett's
score is lower in 32 of them. They are concentrated in 2015 (23 of 41, against 16% of all shared surveys) and 2019 (11),
not in recent surveys, and multiple-culvert crossings are twice as common among them (22% against 11%). The public
layer's last-edit dates fall after the survey, so coordinator edits and different handling of crossings with several
structures are both plausible. The cause is not established (`scripts/plunkett_public_layer_diff.py`).

## Still uncertain

- Surveyed crossings are not the population the model would be applied to.
- The cell size of `d8accum` (30 m assumed) is not documented in the file; no result depends on it.
- The meaning of `tsurveyed = 2` (1,546 rows) and how one survey per crossing was chosen are undocumented.
- Why NJ is poorly predicted in every setup.
- Whether OSM adds anything beyond NBI (+0.005, interval includes zero).
- Plunkett's current accuracy and validation method are unknown to us.
- The volunteer error rate for a yes/no perched-outlet check is unmeasured.
