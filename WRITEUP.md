# Predicting NAACC passability from map data

Akshit Erukulla, October 2026

## Summary

Most road-stream crossings have never been surveyed, so DSL predicts the NAACC aquatic passability score from map
layers. This project rebuilt that kind of model, first on New York surveys and then, after review by Ethan Plunkett,
on his file of 56,525 surveys from 13 northeastern and mid-Atlantic states.

On 8,995 crossings in seven river basins set aside before modelling and scored once, a random forest on his 39
predictors reaches R2 0.286. A boosted ensemble on the same predictors reaches 0.299, adding the FHWA National Bridge
Inventory 0.315, and adding OpenStreetMap bridge and culvert tags 0.320. The gain over the forest is 0.034 (95%
interval 0.026 to 0.045). That is a small improvement, and we do not claim more.

## What limits accuracy

Most of the remaining error is whether a culvert's outlet is perched. The score depends heavily on outlet drop, and no
public map layer measures it. If outlet drop were known, the model's R2 would be about 0.82 on New York crossings; no
other single field property adds more than 0.07. From map data alone, perched outlets are identified with an AUC of
0.76. Early in this project we argued that R2 near 0.3 is a ceiling for any map-only model. Later checks showed that
argument was too strong, and it is withdrawn. What the data support is that the missing information is mainly outlet
condition.

## What changed after review

- A catchment-area input from the first version was linked to the wrong stream segment for many crossings. Using the
  DSL `d8accum` instead raised the old New York model from 0.289 to 0.327. The earlier claim that catchment area does
  not matter was wrong.
- Pooling states did not hurt. Averaged over eight states, a pooled model scored 0.28, a model that never saw the state
  0.26, and a model trained on that state alone 0.23. All states tested are in the Northeast and mid-Atlantic.
- The 0.31 that DSL reports is an out-of-bag figure. Rebuilding their documented procedure and scoring it on the same
  held-out basins gives 0.280, so the like-for-like comparison is 0.280 against 0.320.

## What was checked

The headline numbers were re-derived with separate code from the raw file and the saved predictions
(`scripts/verify_headline_numbers.py`): the locked scores, the baseline refit, the paired interval, calibration, and the
NBI join. Errors found along the way, and open questions, are listed in `reports/CHECKS.md`.

## Limits

The model has not been tried on unsurveyed crossings, which may differ from surveyed ones. Predictions are shrunk toward
the mean and single-crossing intervals are wide (about 0.73), so the output suits ranking and area totals rather than
judging one site. Bridges are under-predicted, partly because coordinators in some states mark large bridges "Bridge
Adequate" (a fixed score of 1.0) far more often than in others, which map data cannot see. New Jersey is poorly
predicted and we do not know why.

## Reuse

The National Bridge Inventory join is self-contained and can be run on any list of crossings: `docs/NBI_JOIN.md`.
Full results are in `reports/RESULTS.md`.

## Reference

Plunkett EB, McGarigal K, Compton BW, Jackson SD, DeLuca WV, Grand J. 2022. Designing Sustainable Landscapes: Aquatic
Barriers data product. University of Massachusetts, Amherst.
