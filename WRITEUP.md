# Predicting NAACC Aquatic Passability from GIS Data: A New York Case Study

Akshit Erukulla
October 2026

## Introduction

This report describes an attempt to reproduce, on New York data, the
structure of the GIS-only passability prediction model developed by the
Designing Sustainable Landscapes (DSL) project (Plunkett et al. 2022). The
DSL model predicts the NAACC aquatic passability score for unsurveyed
road-stream crossings using only GIS-derived predictors, since the majority
of crossings in the landscape have never been field-surveyed. Plunkett et
al. (2022) report an out-of-bag R² of 0.31 for this model and note two
specific limitations: a low R², and a tendency to predict values closer to
the mean rather than the more extreme values found in the field data.

The purpose of this work was twofold: to reproduce the DSL approach with
full methodological rigor, particularly with respect to spatial validation,
and to examine whether the limitations documented by Plunkett et al. (2022)
reflect a modeling shortfall or a more fundamental property of the
prediction problem itself.

## Data and Methods

The analysis uses 44,871 NAACC survey records from New York State, filtered
to 28,141 unique surveyed crossings. The NAACC 2016 scoring formula was
independently re-implemented from its published specification and validated
against the published scores (r = 0.89) before being used as the target
variable. The modeling subset comprises 21,111 crossings across all 55 of
New York's HUC8 watersheds, capped at 1,000 crossings per watershed so that
no single large watershed dominates the sample.

A central methodological choice in this work is the use of spatial
cross-validation, in which entire HUC8 watersheds are held out rather than
individual rows, alongside conventional random k-fold cross-validation. The
difference between the two validation schemes is substantial and persists
throughout the analysis, consistent with the caution in Plunkett et al.
(2022) that out-of-bag estimates from a random forest may not fully reflect
performance on genuinely unsurveyed regions.

## Results

The best-performing model, an equal-weight ensemble of random forest, extra
trees, histogram gradient boosting, and XGBoost, achieves R² = 0.273 under
spatial cross-validation (MAE = 0.210) and R² = 0.312 under random k-fold
cross-validation.

Plunkett et al. (2022) report an out-of-bag R² of 0.31. Out-of-bag
estimation draws held-out rows from bootstrap resampling of the entire
training set rather than from held-out geographic regions, so spatially
correlated crossings routinely appear together in a tree's in-bag and
out-of-bag partitions. Methodologically, this places the out-of-bag estimate
closer to the random cross-validation result obtained here (0.312) than to
the spatial cross-validation result (0.273). Under the comparison that the
spatial structure of the data would suggest is appropriate, the model
described here is not distinguishable in accuracy from the DSL figure.
Under the harder, held-out-region standard used throughout this analysis,
it falls somewhat below that figure, on a reduced predictor set and a
single state rather than thirteen.

Two further findings are reported in full in the accompanying methodology
document. First, the predictors with the largest effect on accuracy were
not landscape-context variables but structure-level data: bridge and
culvert age, condition, and material, obtained from the New York State
Department of Transportation's structure inventory. This is consistent with
the NAACC scoring formula itself being computed from field-measured
structure properties rather than surrounding landscape characteristics.

Second, and bearing directly on the limitation described in Plunkett et al.
(2022), the passability score was decomposed into its thirteen published,
weighted sub-components, and each component's predictability from GIS data
alone was tested independently. The components tied to structure geometry
(constriction, opening height, outlet drop) proved genuinely learnable,
each reaching an R² comparable to or exceeding that of the composite score.
Components that require a direct field observation at the time of survey
(presence of debris or barriers, scour pool condition, substrate match,
water depth, and water velocity) could not be predicted above chance
regardless of which predictors were supplied, and together these account
for just over half of the score's total weight. This result quantifies,
rather than simply restates, the observation in Plunkett et al. (2022) that
the model does not predict extreme values well: approximately half of what
the NAACC score measures is, by construction, unobservable without a site
visit. This suggests that an R² in the range of 0.27 to 0.31 may be close
to the practical ceiling for any GIS-only approach to this problem, rather
than a deficiency correctable by additional features.

## Conclusion

This work is not presented as a general improvement on the DSL model. It is
restricted to one state, uses a smaller and partially different predictor
set, and has not been evaluated for operational use. Within those limits,
it reproduces the DSL approach under a validation standard at least as
rigorous as the original, and it offers one result that may be of
independent interest beyond this particular model: a quantified account of
why GIS-only prediction of aquatic passability appears to have a ceiling
close to where both this analysis and Plunkett et al. (2022) have found it,
and why that ceiling follows from a property of the target variable rather
than from any particular model.

## Code and Reproducibility

The accompanying repository contains the full pipeline, from the raw NAACC
download through model training and inference, together with a detailed
methodology document (`reports/METHODOLOGY.md`). All reported figures are
reproducible from the included data and code. Comments on any part of this
analysis, particularly on any point at which the comparison to the DSL
model may not hold, would be welcome.

## Reference

Plunkett EB, McGarigal K, Compton BW, Jackson SD, DeLuca WV, Grand J. 2022.
Designing Sustainable Landscapes: Aquatic Barriers data product. University
of Massachusetts, Amherst.
