# Methodology & scope

## The problem

NAACC has ~78,000 field-surveyed road-stream crossings across 13 states, each
with a 0-1 aquatic passability score computed from the survey. Most crossings
in the landscape have never been surveyed. The UMass Designing Sustainable
Landscapes (DSL) project (Plunkett et al. 2022) built a Random Forest that
predicts the passability score for unsurveyed crossings from GIS-derived
predictors only (drainage area, stream gradient, terrain, traffic, road
class, county income). DSL reports an OOB R² of 0.31 and notes the model
"generally doesn't predict the more extreme values — it tends to predict
values closer to the mean."

This project reproduces the structure of that approach on New York data,
with an honest account of what could and couldn't be reproduced here.

## What was reproduced

| Phase | Deliverable | Status |
|---|---|---|
| Data | 44,871 NY NAACC surveys, public source, documented provenance | done |
| 1 | Dataset audit: score distribution, missingness, HUC8/county/year breakdown, repeat-survey analysis (`reports/01_dataset_audit.md`) | done |
| 2 | Re-implementation of the NAACC 2016 scoring algorithm, validated against the published score (r = 0.89, MAE 0.055) (`reports/02_score_reconstruction.md`, `src/naacc_score.py`) | done |
| 3 | Spatial cross-validation: HUC8 group hold-out vs. random k-fold (`src/spatial_cv.py`) | done |
| 4 | Model ladder: mean, linear, ridge, elastic net, random forest, extra trees, HistGBM, XGBoost, voting ensemble (`src/models.py`) | done |
| 5 | Catchment-area investigation: catchment-only model, catchment-removed model, residual stack (`src/model_experiments.py`) | done |
| 6 | Metrics: MAE/RMSE/R²/Spearman, calibration, predict-toward-the-mean diagnostic | done |
| 7 | Split-conformal prediction intervals, group-aware calibration (`src/uncertainty.py`) | done |
| 8 | Inference pipeline: crossings CSV → predicted passability + interval (`src/predict.py`) | done |

## What wasn't reproduced, and what replaced it

DSL's full predictor set needs regional raster geoprocessing (GDAL/rasterio,
multi-hour DEM processing) that's out of scope here. Each gap below was
filled with a real, verified alternative rather than left out silently.

- **Multi-scale terrain** (DSL's windowed elevation-range/incisement
  features) — approximated with USGS 3DEP bulk multi-point elevation
  sampling: a ring of points at 100m/500m radii around each crossing,
  slope from central-difference gradient, roughness from the ring's
  elevation SD (`gis_predictors.py::fetch_terrain_features_bulk`). Not a
  reproduction of DSL's own method, but a real local-terrain signal —
  ranks #2/#3 by permutation importance.
- **D8 flow accumulation** — approximated by the NLDI upstream-basin
  polygon's area. A genuine drainage area, derived from NHDPlus
  catchments rather than a re-run D8 grid.
- **Stream gradient** — NHDPlusV2 Value Added Attributes `slope` and
  `streamorde`, matched 100% by COMID (`fetch_nhdplus_vaa_batch`). Now the
  single highest-importance predictor in the model.
- **Traffic (ADT)** — NYSDOT's public statewide AADT layer, joined by
  nearest road segment within 200m (`fetch_aadt_bulk`). 57.9% match rate
  (NYSDOT's count program covers state/county roads, not every local
  road). Included for completeness — permutation importance has stayed
  near-zero/negative in every run since it was added.
- **Development intensity (bdev)** — county population density, plus 14
  EPA StreamCat watershed metrics (climate, land cover, soils,
  infrastructure density) not in DSL's original set. 13 of the 14 show no
  detectable spatial-CV signal; listed as "added," not "helped."
- **Road class** — originally the survey's own field-observed value,
  which requires a site visit and would leak into training. Replaced with
  the nearest OpenStreetMap `highway`/`railway`-tagged way within 75m
  (`fetch_osm_road_class`). The survey value is retained as
  `road_class_surveyed` for audit only, never used as a model input.
- **Structure-level data** (age, condition, material, geometry) — not in
  DSL's predictor set at all. Added because NAACC's own scoring formula
  runs on field-measured structure properties, so landscape predictors can
  only ever proxy for why a structure was built a certain way. NYSDOT's
  public bridge + large-culvert inventory supplies real `YearBuilt`,
  condition rating, material, and geometry, joined by nearest point within
  100m (`fetch_structure_inventory_bulk`). Match rate: 21.3% — this is the
  state-maintained inventory, not a record of every small local culvert.
  Despite the limited coverage, structure fields occupy 4 of the top 10
  spots by permutation importance.
- **Regional heterogeneity** (EPA Level III ecoregion) — not a DSL
  predictor. Added because spatial CV consistently scores much worse than
  random CV and the catchment-only model stays negative throughout, both
  consistent with the landscape→passability relationship varying by
  region. A real but modest contributor (rank #13).

## Study region & sample

The modelling subset caps each HUC8 watershed at 1,000 crossings, covering
all 55 of NY's watersheds — 21,111 of the full 28,141-crossing dataset.
Uncapped, Middle Hudson alone would be ~18.5% of the dataset; the cap keeps
any single watershed under ~4.7% while letting larger watersheds still
contribute more than small ones. Coastal/tidal/Great-Lakes/Hudson-mainstem
watersheds are included (an early version excluded them over inflated NLDI
drainage-area snaps; those are now backfilled from StreamCat's independent
drainage-area estimate instead of excluded). This is a prototype-scale
demonstration, not a state-wide model — see `reports/03_model_comparison.md`
for the exact N behind any reported number.

## Validation discipline

- Every reported metric comes from an explicitly named split (`random` or
  `spatial`/HUC8-group). The spatial numbers are the headline; random-split
  numbers are shown only to quantify how optimistic ordinary CV is here.
- Predictor choices were not tuned against the spatial CV score.
- Crossings with neither drainage area nor elevation are dropped, not
  imputed with a fabricated value.
- Spatial splits are a genuine 5-repeat mean ± std (`repeated_huc8_group_
  kfold`), not a single split — an early version of this project used
  scikit-learn's `GroupKFold`, which has no `random_state` parameter at
  all in the installed version, so every "spatial CV" number before this
  fix was the same single partition reported under different seeds.
- Hyperparameters are tuned via `RandomizedSearchCV`, one pass per model,
  scored on its own seeded HUC8-group split — not nested inside every
  reporting fold, which would multiply cost by repeats × folds and isn't
  feasible at this row count. A wider search budget was tried and
  discarded (see Results) because it didn't change the deployed model's
  accuracy.

## Results

**Best spatial model: the voting ensemble (equal-weight average of random
forest, extra trees, HistGBM, XGBoost), R² = 0.273 ± 0.001, MAE = 0.210.**
Under random k-fold CV it reaches R² = 0.312. Leave-one-HUC8-out: 41 of 44
individually held-out watersheds are positive.

What got it there, in order of impact:

1. **The NYSDOT structure inventory** (age, condition, material, geometry)
   was the single largest jump (R² 0.184 → 0.263), because it's the first
   predictor category that reflects the structure itself rather than its
   surrounding landscape — which is also what NAACC's own scoring formula
   is actually built on.
2. **A residual diagnosis, then two standard training techniques.**
   Breaking down errors by structure-match status and passability decile
   showed the model is worst exactly where it matters most (bottom-decile
   MAE 0.47 vs. 0.09–0.15 mid-deciles) — a measured version of DSL's own
   "predicts toward the mean" note. Missingness indicators on every
   imputed feature, plus a plain equal-weight voting ensemble of the four
   tree/boosting models, took R² from 0.265 to 0.274 spatial / 0.311
   random — the first time this project matched DSL's headline number,
   under the easier of the two CV standards.
3. **NOAA Atlas 14 storm-intensity data** (24-hour/10-year design-storm
   depth — distinct from the mean-annual precipitation already present)
   gave a real, twice-replicated gain on a single model (0.267 → 0.269)
   that didn't carry through to the full ensemble (statistically
   unchanged). Kept anyway: it shows no multicollinearity dilution of
   other features, it's genuinely new verified data, and a real
   single-model gain not reaching the ensemble average is itself a
   legitimate, reportable finding rather than grounds to discard it.

What was tried and didn't help, briefly (full account in the internal
experiment log): StreamCat local-catchment-scale metrics and a computed
constriction ratio both diluted other features' importance via
multicollinearity and were reverted; a `StackingRegressor` meta-learner was
abandoned as infeasible (~6x the compute for an uncertain gain); widening
the structure-match radius lowered accuracy (farther matches are less
reliable); inverse-density sample weighting traded away overall accuracy for
a small tail gain without fixing the underlying under-dispersion; LightGBM
as a 5th ensemble member added no diversity over the existing boosted-tree
members; two more NYSDOT fields (structure type, inspection recency) tested
flat.

**Why R² ≈ 0.27 looks like a real ceiling, not a shortfall.** The
`passability` target is a weighted composite of 13 published NAACC
sub-components (`src/naacc_score.py::WEIGHTS`). Testing each component's own
GIS-predictability separately: the four components tied to structure
geometry (openness, height, outlet drop, constriction — 34.8% of the
target's weight) are genuinely learnable, each reaching R² comparable to or
better than the full composite. Everything that requires an actual site
observation — physical barriers, armoring, scour pool, substrate match,
water depth, water velocity — sits at R²≈0 no matter what predictors are
added, and together these account for 50.7% of the target's weight. Those
facts (is there a logjam today, does the water depth right now match the
channel) don't exist anywhere outside a field survey. This is a quantified
version of DSL's own "doesn't predict the extremes" limitation.

**Residuals are spatially clustered, but at a scale this project's CV
standard is built to exclude.** K-nearest-neighbor residual correlation is
0.18–0.19 (vs. ≈0 for a shuffled null), but 95.3% of a crossing's nearest
geographic neighbors share its own watershed, and the clustering lives
almost entirely within-watershed (correlation 0.186) rather than across
(−0.003). A fold-safe spatial-lag feature (mean passability of nearest
already-surveyed neighbors) helps under random CV (0.301 → 0.303) but hurts
under spatial CV (0.263 → 0.248–0.255), because holding out whole watersheds
removes exactly the neighbors the feature would need. Not a bug — a genuine,
scenario-dependent result: useful for filling in gaps in an
already-partly-surveyed watershed, not for predicting into an unsurveyed
region, which is the harder problem this project is built around.

## On the comparison to DSL's 0.31

DSL's figure is a Random Forest out-of-bag (OOB) estimate. OOB scores each
tree on rows left out by bootstrap resampling of the whole training set, not
rows held out by region — nearby, spatially correlated crossings routinely
end up split across a tree's in-bag/out-of-bag rows. That makes it
methodologically closer to this project's random-CV column (0.312, past
DSL's figure) than to the spatial-CV column, which is the harder, more
honest number for how this would generalize to an unsurveyed region (0.273).
Neither number should be quoted without the other — they answer different
questions.

## Known limitations

- The NYSDOT structure inventory covers 21.3% of crossings — the
  state-maintained set, mostly bridges and larger/state-system culverts.
  For the rest, structure fields are missing and median-imputed like any
  other missing value. A live check this project ran found no independent,
  more-complete small-culvert inventory exists for NY outside of
  NAACC-derived sources.
- The model still under-predicts extremes (prediction/observation SD ratio
  ≈0.51–0.53, versus 1.0 for a model that fully captures the target's
  spread) — directionally fixed from early versions, not solved.
- This is one state, a reduced predictor set relative to DSL's 13-state
  model, and a demonstration/research pipeline — not adopted or used by
  NAACC/NACC, and not a claim of being ready for operational use.

## References

- Plunkett EB, McGarigal K, Compton BW, Jackson SD, DeLuca WV, Grand J. 2022.
  *Designing Sustainable Landscapes: Aquatic Barriers settings variable.*
  UMass Amherst. (`docs/DSL_documentation_abarriers.pdf`)
- North Atlantic Aquatic Connectivity Collaborative. *Aquatic Connectivity
  Scoring Systems for Non-tidal Crossings* (rev. 2016-06-16). streamcontinuity.org.
- NYS DEC. *NYS NAACC Surveys* ArcGIS feature layer.
- Mulvihill, Baldigo, Miller, DeKoskie & DuBois. 2009. *Bankfull Discharge
  and Channel Characteristics of Streams in New York State.* USGS SIR
  2009-5144.
