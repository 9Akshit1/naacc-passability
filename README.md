# NAACC — GIS prediction of aquatic passability for unsurveyed crossings

Predicts the continuous NAACC aquatic passability score (0–1) of a
road-stream crossing from GIS-derived predictors only — the information
available without sending a crew to survey it. This reproduces the
*structure* of the UMass DSL / Plunkett approach (the model whose
predictions feed the DSL Critical Linkages Analysis) with spatial
cross-validation and uncertainty, on New York data.

**Status: research pipeline, prototype-scale.** Not an improvement on the
DSL model, not adopted by NAACC/NACC. See `reports/METHODOLOGY.md` for exact
scope — several DSL predictors could not be reproduced in this environment
and are documented as omitted there.

## What's done

| | |
|---|---|
| **Data** | 44,871 NY NAACC surveys, public (NYS DEC ArcGIS), documented provenance → 28,141 unique surveyed crossings after a DSL-style filtering recipe |
| **Phase 1 audit** | `reports/01_dataset_audit.md` — score distribution, missingness, HUC8/county/year breakdown, repeat-survey analysis |
| **Phase 2** | `src/naacc_score.py` re-implements the NAACC 2016 scoring algorithm, checked against NAACC's own published score (r = 0.89, MAE = 0.055, `reports/02_score_reconstruction.md`) |
| **Modelling** | `src/model_experiments.py` — model ladder, random vs. HUC8-group spatial CV, permutation importance, conformal prediction intervals, calibration (`reports/03_model_comparison.md`) |
| **Inference** | `src/predict.py` — crossings CSV → predicted passability + 90% interval |
| **Tests** | `tests/test_naacc.py` |

## Install & run

```
cd naacc-passability
python -m pip install -r requirements.txt

# 1. data is already in data/raw/. Rebuild the modelling dataset + audit:
python -m src.data
python -m src.audit                                # -> reports/01_dataset_audit.md

# 2. score reconstruction (Phase 2):
python -c "from src.data import build_modeling_dataset as b; from src.naacc_score import validate_against_published as v; import json; print(json.dumps(v(b()[0]), indent=2))"

# 3. fetch GIS predictors for the modelling subset (slow; caches + resumes):
python -m src.build_subset                          # -> data/processed/subset_predictors.parquet

# 4. run the experiments:
python -m src.model_experiments                     # -> reports/03_model_comparison.md, results/

# 5. predict for new crossings:
python -m src.predict --crossings my_crossings.csv --out predictions.csv
#    my_crossings.csv needs: crossing_id, lat, lon   (optional: County)

# optional: save the fitted model instead of retraining on every call
python -m src.predict --save-model                  # -> results/models/voting_ensemble.joblib
python -m src.predict --crossings my_crossings.csv --out predictions.csv
#    (automatically loads the saved model above if present; --retrain forces a fresh fit)
```

Re-download the raw NY data (if the NYS DEC layer updates):
```
python scripts/download_ny_naacc.py
```

## Repository layout

```
naacc-passability/
├── data/
│   ├── raw/            ny_naacc_surveys.csv + PROVENANCE.json
│   ├── interim/        cleaning report (the per-crossing API response cache
│   │                   is omitted from this export for size; build_subset.py
│   │                   regenerates it automatically, fetching anything missing)
│   └── processed/      modeling_dataset.parquet, subset_predictors.parquet
├── src/
│   ├── data.py             raw -> clean modelling dataset (documented filtering)
│   ├── naacc_score.py      NAACC 2016 scoring algorithm reconstruction
│   ├── audit.py            Phase 1 dataset audit
│   ├── gis_predictors.py   NLDI, 3DEP, Census, StreamCat, NYSDOT, NOAA — with cache + backoff
│   ├── build_subset.py     select the modelling subset + fetch its predictors
│   ├── spatial_cv.py       HUC8 group hold-out vs random k-fold
│   ├── models.py           the model ladder
│   ├── evaluate.py         metrics, calibration, prioritisation overlap
│   ├── uncertainty.py      split conformal intervals
│   ├── model_experiments.py the experiment runner
│   └── predict.py          inference pipeline
├── reports/           METHODOLOGY.md, 01_dataset_audit.md, 02_score_reconstruction.md, 03_model_comparison.md
├── docs/              DSL + NAACC source PDFs
├── results/           figures/, tables/, experiment_results.json
└── tests/
```

## Results

Best spatial model (voting ensemble of random forest, extra trees, HistGBM,
XGBoost): **R² = 0.273, MAE = 0.210** under HUC8-group spatial
cross-validation — the harder, honest standard, since it holds out entire
watersheds rather than individual rows. Under ordinary random k-fold CV it
reaches **R² = 0.312**, past DSL's reported OOB R² = 0.31, though that's the
easier comparison (DSL's OOB estimate doesn't fully control for spatial
autocorrelation either — see `METHODOLOGY.md` for why).

The full story — what was tried, what worked, what didn't, and why R² ≈ 0.27
looks like close to a real ceiling for this problem rather than a modeling
shortfall — is in `reports/METHODOLOGY.md`.

## Key honest caveats

1. DSL's baseline (OOB R² = 0.31) is on a different, larger predictor set, a
   13-state sample, and isn't directly comparable methodologically — see
   `METHODOLOGY.md` for the full comparison.
2. `road_class` comes from OpenStreetMap (nearest tagged way), not the
   survey's own field-observed value, which would leak site-visit
   information into a model meant to run without one.
3. Drainage area is the NLDI basin polygon area, not a re-run D8 grid.
4. The 14 EPA StreamCat metrics are watershed-scale aggregates from national
   land-cover/climate/soils layers, not locally verified for these specific
   crossings.
5. The modelling subset (21,111 of 28,141 crossings, capped at 1,000 per
   watershed) covers every NY watershed at least once but isn't exhaustive —
   some watersheds have far more surveyed crossings than the cap.
6. The NYSDOT structure inventory (age, condition, material, geometry)
   covers only 21.3% of crossings — the state-maintained set, mostly
   bridges and larger/state-system culverts. No more complete inventory was
   found to exist for NY.
7. Distinguish carefully: crossings **scored** (28,141) ≠ predictions
   **produced** ≠ rankings **changed** ≠ model **adopted**. Only the first
   has happened.
