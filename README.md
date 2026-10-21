# NAACC passability from GIS data

Predicts the NAACC aquatic passability score (0 to 1) for a road-stream crossing from map data alone, so that
crossings nobody has surveyed can be ranked. It follows the approach of the UMass Designing Sustainable
Landscapes (DSL) aquatic barriers model and was reworked in October 2026 after feedback from Ethan Plunkett,
using his 56,525-survey, 13-state file.

This is research code. It has not been adopted by NAACC or DSL, has not been tested on unsurveyed crossings,
and is not deployed anywhere.

## Result in one table

Scored once on 8,995 crossings in 7 whole river basins that were set aside before any model was fitted
(40 HUC8 watersheds; PA, NY, VA, MD, NJ, DE, WV). R2 is on the NAACC score.

| Model | R2 | MAE |
|---|---|---|
| Predict the training mean | 0.000 | 0.258 |
| Random forest on Plunkett's 39 predictors | 0.286 | 0.208 |
| Boosted ensemble, same 39 predictors | 0.299 | 0.205 |
| plus FHWA National Bridge Inventory (NBI) | 0.315 | 0.200 |
| plus OpenStreetMap bridge and culvert tags | 0.320 | 0.199 |

The full model beats the random forest by 0.034 R2 (95% interval 0.026 to 0.045, resampling watersheds).
That is a real but small gain. Most of the unexplained variance comes from whether a culvert outlet is perched,
which no map layer measures. Details and caveats are in [reports/RESULTS.md](reports/RESULTS.md); every correction
made along the way is in [reports/CHECKS.md](reports/CHECKS.md).

The NBI part is the easiest piece to reuse. [docs/NBI_JOIN.md](docs/NBI_JOIN.md) explains it, and
`python -m src.nbi` downloads the federal files and joins them to any list of crossings.

## Setup

```
python -m pip install -r requirements.txt
python -m pytest -q tests          # 65 tests; 57 run without Plunkett's data, the other 8 skip until it is in place
```

Plunkett's data file is not in this repository because it is his to distribute. Put `passability_data.parquet`
and `passability_predvars.txt` in `data/raw/plunkett/`, or point the variable `NAACC_PLUNKETT_DIR` at the folder
that holds them.

## Reproducing the multi-state results

```
python -m scripts.download_multistate_inputs          # state and watershed boundaries, NBI files, OSM
python -m src.plunkett_data                           # adds state and HUC8/6/4 to each crossing
python -c "from src import plunkett_features as pf; pf.build().to_parquet('data/processed/plunkett_features_nbi.parquet')"
python -m scripts.build_osm_features                  # adds the OSM columns (T3)
python -m scripts.plunkett_dev_comparison             # development-set comparison, grouped CV
python -m scripts.plunkett_locked_eval T1 T2 T3       # the single look at the locked basins
python -m scripts.verify_headline_numbers             # re-derives the headline numbers with separate code
```

`plunkett_locked_eval` refuses to score a tier a second time. The saved predictions it produced are in
`results/plunkett/`. The longer list of scripts, one per experiment, is in section 9 of
[reports/04_multistate_plunkett_results.md](reports/04_multistate_plunkett_results.md).

## Layout

```
src/            code: nbi.py (NBI download and join), osm_structures.py, plunkett_*.py (data, features,
                cross-validation, final ensemble), naacc_score.py (reconstruction of the 2016 scoring formula),
                and the earlier New York pipeline (data.py, gis_predictors.py, models.py, predict.py, ...)
scripts/        one script per experiment or check
tests/          65 tests (8 need Plunkett's data)
results/        plunkett/ (metrics, saved locked predictions), nbi/ (NBI columns by crossing code), tables/, figures/
reports/        RESULTS.md and CHECKS.md first; 01 to 07 are the dated working reports; history/ keeps superseded texts
docs/           NAACC and DSL source documents, NBI_JOIN.md
```

## What came before

The first version (September 2026) used New York surveys only, a drainage area that turned out to be linked to
the wrong stream segment for many crossings, and a state transportation inventory. Its numbers (R2 about 0.27
under watershed holdout) are kept in `reports/01` to `03` and `reports/history/` for the record but are
superseded. The old `WRITEUP.md` argued that R2 near 0.3 was a ceiling for GIS-only models; later checks
(`reports/CHECKS.md`, item 7) showed that argument was too strong, so it was replaced.

## Data sources

- NAACC survey scores and survey fields: Plunkett's file (13 states); the public NAACC layer was used for cross-checks.
- FHWA National Bridge Inventory, 2025 delimited files, 13 states (public).
- OpenStreetMap through Overpass and a Geofabrik extract (public, ODbL).
- Census state boundaries and USGS HUC8 boundaries (public).
