# Predicting culvert passability from map data

Roads cross streams in many places, and a pipe or culvert under a road can stop fish and other animals from moving up and down the stream. The North Atlantic Aquatic Connectivity Collaborative (NAACC) trains staff and volunteers to survey these crossings, and each surveyed crossing gets an aquatic passability score from 0 (blocked) to 1 (fully passable). Most crossings have never been surveyed. The Designing Sustainable Landscapes (DSL) project at the University of Massachusetts Amherst predicts the score for every crossing from map layers alone, such as the size of the drainage area above the crossing, the slope of the stream, the type of road and the income of the county.

This repository is a student project that builds a model of the same kind and tests whether two public datasets make it more accurate: the FHWA National Bridge Inventory and OpenStreetMap. It uses a file of 56,525 surveyed crossings in 13 states (CT, DE, MA, MD, ME, NH, NJ, NY, PA, RI, VA, VT, WV) that Ethan Plunkett of DSL shared with me. That file is not included in this repository because it is his to distribute.

The code is research code. NAACC and DSL have not adopted it, it has not been tried on crossings that were never surveyed, and it is not deployed anywhere.

## Main result

Before any model was built, 7 whole river basins (8,995 crossings) were set aside as a test set, and each model below was scored on them once. The score is R², the share of the variation in the NAACC scores that the model explains. An R² of 0 means the model is no better than guessing the average score every time. MAE is the average size of the error, in score units.

| Model | R² | MAE |
|---|---|---|
| Guess the training average | 0.000 | 0.258 |
| Random forest on Plunkett's 39 predictors | 0.286 | 0.208 |
| Boosted ensemble on the same 39 predictors | 0.299 | 0.205 |
| Ensemble plus the FHWA National Bridge Inventory | 0.315 | 0.200 |
| Ensemble plus the National Bridge Inventory and OpenStreetMap tags | 0.320 | 0.199 |

The last model is 0.034 better than the random forest (95% interval 0.026 to 0.045, found by resampling watersheds). That is a small gain. Most of the error that is left comes from whether the outlet of a culvert hangs above the water, which no map layer shows. A separate test of what a single field question about the outlet would add is in [reports/OUTLET_DROP.md](reports/OUTLET_DROP.md).

The full results are in [reports/RESULTS.md](reports/RESULTS.md), the way the tests were set up is in [reports/METHODS.md](reports/METHODS.md), and the mistakes found and fixed along the way are in [reports/CHECKS.md](reports/CHECKS.md).

The file `results/plunkett/predictions_surveyed_crossings.csv` lists all 56,525 surveyed crossings with the NAACC score, the prediction of the boosted ensemble above and the prediction of the random forest, each made without using that crossing's watershed for training. [docs/SCORING_NEW_CROSSINGS.md](docs/SCORING_NEW_CROSSINGS.md) explains how to read it and how to score a list of crossings with `scripts/score_crossings.py`.

If you only want one piece of this work, the National Bridge Inventory join is the easiest to reuse. [docs/NBI_JOIN.md](docs/NBI_JOIN.md) explains it, and `python -m src.nbi` downloads the federal files and joins them to any list of crossings.

## Setup

```
python -m pip install -r requirements.txt
python -m pytest -q tests
```

There are 65 tests. 57 of them run without Plunkett's data and the other 8 are skipped until it is in place. To run the multi-state code, put `passability_data.parquet` and `passability_predvars.txt` in `data/raw/plunkett/`, or set the environment variable `NAACC_PLUNKETT_DIR` to the folder that holds them.

## Reproducing the multi-state results

```
python -m scripts.download_multistate_inputs        # state and watershed boundaries, bridge files, OpenStreetMap
python -m src.plunkett_data                         # adds state and HUC8/6/4 watershed codes to each crossing
python -c "from src import plunkett_features as pf; pf.build().to_parquet('data/processed/plunkett_features_nbi.parquet')"
python -m scripts.build_osm_features                # adds the OpenStreetMap columns
python -m scripts.plunkett_dev_comparison           # comparison on the development set
python -m scripts.plunkett_locked_eval T1 T2 T3     # scores the set-aside basins
python -m scripts.verify_headline_numbers           # recomputes the headline numbers with separate code
```

`plunkett_locked_eval` refuses to score a model on the set-aside basins a second time. Its saved predictions are in `results/plunkett/`. There is one script per experiment in `scripts/`, and each begins with a short description of what it tests.

## Folder layout

- `src/` holds the code. `nbi.py` downloads and joins the bridge inventory, `osm_structures.py` and `osm_context.py` read OpenStreetMap, the `plunkett_*.py` files hold the data loading, features, cross-validation and final model, `naacc_score.py` rebuilds the 2016 NAACC scoring formula, and the remaining files are the earlier New York pipeline.
- `scripts/` holds one script per experiment or check.
- `tests/` holds the tests.
- `results/` holds metrics, saved predictions on the set-aside basins, and the bridge columns for Plunkett's crossings.
- `reports/` holds the written results.
- `docs/` holds NAACC and DSL source documents and the bridge join guide.

## The first version

The first version, from September 2026, used New York surveys only. It had a drainage-area input that was attached to the wrong stream segment for many crossings, and it used a New York transportation department inventory that does not exist in other states. Its R² of about 0.27 on held-out watersheds is superseded by the multi-state results above.

## Data sources

- NAACC survey scores: Plunkett's file. The public NAACC crossing layer was used to check those scores and to look up outlet drop (see [reports/OUTLET_DROP.md](reports/OUTLET_DROP.md)).
- FHWA National Bridge Inventory, 2025 delimited files for the 13 states.
- OpenStreetMap, through the Overpass service and a Geofabrik extract (ODbL licence).
- Census state boundaries and USGS HUC8 watershed boundaries.

## Reference

Plunkett EB, McGarigal K, Compton BW, Jackson SD, DeLuca WV, Grand J. 2022. Designing Sustainable Landscapes: Aquatic Barriers data product. University of Massachusetts, Amherst.
