# Scoring crossings with this model and comparing it with the random forest

This page is for someone who wants to check the model against their own random forest without reading the rest of the repository. It covers what is already computed, how to score a list of crossings, and what the model cannot do.

## What is already computed

The file `results/plunkett/predictions_surveyed_crossings.csv` has one row for each of the 56,525 surveyed crossings in Plunkett's file. It gives the crossing code, state, HUC8 watershed, the NAACC aquatic passability score, and two predictions of that score: one from the boosted ensemble with the bridge inventory and OpenStreetMap inputs (the model called T3 in the README), and one from a random forest on Plunkett's predictors alone.

No prediction in the file used the crossing's own watershed for training. For the 47,530 development crossings the predictions come from 5-fold cross-validation that holds out whole HUC8 watersheds. For the 8,995 crossings in the seven set-aside river basins the predictions come from models trained on the development crossings only. The column `basis` says which of the two applies to each row. Comparing the two prediction columns against `naacc_score` therefore needs no further modelling.

Over all 56,525 crossings the boosted ensemble has an R-squared of 0.326 and the random forest 0.282. On the set-aside basins alone the figures are 0.320 and 0.286, a difference of 0.034 (95% interval 0.026 to 0.045). The ensemble has the higher R-squared in 11 of the 13 states. The two exceptions are Delaware, which has 40 crossings, and New Hampshire, where both models score about 0.04. The gain is largest in Maryland (0.10) and New Jersey (0.08) and smallest in Vermont (0.01).

## Scoring a list of crossings

The script `scripts/score_crossings.py` takes a file with one row per crossing and writes a predicted score for each. The input needs the columns `crossing_code`, `state`, `lat_nad83` and `lon_nad83`, plus every column named in `passability_predvars.txt`. These are the same columns as in `passability_data.parquet`, so a file made the same way as that one will work.

```
python -m pip install -r requirements.txt
python -m scripts.download_multistate_inputs
python -m src.plunkett_data
python -c "from src import plunkett_features as pf; pf.build().to_parquet('data/processed/plunkett_features_nbi.parquet')"
python -m scripts.build_osm_features
python -m scripts.score_crossings --check
python -m scripts.score_crossings --input my_crossings.parquet --out predictions.csv
```

The first five lines are the setup described in the README. They download the bridge inventory and OpenStreetMap data and build the table of surveyed crossings that the model is fitted on. The `--check` line rebuilds the model inputs of all 56,525 surveyed crossings from the raw columns alone and confirms that they equal the stored table, so it shows that the script builds the inputs the same way the experiments did. On the author's computer this check reported no differences. The last line fits the model on all surveyed crossings the first time it runs, which takes a few minutes, and then reuses it.

The output has `crossing_code`, `state`, `predicted_score`, `low_90` and `high_90`. The interval is the prediction plus or minus 0.42, which is the 90th percentile of the absolute error of the cross-validated predictions. It covered 89% of the set-aside crossings. It is wide because the model explains about a third of the variation in the scores, and it is not much use for a single crossing. The predicted score is more useful for ranking many crossings than for deciding about one.

## Limits

The model was fitted to crossings that someone chose to survey, in 13 states. Surveyed crossings are not a random sample of all road and stream crossings, and the model has not been tested on crossings that were never surveyed, because their true scores are not known. Results for a state with few surveys, such as New Hampshire, Delaware or Vermont, rest on very little data.

The bridge inventory match pairs a crossing with a bridge only when each is the other's nearest neighbour, so the script should be given the full list of crossings in the area. A short list will match differently from the full list used in the experiments.

The gain over the random forest is 0.034 in R-squared. Most of the error that remains comes from whether the outlet of a culvert hangs above the water, which no map layer shows. [OUTLET_DROP.md](../reports/OUTLET_DROP.md) reports what one field question about the outlet adds.
