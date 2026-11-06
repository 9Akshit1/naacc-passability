# What a one-question field check adds, in all 13 states

This file describes a test of a different product from the main model. The main model predicts a crossing's score from map data only. Here the model is also given one answer that a person at the crossing could give: is the outlet of the culvert perched, meaning that the water falls more than 0.25 ft from the outlet to the water below it? The question matters because the NAACC score is partly computed from the outlet drop, and no map layer measures it (see "Why accuracy stops where it does" in [RESULTS.md](RESULTS.md)).

All of the work below uses answers taken from NAACC surveys, where a trained surveyor measured the drop. It does not show how accurately volunteers would answer the question. That has not been measured.

## Where the outlet drop values came from

Plunkett's file does not contain the outlet drop. The first version of this test used the older New York survey export, which is why it covered only New York. NAACC also publishes a public crossing layer (the ArcGIS service `NAACC_Culvert_Assessment_2`, hosted by NAACC and The Nature Conservancy, which needs no login) that has the outlet drop for all 13 states. `scripts/build_outlet_drop_join.py` joins it to Plunkett's crossings. A layer row is accepted for a crossing only when the two points are within 30 m of each other and the NAACC aquatic scores in the two files differ by less than 0.02. Rows with the same survey id are tried first, and the crossings that are left are then matched by location and score alone. Crossings still left empty after that get a value from the older New York export when it has one.

In the shapefile version of the layer, `Outlet_D_1` is the drop to the water surface and `Outlet_Dro` is the drop to the stream bottom. These were checked against the older New York export, and they agree on 97% of the non-zero rows. A value of -1 means "not measured" and is treated as missing.

The layer has one row for each measured pipe, so a crossing with two pipes appears twice with the same score and sometimes two different drops. For 1,076 crossings the drops differed by more than 0.05 ft, and for 431 of them the yes/no answer itself differed. Which pipe a single answer should describe is not defined, so these crossings are given no outlet drop value. After that change, the join agrees with the older New York export on 99.4% of crossings that appear in both.

The result is an outlet drop for 31,140 of the 56,525 crossings (55%). The other crossings were left empty. We did not fill them in from the map, because predicting the outlet drop from the map is exactly the part that does not work well.

## Why 45% of the crossings have no value

For 83% of the crossings without a value, the public layer has no point at all within 100 m, so they are simply not in it. There are two reasons:

- Plunkett's file draws on two NAACC databases, and the older one (the UMass Stream Continuity Project, survey ids below 20,000) is not in the public layer. This is 5,403 crossings, including 95% of the crossings in New Hampshire, 83% in Vermont, 60% in Rhode Island, 40% in Massachusetts and 20% in Connecticut.
- The public layer was saved around November 2020, and its highest survey id is 76,288. 23,663 of Plunkett's crossings have a higher id, because they were added in the April 2021 export he used. For ids inside the layer's range, 87% of crossings are covered.

The crossings without a value are therefore the oldest New England surveys and the newest surveys of any state, and they are not a random sample. The labelled crossings are mostly in New York, Pennsylvania and Maine. The numbers of labelled crossings in the weakest states are New Hampshire 18, Rhode Island 18, Vermont 28, Delaware 39, West Virginia 301 and New Jersey 523. The state of each match was checked as well: of 31,992 matches, 22 had different states in the two files (21 of them on the West Virginia and Virginia border), and the rest agree.

The map-only model still scores all 56,525 crossings in all 13 states. Only this one-question test is limited by the missing outlet values.

## One model or several

We compared one model trained on all states with separate models for nearby states, with an average of the two, and with a model that weights training crossings by how similar they are to the target state. Each was tested by leaving out one state at a time and predicting it from the others, and also by cross-validation inside states that have local data. The results are in `results/plunkett/regions_A_2026-11-05.log`, `regions_B_2026-11-05.log` and `regions_C_2026-11-05.log`.

| Task, all held-out crossings together | One model | Nearby states | Average of both | Similarity weights |
|---|---|---|---|---|
| Map only, no raster inputs | 0.265 | 0.253 | 0.263 | 0.248 |
| Map only, with raster inputs | 0.310 | 0.293 | 0.306 | 0.297 |
| Map plus the yes/no, no raster inputs | 0.692 | 0.690 | 0.692 | 0.684 |
| Map plus the yes/no, with raster inputs | 0.703 | 0.699 | 0.703 | 0.695 |

One model for all states is the best or tied for best in every row. Models for nearby states lose because each one sees less data and the states are not different enough to make up for it. Inside states that have local data, one model scores 0.277 for map only and 0.698 with the yes/no, against 0.260 and 0.650 for a separate model for each state. We therefore use one model for all states and no state input. How well it carries over varies by state: New Jersey and New Hampshire are poorly predicted from the other states, and Maryland, West Virginia and Vermont are predicted well.

## Score on the set-aside basins

The same final model (`src/plunkett_final.py`) was trained on the development crossings that have an outlet drop and scored on the 5,102 set-aside crossings that have one (New York 1,997, Pennsylvania 1,315, Virginia 926, Maryland 795, Delaware 39, West Virginia 29, New Jersey 1). The inputs include no raster layers. The script is `scripts/plunkett_locked_quickcheck.py`. This is the second time these basins were used for this test, after a first run that kept an arbitrary pipe for multi-pipe crossings (first run: 0.194 for the map only and 0.673 with the yes/no).

| | R² | 95% interval | MAE |
|---|---|---|---|
| Map only, same crossings | 0.205 | 0.163 to 0.233 | 0.216 |
| Map plus yes/no | 0.698 | 0.658 to 0.718 | 0.134 |
| Difference | +0.494 | 0.462 to 0.530 | |

By state, the map-only score and the score with the yes/no are 0.19 and 0.69 in New York, 0.14 and 0.69 in Pennsylvania, 0.28 and 0.72 in Virginia, and 0.03 and 0.60 in Maryland.

## What the 0.70 means

On the development crossings, the yes/no answer by itself explains an R² of 0.618, the exact outlet drop by itself explains 0.750, the map by itself explains 0.229, the map with the yes/no explains 0.683, and the map with the exact drop explains 0.806. So almost all of the 0.70 comes from the yes/no, and the map adds about 0.065 on top. Because the score is partly computed from the outlet drop, this is mostly arithmetic and it is not evidence of a clever model. The map-only score of 0.320 on the set-aside basins is unchanged by any of this.

Crossings with a recorded outlet drop have lower scores on average than the others (0.61 against 0.69), so they are not a random sample of the file.

## If volunteers make mistakes

Volunteers will sometimes answer wrongly, and we have not measured how often. To see what that would do, we simulated wrong answers on the development crossings (26,038 with a value; log `results/plunkett/quickcheck_noise_2026-11-05.log`). The table shows R² when the answers in the test data are flipped at random with the given probability, for models trained on answers with different amounts of error. The map-only R² on these crossings is 0.233.

| Model trained on answers with this error rate | 0% wrong | 10% wrong | 20% wrong | 30% wrong |
|---|---|---|---|---|
| 0% (surveyor answers only) | 0.695 | 0.403 | 0.121 | -0.182 |
| 10% | 0.638 | 0.493 | 0.350 | 0.205 |
| 20% | 0.565 | 0.469 | 0.373 | 0.278 |
| 30% | 0.464 | 0.407 | 0.350 | 0.293 |

A model trained only on correct answers falls apart when the answers are wrong. At 30% wrong it does worse than the map alone. A model trained on answers with about the same error rate never falls below the map-only score in this test, and it still ranks crossings better than the map does. At 20% wrong answers and 20% training error, its rank correlation is 0.62 (map only 0.51), and it catches 53% of the worst 20% of crossings (map only 36%).

More realistic errors, where a truly perched outlet is called "no" with probability e and a non-perched outlet is called "yes" with probability e divided by 6, hurt less. With e at 30%, the R² is 0.376 for a model trained on correct answers and 0.431 for one trained at 0.2. Training on noisy answers costs accuracy when the volunteers are in fact accurate: R² falls from 0.695 to 0.565 in the table above.

The error rates in these tests are assumptions. Before this could be used, volunteers would have to answer the question at crossings that surveyors have also measured, so that the share of missed perched outlets and the share of false alarms can be measured separately. False alarms matter more, because most outlets are not perched. The model should then be trained on volunteer answers, or on answers simulated from the measured error rates, and not on surveyor answers alone.

## Which pipe

Among the crossings where the pipes disagree, the NAACC score follows the lowest drop best (rank correlation -0.83, against -0.58 for the highest drop). We did not turn this into a rule. Which pipe a volunteer should look at would have to be decided as part of the field protocol.

## What this does not show

It does not show that volunteers can answer the question accurately, or how the model performs outside the 13 NAACC states. Only New York, Pennsylvania, Virginia and Maryland (plus a few crossings in Delaware and West Virginia) are in the set-aside test, and the other states are tested only by leaving each state out in development. It also says nothing about crossings with no recorded outlet drop, which are 45% of the file and may differ from the rest.

## Permission to use the public layer

The public layer is an open service hosted by NAACC and The Nature Conservancy. It needs no login, and its metadata contains no licence or terms of use. Plunkett's data was compiled from NAACC database exports, so the layer is the same underlying source and not an independent one. The download script (`scripts/download_naacc_public_layer.py`) was first written to compare scores and was then also used for this join. The layer files themselves are not committed to this repository. This is not a legal clearance, and NAACC and Plunkett should be asked before results built on the layer are used outside this project.

## What would fix the gaps

The most direct fix is for Plunkett or NAACC to provide the survey fields for his crossings from the April 2021 export: outlet drop to the water surface, outlet grade, and the number of structures at each crossing, keyed by `crossing_survey_id`. That would cover all 56,525 crossings with authorised data and would settle the question of which pipe a single answer describes. We could not find another organisation that holds these values. The Connecticut and Vermont culvert inventories and the New York Department of Transportation inventory were tested and do not include the outlet drop.
