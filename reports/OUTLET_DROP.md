# What a one-question field check adds, in all 13 states

This file describes a test of a different product from the main model. The main model predicts a crossing's score from map data only. Here the model is also given one answer that a person at the crossing could give: is the outlet of the culvert perched, meaning that the water falls more than 0.25 ft from the outlet to the water below it? The question matters because the NAACC score is partly computed from the outlet drop, and no map layer measures it (see "Why accuracy stops where it does" in [RESULTS.md](RESULTS.md)).

All of the work below uses answers taken from NAACC surveys, where a trained surveyor measured the drop. It does not show how accurately volunteers would answer the question. That has not been measured.

## Where the outlet drop values came from

Plunkett's file does not contain the outlet drop. The first version of this test used the older New York survey export, which is why it covered only New York. NAACC also publishes a public crossing layer (the ArcGIS service `NAACC_Culvert_Assessment_2`, hosted by NAACC and The Nature Conservancy, which needs no login) that has the outlet drop for all 13 states. `scripts/build_outlet_drop_join.py` joins it to Plunkett's crossings. A layer row is accepted for a crossing only when the two points are within 30 m of each other and the NAACC aquatic scores in the two files differ by less than 0.02. Rows with the same survey id are tried first, and the crossings that are left are then matched by location and score alone. Crossings still left empty after that get a value from the older New York export when it has one.

In the shapefile version of the layer, `Outlet_D_1` is the drop to the water surface and `Outlet_Dro` is the drop to the stream bottom. These were checked against the older New York export, and they agree on 97% of the non-zero rows. A value of -1 means "not measured" and is treated as missing.

The layer has one row for each measured pipe, so a crossing with two pipes appears twice with the same score and sometimes two different drops. For about 1,080 crossings the drops differ by more than 0.05 ft, and for 431 of them the yes/no answer itself differs. Until 2026-11-09 these crossings were given no outlet drop value. Plunkett then explained that NAACC scores each structure of a multi-part crossing and keeps the maximum passability, so the score follows the most passable structure. These crossings now get the lowest drop among their pipes. The join agrees with the older New York export, which keeps one pipe, on 97.5% of the 8,163 crossings that appear in both (99.4% when the multi-pipe crossings were left out).

The result is an outlet drop for 32,223 of the 56,525 crossings (57%). The other crossings were left empty. We did not fill them in from the map, because predicting the outlet drop from the map is exactly the part that does not work well.

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

### The lowest-drop rule, and why the table above was not redone

The table above was scored before the multi-pipe rule changed (see "Where the outlet drop values came from"), so its 5,102 set-aside crossings exclude the ones whose pipes disagree. We did not score the set-aside basins a third time, because that would use up the held-out test again for a small change. We measured the change on the development crossings only, with the same checks as in "One model or several" (logs `results/plunkett/regions_B_min_rule_2026-11-09.log` and `regions_C_min_rule_2026-11-09.log`). The number of development crossings with a recorded drop rose from 26,038 to 26,885.

| Development crossings, leave one state out, all held-out rows together | Before | With the lowest drop |
|---|---|---|
| Map only, no raster inputs, same crossings as the next row | 0.2195 | 0.2132 |
| Map plus yes/no, no raster inputs | 0.6922 | 0.6891 |
| Map plus yes/no, with raster inputs | 0.7030 | 0.6996 |
| Cross-validation inside states, map plus yes/no, one pooled model | 0.6979 | 0.6952 |

The yes/no still adds about 0.48 on these crossings, and one pooled model is still the best or tied in every row, so the decision does not change. The scores are about 0.003 lower because the added crossings have several pipes and are harder to predict. The set-aside figures of 0.205 and 0.698 were measured on the earlier row set. The next section repeats the set-aside test with the detailed NAACC export.

### Third scoring, with the detailed export

After the detailed export was joined (see "What would fix the gaps"), the development checks were repeated first (logs `results/plunkett/regions_B_export_2026-11-11.log` and `regions_C_export_2026-11-11.log`). The development crossings with a recorded drop rose to 38,782. Leaving one state out, one pooled model was still the best or tied, and the yes/no still added about 0.47 (map only 0.211, map plus yes/no 0.684, no raster inputs).

The set-aside basins were then scored a third time with a plan written beforehand (`results/plunkett/locked_plan_T5_v3.txt`). Only the outlet drop changed. The models, the inputs, the 0.25 ft cut-off and the way the interval is found stayed the same, and nothing was adjusted after the result was seen. The script is `scripts/plunkett_locked_quickcheck_v3.py`. The model was trained on the 38,782 development crossings with a drop and scored on 7,767 set-aside crossings that have one (5,102 before).

| | R� | 95% interval | MAE |
|---|---|---|---|
| Map only, same crossings | 0.207 | 0.181 to 0.227 | 0.216 |
| Map plus yes/no | 0.693 | 0.667 to 0.711 | 0.134 |
| Difference | +0.486 | 0.465 to 0.512 | |

By state, the map-only score and the score with the yes/no are 0.19 and 0.70 in New York, 0.20 and 0.69 in Pennsylvania, 0.24 and 0.70 in Virginia, and 0.03 and 0.58 in Maryland. The result is close to the second scoring (0.205 and 0.698), with about 50% more test crossings and a narrower interval. These basins have now been used three times for this test, so the three results are not independent of each other. The data changed between the scorings and the model did not.

With the same fitted model, flipping the yes/no at random in the test crossings gave R� of 0.41 when 10% of the answers were wrong, 0.13 when 20% were wrong and -0.16 when 30% were wrong. This is the same collapse as in "If volunteers make mistakes", and it applies to a model trained on surveyor answers only.

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

## Permission to use NAACC data

The public layer is an open service hosted by NAACC and The Nature Conservancy, and its metadata has no licence text. Ethan Plunkett, who works with NAACC's data, replied by email on 2026-11-09 that it is fine to use NAACC data for whatever we want, with no restrictions. The layer files themselves are still not committed to this repository, and the download script (`scripts/download_naacc_public_layer.py`) is included so the data can be fetched again.

## What would fix the gaps

The gaps are the crossings the public layer does not cover. Plunkett pointed out that NAACC's Data Center (naacc.org, "Search Crossings") can export a detailed Excel or CSV file of the surveys taken after 6/1/2015, which should have the outlet drop to the water surface, the outlet grade and the number of structures, one row for each structure. He described how to collapse it. A crossing with more than one survey has rows for each survey id, so we use the survey id Plunkett's file already names. A crossing with more than one structure has several rows with the same survey id. His own score is the maximum passability over the structures, so the single drop used here is the lowest one (he also named the mean).

The detailed export was downloaded from the Data Center on 2026-11-11 (the "NAACC (after 6/1/2015)" dataset, no filters: 101,300 rows, 90,061 survey ids, all 13 states). `scripts/build_outlet_drop_from_export.py` joins it to Plunkett's crossings by `crossing_survey_id`, accepts a row only when its location is within 30 m of Plunkett's point and its aquatic score differs by less than 0.02, and takes the lowest drop when a crossing has several structures. The log is `results/plunkett/outlet_drop_export_join_2026-11-11.log` and the result is `data/processed/plunkett_outlet_drop_export.parquet`.

| | Public layer join | Detailed export join |
|---|---|---|
| Crossings with an outlet drop | 32,223 of 56,525 (57%) | 46,522 of 56,525 (82%) |
| Crossings with several structures | | 6,503 (1,822 differ by more than 0.05 ft) |

Where both have a value (32,196 crossings), they are equal on 99.3%. The 14,326 crossings that only the export covers are mostly the surveys added after November 2020. The share with a value is now 92% in New York, 94% in Virginia, 89% in New Jersey, 84% in Pennsylvania, 80% in West Virginia, 77% in Connecticut and 57% in Massachusetts, but still only 5% in New Hampshire, 17% in Vermont and 40% in Rhode Island, which were mostly surveyed under the older UMass project.

The Data Center also has a detailed export of that older UMass dataset (13,206 rows, downloaded the same day). It matches 4,820 of Plunkett's 5,403 older crossings by survey id, location and score. Its outlet drop is stored differently, as text: `0`, whole numbers up to 48 and ranges (`1-6`, `6-12`, `12-24`, `>24`), with a column saying whether it was measured or estimated. The values look like inches, but no document says so, and `1-6` straddles the 3 inch (0.25 ft) cut-off, so it is not used yet. The units are an open question for NAACC.

The sections "One model or several" and "If volunteers make mistakes" and the first two set-aside tables are from the public-layer join. The third set-aside scoring above and the development checks it followed use the export join.
