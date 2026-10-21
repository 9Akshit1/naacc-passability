# Joining the National Bridge Inventory to crossings

This is the piece of the project that could be added to another pipeline (for example the DSL one). It attaches the
nearest federal bridge-inventory structure to each crossing. All code is in `src/nbi.py`; it needs only pandas,
numpy, scipy, pyproj and requests.

## Fastest route

If the crossings are the 56,525 in Plunkett's file, the join is already done:
`results/nbi/nbi_for_plunkett_crossings.csv` has one row per matched crossing (9,248 rows), keyed by `crossing_code`.
Left-join it on `crossing_code`; a crossing with no row has no structure, so set `nbi_matched` to 0 for it and leave
the other columns missing.

## For any other list of crossings

```
python -m src.nbi download --states NY PA CT        # FHWA state files go to data/raw/nbi2025/
python -m src.nbi join --crossings crossings.csv --id-col crossing_code --lat-col lat_nad83 --lon-col lon_nad83 \
                       --out nbi_joined.csv
```

`--crossings` can be CSV or parquet. Options: `--radius` (metres, default 60), `--one-way` (see below),
`--matched-only`, `--dir` (where the NBI files are), `--year`. Download every state whose structures could lie
within the radius of a crossing, including neighbouring states near a border.

Source files: `https://www.fhwa.dot.gov/bridge/nbi/2025/delimited/NY25.txt` (swap the state and the two-digit year).
Index page: https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm

## What the join does

1. Keep structures whose item 42B (service under the structure) is 5, 6, 7, 8 or 9, meaning a waterway is
   involved. In the 13 states that is 68,102 structures after step 2.
2. Decode the position. Item 16 is `DDMMSSss` and item 17 is `DDDMMSSss`, with two implied decimals on the seconds.
   Records whose minutes or seconds are 60 or more are dropped (91 of them), as are positions outside the region.
3. Project both sets of points to EPSG:5070 and take the nearest structure to each crossing. By default the pair
   must also be mutual (the structure's nearest crossing is that crossing) and lie within 60 m.
4. Copy the structure's fields. Year 0 or 9999 and zero or negative sizes become missing; condition ratings that are
   not a single digit become missing.

Matched share in Plunkett's file: 16.4% (9,248 of 56,525); with the one-way rule 16.9%. Median distance to the
matched structure is 6.5 m. NBI lists public-road structures longer than about 6.1 m, so small culverts have no
entry. "No match" therefore means "probably not a large public-road structure", and that is informative.

## Which columns are worth adding

Development set (47,530 crossings), grouped by HUC8, LightGBM with fixed settings, three differently seeded splits.
The locked basins were not used. From `scripts/plunkett_nbi_minimal.py`:

| Inputs | R2 |
|---|---|
| Plunkett's 39 predictors | 0.2873 |
| plus a 0/1 "matched" flag only | 0.3070 |
| plus matched, structure length, maximum span | 0.3114 |
| plus the eight size fields (length, span, number of spans, deck width, roadway width, vertical and lateral clearance, improvement length) | 0.3119 |
| plus every NBI field joined (24 columns) | 0.3126 |

About four fifths of the gain comes from knowing that a large structure is there at all. Size adds a little more.
Condition, scour, waterway adequacy, owner, traffic, skew and year built add almost nothing once size is in. If the
aim is a small, easy addition, three columns are enough.

Mutual versus one-way matching made no difference (R2 0.3126 against 0.3124, `scripts/plunkett_nbi_oneway.py`). When
joining to every crossing on a network rather than only surveyed ones, the one-way rule is the safer choice because
its result does not depend on which other crossings exist.

## Limits

- Positions are imprecise. In New York, the nearest structure's stream name agrees with the surveyed crossing's name
  81% of the time within 10 m and 54% at 60 to 100 m (the chance level is 1.5%). Some matches are wrong and some true
  matches lie farther out. Results barely change between 30 m and 150 m.
- The 2025 file is a snapshot. For 398 of the 9,248 matched crossings (4.3%) NBI gives a build year later than the
  survey year, so the surveyed structure has since been replaced.
- Vertical and lateral clearance under the structure are 0 in many records where the item does not apply to streams.
  They are left as they come; they carry little signal.
- Nothing from the NAACC survey is used in the join. The 2025 inventory exists for crossings that were never surveyed,
  so these columns are available when the model is used for real.
