# Data dictionary

## Raw data

**File:** `data/raw/ny_naacc_surveys.csv` (44,871 rows)
**Source:** NYS DEC ArcGIS FeatureServer `NYS_NAACCSurveys_Projected` layer 0
(`https://services6.arcgis.com/DZHaqZm9cxOD4CWM/arcgis/rest/services/NYS_NAACCSurveys_Projected/FeatureServer/0`)
— public, no login. **Accessed:** 2026-09-10. Provenance: `data/raw/ny_naacc_PROVENANCE.json`.
Downloaded via the ArcGIS REST `query` endpoint, paginated (2,000/page), geometry
in WGS84.

This is the New York subset of the North Atlantic Aquatic Connectivity
Collaborative (NAACC) road-stream crossing assessment database. One row = one
survey of one structure at one crossing.

### Key raw columns

| column | meaning |
|---|---|
| `Survey_Id` | unique id of one field survey |
| `Crossing_Code` | id of the physical crossing (`xy` + packed lat/lon); repeats for multi-structure crossings and repeat surveys |
| `Aqua_Pass_Score` | **NAACC aquatic passability score, 0-1** (`-1` = "no score - missing data"). The target. |
| `AOP` | coarse class: Full AOP / Reduced AOP / No AOP / no score |
| `Evaluation` | barrier class label (No / Insignificant / Minor / Moderate / Significant / Severe barrier) |
| `Terrestrial_Passage_Score` | terrestrial (not aquatic) passability, 0-1 — not used here |
| `GPS_X_Coordinate`, `GPS_Y_Coordinate` | field GPS lon/lat |
| `Crossing_Type` | Culvert / Multiple Culvert / Bridge / Bridge Adequate / Ford / Inaccessible / No Crossing / ... |
| `Number_Of_Culverts` | count (`-1` = missing) |
| `Tidal_Site` | Yes / No / No data / Unknown |
| `Approved` | NAACC QA flag (1 = approved) |
| `County`, `Municipality`, `Road`, `Stream_Name` | location text |
| `Road_Type` | Paved / Unpaved / Multilane road (>2 lanes) / Driveway / Trail / Railroad / No data |
| `NHD_HUC8_Watershed` | HUC8 subbasin name (55 in NY) — used for spatial cross-validation |
| `Date_Observed` | survey date |
| `Crossing_Span` | constriction: Severe / Moderate / Spans Only Bankfull/Active Channel / Spans Full Channel & Banks |
| `Inlet_Grade`, `Outlet_Grade` | At Stream Grade / Inlet Drop / Perched / Free Fall / Cascade / Clogged / ... |
| `Outlet_Drop_To_Water_Surface`, `Outlet_Drop_To_Stream_Bottom` | ft (`-1` = missing) |
| `Inlet_Openness`, `Outlet_Openness` | openness ratio, ft (`-1` = missing) |
| `Inlet_Height`, `Outlet_Height`, `Outlet_Width` | structure dimensions, ft |
| `Crossing_Structure_Length` | ft |
| `Slope_Percent` | culvert slope, % (mostly missing — optional field) |
| `Scour_Pool` | None / Small / Large |
| `Armoring` | None / Not Extensive / Extensive |
| `Barrier_Severity`, `Barrier_Name` | physical-barrier severity + type(s) |
| `Internal_Structure` | None / Baffles-Weirs / Supports / Other |
| `Structure_Substrate_Matches_Stream` | None / Not Appropriate / Contrasting / Comparable |
| `Substrate_Continuous` | substrate coverage: None / 25% / 50% / 75% / 100% |
| `Water_Depth_Matches_Stream`, `Water_Velocity` | Yes / No-Shallower / No-Deeper / No-Faster / No-Slower / Dry |

**Missing-value conventions:** empty string, `-1` / `-1.0` (numeric fields),
`"No data"`, `"Unknown"`.

## Modelling dataset

**File:** `data/processed/modeling_dataset.parquet` (28,141 rows).
Built by `src/data.py`. One row = one **unique surveyed crossing** (most recent
survey; multi-structure rows collapsed to the limiting structure). Cleaning steps
and counts: `data/interim/cleaning_report.md` / `reports/01_dataset_audit.md`.

| column | meaning |
|---|---|
| `crossing_id` | = `Crossing_Code` |
| `lat`, `lon` | WGS84 |
| `passability` | **target**, NAACC aquatic passability score in [0, 1] |
| `huc8_name` | HUC8 watershed (spatial CV grouping) |
| `year`, `date_observed` | most-recent survey |
| `County`, `Road_Type`, `Crossing_Type`, `Stream_Name`, `AOP`, `Evaluation` | identity / context |
| `Terrestrial_Passage_Score` | for reference |
| the `SURVEY_FIELDS` (see `src/data.py`) | field-survey variables, kept **only** for the Phase-2 score reconstruction — never used as GIS-prediction features |

## GIS predictor table (modelling subset)

**File:** `data/processed/subset_predictors.parquet`.
Built by `src/build_subset.py` for a stratified sample of inland central/western
NY HUC8 watersheds. See `reports/METHODOLOGY.md` for scope and each predictor's
source and DSL analogue.

| predictor | units | source |
|---|---|---|
| `drainage_area_km2` | km² | USGS NLDI upstream-basin polygon, area computed locally |
| `log_drainage_area` | log10 km² | derived |
| `basin_span_km` | km | bounding-box diagonal of the basin polygon (crude relief/size proxy) |
| `elev_m` | m | USGS 3DEP point elevation (EPQS) |
| `county_income` | USD | ACS 5-year median household income, county (Census Reporter API) |
| `county_pop_density` | people/km² | ACS population / county land area (TIGERweb) |
| `road_class` | category | dataset `Road_Type` (field-observed proxy; true GIS-only source is OSM) |
| `lat`, `lon` | deg | crossing location |
| `comid` | int | NHDPlus flowline id the crossing snapped to |
