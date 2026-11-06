# Data dictionary for the New York files

This file describes the New York files used by the first version of the project. For the multi-state columns, see Plunkett's `passability_predvars.txt`, `src/plunkett_features.py` and `docs/NBI_JOIN.md`.

## Raw data

The raw file is `data/raw/ny_naacc_surveys.csv`, with 44,871 rows. It was downloaded on 2026-09-10 from the New York State DEC ArcGIS service `NYS_NAACCSurveys_Projected`, layer 0 (`https://services6.arcgis.com/DZHaqZm9cxOD4CWM/arcgis/rest/services/NYS_NAACCSurveys_Projected/FeatureServer/0`), which is public and needs no login. The download used the ArcGIS REST `query` endpoint in pages of 2,000 rows, with coordinates in WGS84. Where the file came from is recorded in `data/raw/ny_naacc_PROVENANCE.json`.

The file is the New York part of the NAACC road-stream crossing database. One row is one survey of one structure at one crossing.

Missing values appear as an empty string, as `-1` or `-1.0` in numeric fields, or as the text "No data" or "Unknown".

| Column | Meaning |
|---|---|
| `Survey_Id` | Unique id of one field survey. |
| `Crossing_Code` | Id of the physical crossing, built from the packed latitude and longitude. It repeats when a crossing has several structures or was surveyed more than once. |
| `Aqua_Pass_Score` | The NAACC aquatic passability score from 0 to 1, where -1 means no score because data is missing. This is the value we predict. |
| `AOP` | Coarse class: Full AOP, Reduced AOP, No AOP, or no score. |
| `Evaluation` | Barrier class: No, Insignificant, Minor, Moderate, Significant or Severe barrier. |
| `Terrestrial_Passage_Score` | Passability for land animals, 0 to 1. Not used here. |
| `GPS_X_Coordinate`, `GPS_Y_Coordinate` | Longitude and latitude from the field GPS. |
| `Crossing_Type` | Culvert, Multiple Culvert, Bridge, Bridge Adequate, Ford, Inaccessible, No Crossing and a few others. |
| `Number_Of_Culverts` | Number of culverts, with -1 for missing. |
| `Tidal_Site` | Yes, No, No data or Unknown. |
| `Approved` | NAACC quality-control flag, 1 when approved. |
| `County`, `Municipality`, `Road`, `Stream_Name` | Location text. |
| `Road_Type` | Paved, Unpaved, Multilane road (more than 2 lanes), Driveway, Trail, Railroad or No data. |
| `NHD_HUC8_Watershed` | Name of the HUC8 watershed (55 in New York), used to group crossings for cross-validation. |
| `Date_Observed` | Survey date. |
| `Crossing_Span` | How the structure compares with the stream: Severe or Moderate constriction, Spans Only Bankfull/Active Channel, or Spans Full Channel and Banks. |
| `Inlet_Grade`, `Outlet_Grade` | At Stream Grade, Inlet Drop, Perched, Free Fall, Cascade, Clogged and others. |
| `Outlet_Drop_To_Water_Surface`, `Outlet_Drop_To_Stream_Bottom` | Drop in feet, with -1 for missing. |
| `Inlet_Openness`, `Outlet_Openness` | Openness ratio in feet, with -1 for missing. |
| `Inlet_Height`, `Outlet_Height`, `Outlet_Width` | Structure dimensions in feet. |
| `Crossing_Structure_Length` | Length in feet. |
| `Slope_Percent` | Culvert slope in percent. Mostly missing because it is optional. |
| `Scour_Pool` | None, Small or Large. |
| `Armoring` | None, Not Extensive or Extensive. |
| `Barrier_Severity`, `Barrier_Name` | Severity and type of any physical barrier. |
| `Internal_Structure` | None, Baffles-Weirs, Supports or Other. |
| `Structure_Substrate_Matches_Stream` | None, Not Appropriate, Contrasting or Comparable. |
| `Substrate_Continuous` | How much of the structure has substrate: None, 25%, 50%, 75% or 100%. |
| `Water_Depth_Matches_Stream`, `Water_Velocity` | Yes, No-Shallower, No-Deeper, No-Faster, No-Slower or Dry. |

## Modelling dataset

The file `data/processed/modeling_dataset.parquet` has 28,141 rows and is built by `src/data.py`. One row is one surveyed crossing. It uses the most recent survey, and when a crossing has several structures the row describes the limiting one. The cleaning steps and counts are in `data/interim/cleaning_report.md`.

| Column | Meaning |
|---|---|
| `crossing_id` | The same as `Crossing_Code`. |
| `lat`, `lon` | Location in WGS84. |
| `passability` | The value we predict, the NAACC aquatic passability score from 0 to 1. |
| `huc8_name` | HUC8 watershed, used to group crossings for cross-validation. |
| `year`, `date_observed` | Date of the most recent survey. |
| `County`, `Road_Type`, `Crossing_Type`, `Stream_Name`, `AOP`, `Evaluation` | Identifying and descriptive columns. |
| `Terrestrial_Passage_Score` | Kept for reference. |
| the `SURVEY_FIELDS` listed in `src/data.py` | Field survey measurements. They are kept only to rebuild the scoring formula and are never used as model inputs. |

## Map-based predictors for the modelling subset

The file `data/processed/subset_predictors.parquet` is built by `src/build_subset.py` for a sample of inland central and western New York watersheds. [../reports/METHODS.md](../reports/METHODS.md) describes how the New York version was built.

| Predictor | Units | Source |
|---|---|---|
| `drainage_area_km2` | km² | Area of the USGS NLDI upstream-basin polygon, computed locally. |
| `log_drainage_area` | log10 km² | Calculated from the previous column. |
| `basin_span_km` | km | Diagonal of the bounding box of the basin polygon, a rough measure of size and relief. |
| `elev_m` | m | USGS 3DEP point elevation. |
| `county_income` | USD | Median household income of the county, from the 5-year American Community Survey. |
| `county_pop_density` | people per km² | County population divided by county land area. |
| `road_class` | category | The survey's `Road_Type`. A road class from OpenStreetMap is used in later versions, since it does not depend on a site visit. |
| `lat`, `lon` | degrees | Crossing location. |
| `comid` | integer | Id of the NHDPlus stream line that the crossing was snapped to. |
