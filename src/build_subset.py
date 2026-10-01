"""Select a spatial modelling subset and fetch its GIS predictors.

Scope note (see reports/METHODOLOGY.md): reproducing the full DSL / Plunkett GIS
feature set requires regional raster processing that is out of scope here. This
fetches a reduced predictor set from public web services (USGS NLDI drainage
basin, USGS 3DEP elevation, US Census county income + population density, road
class from OpenStreetMap, EPA StreamCat upstream-watershed metrics, NHDPlusV2
stream slope/order) for a sample of NY HUC8 watersheds.

Run: ``python -m src.build_subset``  (resumable; writes
``data/processed/subset_predictors.parquet``)

**2026-09-22: scaled to all 55 HUC8 watersheds** (previously a hand-picked 16,
excluding coastal/Great-Lakes-mouth ones over NLDI snap-error concerns). Two
things changed that make the full set now the better default:
1. The StreamCat watershed-area backfill (gis_predictors.py::
   fetch_streamcat_batch) now recovers rows where NLDI's basin-polygon fetch
   snaps badly, instead of just losing them - the concrete failure mode the
   coastal exclusion was hedging against.
2. Every prior scaling step (7->16 watersheds) improved the spatial-CV result;
   there was no sign of diminishing returns yet, so leaving 39 more
   watersheds on the table was the more likely-costly choice.
``PER_HUC8`` still caps the largest watersheds (Middle Hudson alone has 5,215
of the dataset's 28,141 crossings) so no single one dominates every spatial
fold; smaller watersheds just contribute all they have.

**2026-09-23: raised ``PER_HUC8`` 300 -> 1000** (11,489 -> 21,111 crossings;
computed directly from the full dataset's per-watershed counts, not a guess).
Uncapping entirely would let Middle Hudson alone be ~18.5% of the whole
dataset; 1000 keeps any single watershed under ~4.7% while still letting the
~30 watersheds with >300 crossings contribute meaningfully more than before.
Also added two new predictors (``do_aadt``, ``do_terrain``) - see
``gis_predictors.py`` module docstring for the NYSDOT AADT traffic layer and
USGS 3DEP bulk multi-scale terrain sampling, both added the same day to fill
the two predictor gaps this project's docs have listed as omitted since the
start.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import build_modeling_dataset
from .gis_predictors import build_predictor_table

PER_HUC8 = 1000         # per-watershed cap; smaller watersheds contribute all they have
RANDOM_SEED = 20260910


def select_subset(per_huc8: int = PER_HUC8) -> pd.DataFrame:
    """Deterministic prefix sample over every HUC8 watershed in the cleaned
    dataset: each watershed is shuffled once with a fixed seed, then the
    first ``per_huc8`` rows are taken (or all of it, if smaller). Reducing
    ``per_huc8`` yields a strict subset, so the disk cache is always reused."""
    m, _ = build_modeling_dataset()
    parts = []
    for huc, g in m.groupby("huc8_name"):
        g = g.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        parts.append(g.head(per_huc8))
    return pd.concat(parts).reset_index(drop=True)


def main(limit: int | None = None) -> None:
    sub = select_subset()
    if limit:
        sub = sub.head(limit)
    print(f"subset: {len(sub)} crossings across {sub['huc8_name'].nunique()} HUC8 watersheds")
    print(sub.groupby("huc8_name").size().to_string())

    # road_class_mode="bulk": a local OSM spatial index (built once from a NY
    # extract, see gis_predictors.py) instead of one rate-limited Overpass
    # call per crossing - infeasible at this scale (28k rows * ~2s/row would
    # be ~16h). sleep=0.1 only paces the still-per-row NLDI/EPQS calls.
    # do_aadt/do_terrain (2026-09-23): NYSDOT traffic + USGS 3DEP bulk terrain
    # sampling, both vectorized/batched like road_class_mode="bulk" - no new
    # per-row rate limit added.
    # do_ecoregion/do_structures (2026-09-24): EPA ecoregion (regional-
    # heterogeneity control) + NYSDOT bridge/large-culvert inventory (actual
    # structure data) - see gis_predictors.py sections 9-10 for the
    # diagnostic reasoning behind each. Both also vectorized/batched.
    # do_constriction (2026-09-26): structure opening width vs. USGS
    # regionalized bankfull-width curves - see gis_predictors.py section 11.
    # do_precip (2026-09-29): NOAA Atlas 14 storm-intensity grids (distinct
    # from StreamCat's mean-annual precip) - see gis_predictors.py section 12.
    pred = build_predictor_table(sub, do_drainage=True, do_elev=True, do_road_class=True,
                                 road_class_mode="bulk",
                                 do_streamcat=True, do_vaa=True,
                                 do_aadt=True, do_terrain=True,
                                 do_ecoregion=True, do_structures=True,
                                 do_constriction=True, do_precip=True,
                                 sleep=0.1, log_every=200)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    pred.to_parquet("data/processed/subset_predictors.parquet", index=False)
    pred.to_csv("data/processed/subset_predictors.csv", index=False)
    ok = pred["drainage_area_km2"].notna().mean() if "drainage_area_km2" in pred else 0
    el = pred["elev_m"].notna().mean() if "elev_m" in pred else 0
    rc = (pred["road_class"] != "unknown").mean() if "road_class" in pred else 0
    sc = pred["sc_precip_mm"].notna().mean() if "sc_precip_mm" in pred else 0
    vaa = pred["stream_slope"].notna().mean() if "stream_slope" in pred else 0
    aadt = pred["aadt"].notna().mean() if "aadt" in pred else 0
    terr = pred["terrain_slope_100m"].notna().mean() if "terrain_slope_100m" in pred else 0
    eco = (pred["ecoregion"] != "unknown").mean() if "ecoregion" in pred else 0
    struct = (pred["structure_is_bridge"] != "none").mean() if "structure_is_bridge" in pred else 0
    constr = pred["constriction_ratio"].notna().mean() if "constriction_ratio" in pred else 0
    precip = pred["precip_24h_10yr_in"].notna().mean() if "precip_24h_10yr_in" in pred else 0
    print(f"\ndone: drainage_area present {ok:.1%}, elevation present {el:.1%}, "
         f"road_class matched (OSM) {rc:.1%}, StreamCat present {sc:.1%}, "
         f"stream_slope present {vaa:.1%}, AADT matched {aadt:.1%}, "
         f"terrain matched {terr:.1%}, ecoregion matched {eco:.1%}, "
         f"structure matched {struct:.1%}, constriction ratio {constr:.1%}, "
         f"precip intensity matched {precip:.1%}")
    print("wrote data/processed/subset_predictors.parquet")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(lim)
