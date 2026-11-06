"""Choose a sample of New York watersheds and fetch the map-based predictors for their crossings.

The predictors come from public web services (see gis_predictors.py) and not from the regional raster processing that
DSL used. The sample uses all 55 HUC8 watersheds in New York, with at most PER_HUC8 crossings from each so that no single
watershed (Middle Hudson alone has 5,215 of the 28,141 crossings) dominates the folds. With PER_HUC8 = 1000 the subset has
21,111 crossings and no watershed is more than about 4.7% of it.

    python -m src.build_subset

The run can be stopped and restarted. It writes data/processed/subset_predictors.parquet.
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

    # road_class_mode="bulk" uses a local OpenStreetMap index built once from a New York extract (see gis_predictors.py)
    # instead of one rate-limited Overpass call per crossing, which would take about 16 hours for 28,000 rows at 2 s each.
    # sleep=0.1 only paces the NLDI and EPQS calls, which are still made once per row. The traffic, terrain, ecoregion,
    # structure, constriction and precipitation options are vectorised or batched in the same way (see gis_predictors.py).
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
