"""Link every crossing to its nearest NHDPlusV2 stream line. It needs data/interim/nhd_flowlines.jsonl from scripts/fetch_flowlines.py.

    python -m scripts.make_links
"""
import pandas as pd

from src import streamcat_feats as sf

if __name__ == "__main__":
    d = pd.read_parquet("data/processed/plunkett_features_osm.parquet", columns=["crossing_survey_id", "lon_nad83", "lat_nad83", "state"])
    links = sf.link_crossings(d)
    links["comid"] = links["comid"].fillna(0).astype("int64")
    links[["crossing_survey_id", "comid", "snap_m", "snap2_m", "order_line", "ftype_line"]].to_csv(sf.COMIDS, index=False)
    ok = links.comid > 0
    print(f"{ok.mean():.3f} of {len(d)} crossings have a stream line within 500 m; "
          f"within 50 m {(links.snap_m <= 50).mean():.3f}; within 30 m {(links.snap_m <= 30).mean():.3f}")
    print(links.assign(state=d["state"].values).groupby("state").apply(lambda g: (g.snap_m <= 50).mean(), include_groups=False).round(2).to_dict())
