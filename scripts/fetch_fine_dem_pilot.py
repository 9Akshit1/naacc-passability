"""Pilot: fine-scale DEM samples for 6,000 random NY development crossings that also have survey fields."""
import numpy as np, pandas as pd
from src import fine_dem, plunkett_cv as cv
d = pd.read_parquet("data/processed/plunkett_features_osm.parquet"); lock = cv.locked_test_mask(d)
old = pd.read_parquet("data/processed/subset_predictors.parquet", columns=["Crossing_Code"])
x = d.loc[~lock & d.crossing_code.isin(old.Crossing_Code)]
x = x.sample(6000, random_state=0)
z = fine_dem.fetch(x)
np.save("data/processed/fine_dem_pilot_z.npy", z); x[["crossing_code"]].to_csv("data/processed/fine_dem_pilot_codes.csv", index=False)
print("done", z.shape, "share rows with centre elevation:", float(np.isfinite(z[:, 0]).mean()))
