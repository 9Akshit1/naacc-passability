"""Build data/processed/plunkett_features_osm.parquet (Plunkett + NBI + OSM) once ALL 13 OSM state files exist.
Refuses to run with a missing state, so a partially fetched OSM layer can never be mistaken for 'no structure'."""
import sys
from pathlib import Path
from src import plunkett_features as pf, osm_structures as osm
STATES = ["CT", "DE", "MA", "MD", "ME", "NH", "NJ", "NY", "PA", "RI", "VA", "VT", "WV"]
def _have(s):
    j = osm.OSM_DIR / f"{s}_structures.json"
    pbf = osm.PBF_DIR / {"PA": "pennsylvania", "RI": "rhode-island"}.get(s, s.lower())
    return (j.exists() and j.stat().st_size > 1000) or pbf.with_suffix(".osm.pbf").exists()
missing = [s for s in STATES if not _have(s)]
if missing:
    sys.exit(f"OSM files missing for {missing}; run: python -m src.osm_structures {' '.join(missing)}")
d = pf.build(use_osm=True)
d.to_parquet("data/processed/plunkett_features_osm.parquet")
print(d.shape, "OSM columns:", len(pf.feature_groups(d)["OSM"]))
