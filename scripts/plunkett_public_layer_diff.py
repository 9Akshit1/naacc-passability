"""Where do Plunkett's scores differ from the public NAACC layer for the same survey?

The public layer (scripts/download_naacc_public_layer.py) has one row per structure, so a survey with several
structures appears several times; its rows agree on the score, so they are reduced to one per survey id first.

    python -m scripts.download_naacc_public_layer      # first, if data/raw/naacc_public_layer.parquet is missing
    python -m scripts.plunkett_public_layer_diff
"""
import json

import numpy as np
import pandas as pd

from src import plunkett_data as pdm

pub = pd.read_parquet("data/raw/naacc_public_layer.parquet")
pl = pd.read_parquet(pdm.FTP_DIR / "passability_data.parquet")

pub["sid"] = pd.to_numeric(pub.Survey_Id, errors="coerce")
pub["score_pub"] = pd.to_numeric(pub.Aqua_Pass_, errors="coerce")
by_survey = pub.groupby("sid").agg(lo=("score_pub", "min"), hi=("score_pub", "max"), n_rows=("score_pub", "size"),
                                   last_edit=("Date_Last_", "max"), date_pub=("Date_Obser", "first")).reset_index()
m = pl.merge(by_survey, left_on="crossing_survey_id", right_on="sid")
m = m[m.lo >= 0].copy()                       # -1 means "not scored" in the public layer
m["diff"] = (m.aquatic_score - m.hi).abs()
bad = m[m["diff"] > 0.05].copy()
year = pd.to_datetime(m.date_observed).dt.year

out = {
    "public_rows_that_disagree_within_a_survey": int((by_survey.hi - by_survey.lo > 0.001).sum()),
    "surveys_in_both_with_a_public_score": int(len(m)),
    "identical_within_0.001": float((m["diff"] < 0.001).mean()),
    "differ_at_least_0.001": int((m["diff"] >= 0.001).sum()),
    "differ_more_than_0.05": int(len(bad)),
    "differ_more_than_0.2": int((m["diff"] > 0.2).sum()),
    "differing_surveys_with_the_same_observation_date": int((pd.to_datetime(bad.date_observed).dt.normalize()
                                                             == pd.to_datetime(bad.date_pub).dt.normalize()).sum()),
    "plunkett_lower_than_public": int((bad.aquatic_score < bad.hi).sum()),
    "survey_year_of_differing": pd.to_datetime(bad.date_observed).dt.year.value_counts().sort_index().to_dict(),
    "share_of_all_shared_surveys_from_2015": float((year == 2015).mean()),
    "share_of_differing_surveys_from_2015": float((pd.to_datetime(bad.date_observed).dt.year == 2015).mean()),
    "crossing_type_share_differing": bad.crossing_type.value_counts(normalize=True).round(3).to_dict(),
    "crossing_type_share_all_shared": m.crossing_type.value_counts(normalize=True).round(3).to_dict(),
}
out["survey_year_of_differing"] = {int(k): int(v) for k, v in out["survey_year_of_differing"].items()}
json.dump(out, open("results/plunkett/public_layer_diff.json", "w"), indent=1)
print(json.dumps(out, indent=1))
