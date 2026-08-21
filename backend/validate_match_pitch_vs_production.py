"""Validate the v3.10 "boundary_match_pitch" candidate's range-hit-rate
against the ACTUAL current production PredictionEngine (not the old v3.2/
v3.9 internal candidate baselines it was originally compared to -- same
"wrong baseline" issue found and corrected for the bowler-spell candidate).

Reuses the candidate's own saved artifacts (boundary_submodel.pkl,
boundary_match_pitch_model.pkl) and precomputed feature CSVs -- no
retraining. Then replays the exact same holdout matches (by source_file)
through PredictionEngine to get a genuinely comparable range-hit-rate on
identical overs.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_validate_candidate_v3 import CATEGORICAL_FEATURES, V3_FEATURES, _features
from train_sharp_range_candidate import MAX_RUN_CLASS, _best_bands

KEYS = ["source_file", "match_date", "innings", "over"]
CANDIDATE_DIR = PROJECT_ROOT / "models" / "candidates" / "v3.10_current_match_pitch"


def _feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = _features(frame, columns)
    if "venue_track_type" in result:
        result["venue_track_type"] = result["venue_track_type"].astype("category")
    return result


print("Loading and merging candidate feature CSVs...")
base = pd.read_csv(PROJECT_ROOT / "data/candidates/v3/verified_training_overs.csv")
venue = pd.read_csv(PROJECT_ROOT / "data/candidates/v3.7/venue_track_features.csv")
pitch = pd.read_csv(PROJECT_ROOT / "data/candidates/v3.10/current_match_pitch_features.csv")
components = pd.read_csv(PROJECT_ROOT / "data/candidates/v3.4/over_components.csv")
for frame in (base, venue, pitch, components):
    frame["match_date"] = frame["match_date"].astype(str)
components["boundary_count"] = components["fours"] + components["sixes"]
data = (
    base.merge(venue, on=KEYS, validate="one_to_one")
    .merge(pitch, on=KEYS, validate="one_to_one")
    .merge(components[KEYS + ["boundary_count"]], on=KEYS, validate="one_to_one")
)
for name in ("runs", "boundaries", "dots", "wickets"):
    venue_rate = data[f"venue_track_venue_{name}_rate"]
    data[f"match_pitch_delta_match_{name}"] = data[f"match_pitch_match_{name}_rate"] - venue_rate
    data[f"match_pitch_delta_first_innings_{name}"] = (
        data[f"match_pitch_first_innings_{name}_rate"] - venue_rate
    )
data["match_date"] = pd.to_datetime(data["match_date"])
holdout = data[data["match_date"].dt.year >= 2025].reset_index(drop=True)
print(f"Holdout rows: {len(holdout)}, matches: {holdout['source_file'].nunique()}")

print("Applying saved boundary submodel...")
boundary_model = joblib.load(CANDIDATE_DIR / "boundary_submodel.pkl")
boundary_input_columns = V3_FEATURES + [
    c for c in venue.columns if c.startswith("venue_track_")
]
boundary_proba = boundary_model.predict_proba(
    _feature_frame(holdout, boundary_input_columns)
)
holdout["boundary_gate_none_probability"] = boundary_proba[:, 0]
holdout["boundary_gate_one_probability"] = boundary_proba[:, 1]
holdout["boundary_gate_multiple_probability"] = boundary_proba[:, 2]
holdout["boundary_gate_entropy"] = -np.sum(
    boundary_proba * np.log(np.maximum(boundary_proba, 1e-9)), axis=1
)

print("Applying saved boundary_match_pitch model (the original winner)...")
model = joblib.load(CANDIDATE_DIR / "boundary_match_pitch_model.pkl")
feature_cols = joblib.load(CANDIDATE_DIR / "boundary_match_pitch_feature_cols.pkl")
proba = model.predict_proba(_feature_frame(holdout, feature_cols))
low, high, mass = _best_bands(proba, 2)
actual = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
hit = (actual >= low) & (actual <= high)
print(f"\nCandidate (v3.10 boundary_match_pitch) 2-run-band hit rate: {hit.mean()*100:.2f}%")

# Save per-row results for the production comparison
holdout_result = holdout[KEYS + ["runs_in_over"]].copy()
holdout_result["candidate_hit"] = hit
holdout_result["candidate_low"] = low
holdout_result["candidate_high"] = high
holdout_result.to_csv(
    PROJECT_ROOT / "backend" / "match_pitch_holdout_result.csv", index=False
)
print("Saved per-row results to match_pitch_holdout_result.csv for the production comparison.")
