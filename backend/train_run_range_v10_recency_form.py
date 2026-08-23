"""CTO-flagged direction (2026-08-23), user request: "we have enough data
but we are not thinking enough about how we can use it... weightage...
economy rates." Every prior-stat feature this codebase has ever used is a
flat career-to-date cumulative average -- a player's rate from 40 matches
ago counts exactly as much as last week's. `app/ml/recency_weighted_prior_dataset.py`
adds a match-indexed EWMA (decay=0.95, ~13-14 match half-life) alongside
the existing flat features, spot-checked against Kohli's real IPL career
trajectory before trusting it (values diverge from the flat average in
directions consistent with known purple patches/slumps, not degenerate).

Adds recency features on TOP of the existing flat ones (doesn't replace
them) -- lets the model itself decide via feature importance whether
recency adds anything beyond what the flat career average and
already-live phase/venue/competition splits already capture. Same
architecture, split, and calibration as `run_range_v7_competition_prior`
(currently live) otherwise. Candidate-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from app.ml.ipl_phase_moe_dataset import build_ipl_phase_moe_features
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from app.ml.player_competition_dataset import build_player_competition_features
from app.ml.player_venue_phase_dataset import build_player_venue_phase_features
from app.ml.recency_weighted_prior_dataset import build_recency_weighted_prior_dataset
from train_contract22_rigorous import BOWLER_FEATURES, build_enriched
from train_phase_calibrated_sharp_range_v33 import (
    MAX_RUN_CLASS, best_bands, male_source_files, nll, temperature_scale,
)
from train_run_range_enriched_v3_batting_style import (
    BASE_FEATURES, MOE_EXTRA_FEATURES, VENUE_FEATURES,
    _load_batting_style_lookup, features,
)
from train_run_range_v4_batter_phase import BATTER_PHASE_FEATURES

VERSION = "run_range_v10_recency_form"
COMPETITION_PRIOR_FEATURES = [
    "batter_competition_runs_per_ball_shrunk", "batter_competition_boundary_rate_shrunk",
    "batter_competition_dismissal_rate_shrunk", "batter_competition_balls",
    "bowler_competition_economy_shrunk", "bowler_competition_wicket_rate_shrunk",
    "bowler_competition_balls",
]
RECENCY_FEATURES = [
    "striker_recency_balls", "striker_recency_runs_per_ball", "striker_recency_dot_rate",
    "striker_recency_boundary_rate", "striker_recency_dismissal_rate",
    "partner_recency_runs_per_ball",
    "bowler_recency_balls", "bowler_recency_economy", "bowler_recency_wicket_rate",
]
FEATURES = (
    BASE_FEATURES + MOE_EXTRA_FEATURES + VENUE_FEATURES + BOWLER_FEATURES
    + BATTER_PHASE_FEATURES + COMPETITION_PRIOR_FEATURES + RECENCY_FEATURES
)
CATEGORICAL = [
    "phase", "active_batter_state", "wickets_remaining_bucket",
    "state_regime", "chase_pressure", "venue_scoring_regime",
    "venue_par_source", "batting_team_venue_context", "phase_venue_regime",
    "striker_batting_style", "bowler_type",
]
SHRINKAGE_BALLS = 150

root = Path(__file__).resolve().parents[1]
output = root / "models/candidates" / VERSION
output.mkdir(parents=True, exist_ok=True)

ipl_files = {p.name for p in (root / "data/raw/cricsheet/ipl").glob("*.json")}

base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
eligible = male_source_files(root)
base = base[base["source_file"].isin(eligible)].copy()
base["match_date"] = pd.to_datetime(base["match_date"])

print("Building enriched MOE + venue features (scopes=ipl+t20i)...", flush=True)
moe = build_ipl_phase_moe_features(root, canonical_identities=True, scopes=("ipl", "t20i"))
venue = build_ipl_venue_regime_dataset(root, scopes=("ipl", "t20i"))
print("Building bowler features...", flush=True)
enriched = build_enriched(root, eligible)
print("Building batter phase features...", flush=True)
batter_phase = build_player_venue_phase_features(root, scopes=("ipl", "t20i"))
print("Building competition-specific player features...", flush=True)
competition = build_player_competition_features(root, scopes=("ipl", "t20i"))
print("Building recency-weighted form features (the new signal being tested)...", flush=True)
recency = build_recency_weighted_prior_dataset(root, scopes=("ipl", "t20i"))

keys = ["source_file", "innings", "over"]
moe_columns = [c for c in moe.columns if c not in ("match_date",) and c not in keys]
venue_columns = [c for c in venue.columns if c not in ("match_date",) and c not in keys]
data = base.merge(moe[keys + moe_columns], on=keys, how="inner", validate="one_to_one")
data = data.merge(venue[keys + venue_columns], on=keys, how="inner", validate="one_to_one")
data = data.merge(enriched[keys + BOWLER_FEATURES], on=keys, how="inner", validate="one_to_one")
data = data.merge(
    batter_phase[keys + BATTER_PHASE_FEATURES], on=keys, how="inner", validate="one_to_one"
)
comp_cols = [
    "batter_competition_balls", "batter_competition_runs_per_ball",
    "batter_competition_boundary_rate", "batter_competition_dismissal_rate",
    "bowler_competition_balls", "bowler_competition_economy", "bowler_competition_wicket_rate",
]
data = data.merge(competition[keys + comp_cols], on=keys, how="inner", validate="one_to_one")
data = data.merge(recency[keys + RECENCY_FEATURES], on=keys, how="inner", validate="one_to_one")
print(f"Merged rows: {len(data)} (base was {len(base)})")

style_lookup = _load_batting_style_lookup(root)
data["striker_batting_style"] = data["striker"].map(style_lookup)
data["is_ipl"] = data["source_file"].isin(ipl_files)

w = data["batter_competition_balls"] / (data["batter_competition_balls"] + SHRINKAGE_BALLS)
data["batter_competition_runs_per_ball_shrunk"] = w * data["batter_competition_runs_per_ball"] + (1 - w) * data["striker_prior_runs_per_ball"]
data["batter_competition_boundary_rate_shrunk"] = w * data["batter_competition_boundary_rate"] + (1 - w) * data["striker_prior_boundary_rate"]
data["batter_competition_dismissal_rate_shrunk"] = w * data["batter_competition_dismissal_rate"] + (1 - w) * data["striker_prior_dismissal_rate"]

wb = data["bowler_competition_balls"] / (data["bowler_competition_balls"] + SHRINKAGE_BALLS)
bowler_pooled_economy = data["bowl_hist_avg_runs_conceded"] / 6.0
data["bowler_competition_economy_shrunk"] = wb * data["bowler_competition_economy"] + (1 - wb) * bowler_pooled_economy
data["bowler_competition_wicket_rate_shrunk"] = wb * data["bowler_competition_wicket_rate"] + (1 - wb) * data["bowl_hist_wicket_rate"]

train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)

model = lgb.LGBMClassifier(
    objective="multiclass", num_class=MAX_RUN_CLASS + 1,
    n_estimators=60, learning_rate=0.08, num_leaves=25, max_depth=6,
    min_child_samples=100, subsample=0.85, colsample_bytree=0.9,
    reg_lambda=1.0, random_state=42, verbose=-1,
)
train_actual = np.minimum(train["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
model.fit(features(train, FEATURES, CATEGORICAL), train_actual, categorical_feature=CATEGORICAL)

calibration_raw = model.predict_proba(features(calibration, FEATURES, CATEGORICAL))
holdout_raw = model.predict_proba(features(holdout, FEATURES, CATEGORICAL))
calibration_actual = np.minimum(calibration["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
holdout_actual = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
holdout_is_ipl = holdout["is_ipl"].to_numpy()

temperatures = {}
calibration_scaled = calibration_raw.copy()
holdout_scaled = holdout_raw.copy()
for phase in ("powerplay", "middle", "death"):
    calibration_mask = calibration["phase"].astype(str).to_numpy() == phase
    holdout_mask = holdout["phase"].astype(str).to_numpy() == phase
    result = minimize_scalar(
        lambda value: nll(
            temperature_scale(calibration_raw[calibration_mask], value),
            calibration_actual[calibration_mask],
        ),
        bounds=(0.5, 3.0), method="bounded",
    )
    temperatures[phase] = float(result.x)
    calibration_scaled[calibration_mask] = temperature_scale(calibration_raw[calibration_mask], temperatures[phase])
    holdout_scaled[holdout_mask] = temperature_scale(holdout_raw[holdout_mask], temperatures[phase])

calibration_low, calibration_high = best_bands(calibration_scaled)
holdout_low, holdout_high = best_bands(holdout_scaled)
holdout_hit = (holdout_actual >= holdout_low) & (holdout_actual <= holdout_high)
raw_calibration_nll = nll(calibration_raw, calibration_actual)
scaled_calibration_nll = nll(calibration_scaled, calibration_actual)

# Reference: currently-live run_range_v7_competition_prior (width=2 holdout figures).
v7_blended, v7_ipl, v7_t20i = 0.2875, 0.2500, 0.2943
new_blended = float(np.mean(holdout_hit))
new_ipl = float(np.mean(holdout_hit[holdout_is_ipl]))
new_t20i = float(np.mean(holdout_hit[~holdout_is_ipl]))
beats_v7 = bool(new_blended > v7_blended and scaled_calibration_nll <= raw_calibration_nll)
beats_v7_on_ipl_specifically = bool(new_ipl > v7_ipl)

importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
importance_rank = {name: rank + 1 for rank, (name, _) in enumerate(importances)}

report = {
    "candidate_version": VERSION,
    "candidate_only": True,
    "production_changed": False,
    "run_model_changed": False,
    "note": (
        "run_range_v7_competition_prior + recency-weighted (match-EWMA, "
        "decay=0.95) batter/bowler form, added alongside the existing "
        "flat career-average features. Tests against the real current "
        "live architecture."
    ),
    "split": {"train_rows": len(train), "holdout_rows": len(holdout)},
    "reference_run_range_v7_competition_prior": {
        "blended": v7_blended, "ipl": v7_ipl, "t20i": v7_t20i,
    },
    "holdout": {
        "blended_hit_rate": new_blended,
        "ipl_hit_rate": new_ipl,
        "t20i_hit_rate": new_t20i,
    },
    "beats_v7_blended_and_calibration_gate": beats_v7,
    "beats_v7_on_ipl_specifically": beats_v7_on_ipl_specifically,
    "recency_feature_importance_rank": {
        f: importance_rank[f] for f in RECENCY_FEATURES
    },
    "total_features": len(FEATURES),
    "decision": "promote_candidate" if beats_v7 else "reject_keep_research",
}
joblib.dump(model, output / "sharp_range_model.pkl")
joblib.dump(temperatures, output / "phase_temperatures.pkl")
joblib.dump(FEATURES, output / "feature_cols.pkl")
joblib.dump(CATEGORICAL, output / "categorical_cols.pkl")
(output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
