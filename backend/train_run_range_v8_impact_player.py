"""CTO-agreed next priority (2026-08-23): IPL Impact Player state as a real
feature. `EngineRouter`'s IPL `EngineSpecification` has declared
`extension_features=("impact_player_state", ...)` since it was written, but
it was never implemented -- pure placeholder strings, nothing computed,
nothing fed to a model.

`app/ml/ipl_impact_dataset.py` already existed, correctly parses Cricsheet's
real `replacements.match[].reason == "impact_player"` events (leakage-safe:
state as of the start of each over, using only replacements applied before
that over began), and was run once (`data/candidates/ipl_v1/
impact_player_features.csv` exists, dated Jul 22). A prior training script
(`train_ipl_engine_v1.py`) tested it but against a stale architecture: old
V3 feature set, fixed band width=2, and a training/holdout split with no
temperature calibration -- not comparable to what's actually live now
(`run_range_v7_competition_prior`, band width=3 for IPL in production). That
prior result (0.09pp improvement, p=0.87, gates failed) is not evidence
either way for the current architecture.

This candidate merges the same impact-player features onto the *actual
live* v7 architecture (same base features, same calibration, same holdout
split) so the comparison is against the real current baseline, per standing
practice. Impact features only exist for IPL matches (T20I rows get neutral
defaults: rule_era=0, used=0, available=0, role="none", overs_since=-1,
which is also the correct pre-2023/no-substitution-yet default within IPL
itself).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from app.ml.ipl_impact_dataset import build_ipl_impact_dataset
from app.ml.ipl_phase_moe_dataset import build_ipl_phase_moe_features
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from app.ml.player_competition_dataset import build_player_competition_features
from app.ml.player_venue_phase_dataset import build_player_venue_phase_features
from train_contract22_rigorous import BOWLER_FEATURES, build_enriched
from train_phase_calibrated_sharp_range_v33 import (
    MAX_RUN_CLASS, best_bands, male_source_files, nll, temperature_scale,
)
from train_run_range_enriched_v3_batting_style import (
    BASE_FEATURES, MOE_EXTRA_FEATURES, VENUE_FEATURES,
    _load_batting_style_lookup, features,
)
from train_run_range_v4_batter_phase import BATTER_PHASE_FEATURES

VERSION = "run_range_v8_impact_player"
COMPETITION_PRIOR_FEATURES = [
    "batter_competition_runs_per_ball_shrunk", "batter_competition_boundary_rate_shrunk",
    "batter_competition_dismissal_rate_shrunk", "batter_competition_balls",
    "bowler_competition_economy_shrunk", "bowler_competition_wicket_rate_shrunk",
    "bowler_competition_balls",
]
IMPACT_FEATURES = [
    "ipl_impact_rule_era",
    "ipl_impact_batting_used", "ipl_impact_bowling_used",
    "ipl_impact_batting_available", "ipl_impact_bowling_available",
    "ipl_impact_batting_role", "ipl_impact_bowling_role",
    "ipl_impact_batting_overs_since", "ipl_impact_bowling_overs_since",
]
FEATURES = (
    BASE_FEATURES + MOE_EXTRA_FEATURES + VENUE_FEATURES + BOWLER_FEATURES
    + BATTER_PHASE_FEATURES + COMPETITION_PRIOR_FEATURES + IMPACT_FEATURES
)
CATEGORICAL = [
    "phase", "active_batter_state", "wickets_remaining_bucket",
    "state_regime", "chase_pressure", "venue_scoring_regime",
    "venue_par_source", "batting_team_venue_context", "phase_venue_regime",
    "striker_batting_style", "bowler_type",
    "ipl_impact_batting_role", "ipl_impact_bowling_role",
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
print("Building IPL impact-player features (the new signal being tested)...", flush=True)
impact = build_ipl_impact_dataset(root)

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
data = data.merge(impact[keys + IMPACT_FEATURES], on=keys, how="left", validate="one_to_one")
print(f"Merged rows: {len(data)} (base was {len(base)})")

impact_defaults = {
    "ipl_impact_rule_era": 0, "ipl_impact_batting_used": 0, "ipl_impact_bowling_used": 0,
    "ipl_impact_batting_available": 0, "ipl_impact_bowling_available": 0,
    "ipl_impact_batting_role": "none", "ipl_impact_bowling_role": "none",
    "ipl_impact_batting_overs_since": -1, "ipl_impact_bowling_overs_since": -1,
}
for column, default in impact_defaults.items():
    data[column] = data[column].fillna(default)
for column in (
    "ipl_impact_rule_era", "ipl_impact_batting_used", "ipl_impact_bowling_used",
    "ipl_impact_batting_available", "ipl_impact_bowling_available",
    "ipl_impact_batting_overs_since", "ipl_impact_bowling_overs_since",
):
    data[column] = data[column].astype(int)

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

# References: currently-live run_range_v7_competition_prior (width=2 holdout figures,
# before production's IPL-specific width=3 band widening).
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
        "run_range_v7_competition_prior + IPL impact-player state "
        "(used/available/role/overs-since for both batting and bowling "
        "sides, leakage-safe as-of-start-of-over). Tests against the real "
        "current live architecture, not the stale train_ipl_engine_v1.py "
        "comparison from before v7 existed."
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
    "impact_feature_importance_rank": {
        f: importance_rank[f] for f in IMPACT_FEATURES
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
