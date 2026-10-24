"""Tests whether a faster-reacting momentum window improves real run-range
accuracy -- docs/live_field_test_2026-09-15.md's "run-range momentum lag"
finding: a real match showed the model predicting a low run-range through
two consecutive massive overs (20, then 27) before finally reacting on the
third. Diagnosed 2026-09-18 (diagnose_momentum_lag.py, against a real 2026
match with the same pattern): recent_runs_per_ball correctly reaches an
extreme value (3.58) once both massive overs are inside its 12-ball/
2-over window, but the live model's own predicted_runs barely moves
(7.00 -> 8.00) even then. Feature-importance check on the real live model
(models/candidates/run_range_v11_partnership_rate/sharp_range_model.pkl)
confirmed why: recent_runs_per_ball ranks 28th/63 (1.9%), and all six
recent_*/state_regime momentum features combined total under 6% -- the
model structurally under-weights in-the-moment momentum relative to
career/context priors.

This candidate is IDENTICAL to the live run_range_v11_partnership_rate in
every respect (same features, same architecture, same split, same
calibration, same IPL/T20I band-width policy) except recent_*/phase_*_
pressure/phase_rotation_value are computed over a 6-ball (1-over) window
instead of 12 (2 overs) -- see app/ml/candidate_v3_dataset.py's
recent_window parameter and build_verified_dataset_recent6.py. If a
shorter window measurably improves real holdout accuracy, that's honest
evidence the lag is a real, fixable problem, not a correctly-calibrated
dampening of noisy short-term variance -- and if it doesn't help, that's
equally real evidence the model's low weighting is already appropriate.

Reference is run_range_v11_partnership_rate's own CURRENT real numbers
(30.04%/33.25%/29.45% blended/IPL/T20I, post the 2026-08-26 IPL
band-width fix -- docs/finding_run_range_nn_gbm_ensemble_real_win.md),
not a stale v7 comparison, per standing practice (validate against the
actual production baseline).
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
from app.ml.partnership_dataset import build_partnership_dataset
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

VERSION = "run_range_v25_faster_momentum"
COMPETITION_PRIOR_FEATURES = [
    "batter_competition_runs_per_ball_shrunk", "batter_competition_boundary_rate_shrunk",
    "batter_competition_dismissal_rate_shrunk", "batter_competition_balls",
    "bowler_competition_economy_shrunk", "bowler_competition_wicket_rate_shrunk",
    "bowler_competition_balls",
]
PARTNERSHIP_FEATURES = ["partnership_runs", "partnership_balls", "partnership_run_rate"]
FEATURES = (
    BASE_FEATURES + MOE_EXTRA_FEATURES + VENUE_FEATURES + BOWLER_FEATURES
    + BATTER_PHASE_FEATURES + COMPETITION_PRIOR_FEATURES + PARTNERSHIP_FEATURES
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

# Only difference from train_run_range_v11_partnership_rate.py: the recent6
# variant of the base CSV (6-ball window instead of 12 for recent_*/
# phase_*_pressure/phase_rotation_value).
base = pd.read_csv(root / "data/candidates/v3/verified_training_overs_recent6.csv")
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
print("Building partnership rate features...", flush=True)
partnership = build_partnership_dataset(root, scopes=("ipl", "t20i"))

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
data = data.merge(partnership[keys + PARTNERSHIP_FEATURES], on=keys, how="inner", validate="one_to_one")
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

holdout_low = np.empty(len(holdout), dtype=int)
holdout_high = np.empty(len(holdout), dtype=int)
holdout_low[holdout_is_ipl], holdout_high[holdout_is_ipl] = best_bands(holdout_scaled[holdout_is_ipl], width=3)
holdout_low[~holdout_is_ipl], holdout_high[~holdout_is_ipl] = best_bands(holdout_scaled[~holdout_is_ipl], width=2)
holdout_hit = (holdout_actual >= holdout_low) & (holdout_actual <= holdout_high)
raw_calibration_nll = nll(calibration_raw, calibration_actual)
scaled_calibration_nll = nll(calibration_scaled, calibration_actual)

# The real current live baseline (run_range_v11_partnership_rate), not a
# stale v7 comparison -- see this file's module docstring.
live_blended, live_ipl, live_t20i = 0.3004, 0.3325, 0.2945
new_blended = float(np.mean(holdout_hit))
new_ipl = float(np.mean(holdout_hit[holdout_is_ipl]))
new_t20i = float(np.mean(holdout_hit[~holdout_is_ipl]))
beats_live = bool(new_blended > live_blended and scaled_calibration_nll <= raw_calibration_nll)
beats_live_on_ipl_specifically = bool(new_ipl > live_ipl)
beats_live_on_t20i_specifically = bool(new_t20i > live_t20i)

importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
importance_rank = {name: rank + 1 for rank, (name, _) in enumerate(importances)}
recent_features = [
    "recent_legal_balls", "recent_runs_per_ball", "recent_dot_rate",
    "recent_single_rate", "recent_boundary_rate", "recent_wicket_rate",
    "state_regime",
]

report = {
    "candidate_version": VERSION,
    "candidate_only": True,
    "production_changed": False,
    "run_model_changed": False,
    "note": (
        "Identical to run_range_v11_partnership_rate except recent_*/"
        "phase_*_pressure/phase_rotation_value use a 6-ball (1-over) "
        "window instead of 12 (2 overs) -- tests docs/"
        "live_field_test_2026-09-15.md's 'run-range momentum lag' finding."
    ),
    "split": {"train_rows": len(train), "holdout_rows": len(holdout)},
    "reference_run_range_v11_partnership_rate_live": {
        "blended": live_blended, "ipl": live_ipl, "t20i": live_t20i,
    },
    "holdout": {
        "blended_hit_rate": new_blended,
        "ipl_hit_rate": new_ipl,
        "t20i_hit_rate": new_t20i,
    },
    "beats_live_blended_and_calibration_gate": beats_live,
    "beats_live_on_ipl_specifically": beats_live_on_ipl_specifically,
    "beats_live_on_t20i_specifically": beats_live_on_t20i_specifically,
    "recent_momentum_feature_importance_rank": {
        f: importance_rank[f] for f in recent_features
    },
    "recent_momentum_feature_importance_share": {
        f: float(model.feature_importances_[FEATURES.index(f)] / sum(model.feature_importances_))
        for f in recent_features if f in FEATURES
    },
    "total_features": len(FEATURES),
    "decision": "promote_candidate" if beats_live else "reject_keep_research",
}
joblib.dump(model, output / "sharp_range_model.pkl")
joblib.dump(temperatures, output / "phase_temperatures.pkl")
joblib.dump(FEATURES, output / "feature_cols.pkl")
joblib.dump(CATEGORICAL, output / "categorical_cols.pkl")
(output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
