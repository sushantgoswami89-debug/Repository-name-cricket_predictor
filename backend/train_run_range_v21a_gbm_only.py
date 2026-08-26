"""Part 1/3 of a GBM+NN ensemble test for run-range -- never tried before
on this target. The GBM+NN ensemble is a proven architecture change: it
gave a real, measured win on the wicket target
(finding_wicket_nn_gbm_ensemble_real_win.md, now live as a shadow). This
tests the same architecture idea against run-range, using the EXACT SAME
feature set as the currently-live run_range_v11_partnership_rate (no new
features -- this test is about the model architecture, not more inputs).

Split into separate processes because torch and lightgbm cannot coexist
in one process on this machine (confirmed: hangs/silently dies regardless
of import order or KMP_DUPLICATE_LIB_OK). This script: lightgbm only, no
torch. Also caches the built feature dataframe to parquet so parts 2 and
3 don't have to rebuild ~10 dataset joins from scratch.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from app.ml.ipl_phase_moe_dataset import build_ipl_phase_moe_features
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from app.ml.partnership_dataset import build_partnership_dataset
from app.ml.player_competition_dataset import build_player_competition_features
from app.ml.player_venue_phase_dataset import build_player_venue_phase_features
from train_contract22_rigorous import BOWLER_FEATURES, build_enriched
from train_phase_calibrated_sharp_range_v33 import MAX_RUN_CLASS, male_source_files
from train_run_range_enriched_v3_batting_style import (
    BASE_FEATURES, MOE_EXTRA_FEATURES, VENUE_FEATURES, _load_batting_style_lookup, features,
)
from train_run_range_v4_batter_phase import BATTER_PHASE_FEATURES

VERSION = "run_range_v21_nn_gbm_ensemble"
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
eligible = male_source_files(root)
base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
base = base[base["source_file"].isin(eligible)].copy()
base["match_date"] = pd.to_datetime(base["match_date"])

print("Building enriched MOE + venue features...", flush=True)
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
data = data.merge(batter_phase[keys + BATTER_PHASE_FEATURES], on=keys, how="inner", validate="one_to_one")
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

cache_columns = list(dict.fromkeys(FEATURES + CATEGORICAL + ["runs_in_over", "phase", "is_ipl"]))
train[cache_columns].to_pickle(output / "train_cache.pkl")
calibration[cache_columns].to_pickle(output / "calibration_cache.pkl")
holdout[cache_columns].to_pickle(output / "holdout_cache.pkl")
print("Cached train/calibration/holdout feature frames to pickle.")

model = lgb.LGBMClassifier(
    objective="multiclass", num_class=MAX_RUN_CLASS + 1,
    n_estimators=60, learning_rate=0.08, num_leaves=25, max_depth=6,
    min_child_samples=100, subsample=0.85, colsample_bytree=0.9,
    reg_lambda=1.0, random_state=42, verbose=-1,
)
train_actual = np.minimum(train["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
model.fit(features(train, FEATURES, CATEGORICAL), train_actual, categorical_feature=CATEGORICAL)

calibration_actual = np.minimum(calibration["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
holdout_actual = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
gbm_cal = model.predict_proba(features(calibration, FEATURES, CATEGORICAL))
gbm_holdout = model.predict_proba(features(holdout, FEATURES, CATEGORICAL))

np.savez(
    output / "gbm_only.npz",
    gbm_cal=gbm_cal, gbm_holdout=gbm_holdout,
    cal_actual=calibration_actual, holdout_actual=holdout_actual,
    holdout_is_ipl=holdout["is_ipl"].to_numpy(),
    cal_phase=calibration["phase"].astype(str).to_numpy(),
    holdout_phase=holdout["phase"].astype(str).to_numpy(),
)
print("Saved gbm_only.npz")
print(json.dumps({"train_rows": len(train), "cal_rows": len(calibration), "holdout_rows": len(holdout)}))
