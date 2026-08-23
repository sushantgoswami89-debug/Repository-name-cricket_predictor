"""Root-cause check for the IPL-vs-T20I holdout accuracy gap found in
docs/finding_blended_holdout_masks_ipl_accuracy.md (IPL 24.19% vs T20I
29.22% on run_range_v4_batter_phase's holdout). Two candidate mechanisms:
(1) IPL is intrinsically higher-mean/higher-variance (8.41/4.72 vs
7.49/4.58 runs-per-over), making any fixed-width band harder to land;
(2) T20I outnumbers IPL ~1.83:1 in training with no reweighting, pulling
the model's calibration toward the pooled center rather than IPL's own
higher one. This checks for (2) directly: is the model's predicted
distribution SYSTEMATICALLY biased low for IPL specifically?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from app.ml.ipl_phase_moe_dataset import build_ipl_phase_moe_features
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from app.ml.player_venue_phase_dataset import build_player_venue_phase_features
from train_contract22_rigorous import BOWLER_FEATURES, build_enriched
from train_phase_calibrated_sharp_range_v33 import (
    MAX_RUN_CLASS, male_source_files, nll, temperature_scale,
)
from train_run_range_enriched_v3_batting_style import _load_batting_style_lookup, features
from train_run_range_v4_batter_phase import BATTER_PHASE_FEATURES, FEATURES, CATEGORICAL
import lightgbm as lgb

root = Path(__file__).resolve().parents[1]

ipl_files = {p.name for p in (root / "data/raw/cricsheet/ipl").glob("*.json")}

eligible = male_source_files(root)
base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
base = base[base["source_file"].isin(eligible)].copy()
base["match_date"] = pd.to_datetime(base["match_date"])

moe = build_ipl_phase_moe_features(root, canonical_identities=True, scopes=("ipl", "t20i"))
venue = build_ipl_venue_regime_dataset(root, scopes=("ipl", "t20i"))
enriched = build_enriched(root, eligible)
batter_phase = build_player_venue_phase_features(root, scopes=("ipl", "t20i"))
keys = ["source_file", "innings", "over"]
moe_columns = [c for c in moe.columns if c not in ("match_date",) and c not in keys]
venue_columns = [c for c in venue.columns if c not in ("match_date",) and c not in keys]
data = base.merge(moe[keys + moe_columns], on=keys, how="inner", validate="one_to_one")
data = data.merge(venue[keys + venue_columns], on=keys, how="inner", validate="one_to_one")
data = data.merge(enriched[keys + BOWLER_FEATURES], on=keys, how="inner", validate="one_to_one")
data = data.merge(
    batter_phase[keys + BATTER_PHASE_FEATURES], on=keys, how="inner", validate="one_to_one"
)
style_lookup = _load_batting_style_lookup(root)
data["striker_batting_style"] = data["striker"].map(style_lookup)
data["is_ipl"] = data["source_file"].isin(ipl_files)

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

temperatures = {}
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
    holdout_scaled[holdout_mask] = temperature_scale(holdout_raw[holdout_mask], temperatures[phase])

run_values = np.arange(MAX_RUN_CLASS + 1)
predicted_mean = holdout_scaled @ run_values  # expected value of the model's own distribution, per row

is_ipl = holdout["is_ipl"].to_numpy()
bias_ipl = float(np.mean(predicted_mean[is_ipl] - holdout_actual[is_ipl]))
bias_t20i = float(np.mean(predicted_mean[~is_ipl] - holdout_actual[~is_ipl]))
mean_pred_ipl = float(np.mean(predicted_mean[is_ipl]))
mean_actual_ipl = float(np.mean(holdout_actual[is_ipl]))
mean_pred_t20i = float(np.mean(predicted_mean[~is_ipl]))
mean_actual_t20i = float(np.mean(holdout_actual[~is_ipl]))

print("=== Bias check: model's predicted mean vs actual, by competition ===")
print(f"IPL:  mean predicted={mean_pred_ipl:.3f}  mean actual={mean_actual_ipl:.3f}  bias(pred-actual)={bias_ipl:+.3f}")
print(f"T20I: mean predicted={mean_pred_t20i:.3f}  mean actual={mean_actual_t20i:.3f}  bias(pred-actual)={bias_t20i:+.3f}")
print()

# Per-phase breakdown, since temperature/calibration is phase-specific
print("=== Per-phase bias breakdown ===")
for phase in ("powerplay", "middle", "death"):
    mask_phase = holdout["phase"].astype(str).to_numpy() == phase
    for label, comp_mask in [("IPL", is_ipl), ("T20I", ~is_ipl)]:
        m = mask_phase & comp_mask
        if m.sum() == 0:
            continue
        b = float(np.mean(predicted_mean[m] - holdout_actual[m]))
        print(f"  {phase:>10s} / {label:4s}: n={m.sum():5d}  bias={b:+.3f}  actual_mean={holdout_actual[m].mean():.2f}  pred_mean={predicted_mean[m].mean():.2f}")

print()
print("=== Train-set row counts (imbalance check) ===")
print(f"train IPL rows: {int(train['is_ipl'].sum())}, T20I rows: {int((~train['is_ipl']).sum())}, ratio T20I:IPL = {(~train['is_ipl']).sum()/train['is_ipl'].sum():.3f}")
