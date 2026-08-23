"""Follow-up to docs/finding_blended_holdout_masks_ipl_accuracy.md: neither
reweighting nor full IPL/T20I separation moved IPL's holdout hit rate --
every configuration landed IPL in the same 23.8-24.6% band regardless of
training population. That points to IPL's intrinsically wider run
distribution (std 4.72 vs T20I's 4.58) as the real constraint, not a
fixable training bug: a fixed width-2 band structurally covers less of a
wider distribution.

This checks that directly and cheaply -- band width is a POST-HOC choice
over an already-fitted model's calibrated probability distribution
(`best_bands(probabilities, width)`), no retraining needed. Uses the
SAME already-validated run_range_v4_batter_phase model (single fit, not
a sweep of models) and sweeps width in {2 (current), 3, 4, 5}, reporting
IPL-only and T20I-only hit rate at each. Answers: what width would IPL
need to reach T20I's ~29% hit rate, and is that width still useful (a
sharp_5 band spans 6 runs -- much less informative than sharp_2's 3).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from app.ml.ipl_phase_moe_dataset import build_ipl_phase_moe_features
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from app.ml.player_venue_phase_dataset import build_player_venue_phase_features
from train_contract22_rigorous import BOWLER_FEATURES, build_enriched
from train_phase_calibrated_sharp_range_v33 import (
    MAX_RUN_CLASS, best_bands, male_source_files, nll, temperature_scale,
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

print("Building features...", flush=True)
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

train_actual = np.minimum(train["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
calibration_actual = np.minimum(calibration["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
holdout_actual = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
holdout_is_ipl = holdout["is_ipl"].to_numpy()

print("Fitting the (already-validated) pooled model, same as live...", flush=True)
model = lgb.LGBMClassifier(
    objective="multiclass", num_class=MAX_RUN_CLASS + 1,
    n_estimators=60, learning_rate=0.08, num_leaves=25, max_depth=6,
    min_child_samples=100, subsample=0.85, colsample_bytree=0.9,
    reg_lambda=1.0, random_state=42, verbose=-1,
)
model.fit(features(train, FEATURES, CATEGORICAL), train_actual, categorical_feature=CATEGORICAL)

calibration_raw = model.predict_proba(features(calibration, FEATURES, CATEGORICAL))
holdout_raw = model.predict_proba(features(holdout, FEATURES, CATEGORICAL))

# Phase-temperature calibration is width-independent (operates on the raw
# per-class probabilities, not the band), so this only needs to be fit
# once -- reused for every width in the sweep below.
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

print(f"\nphase_temperatures: {temperatures}")
print("(should match run_range_v4_batter_phase's documented values -- sanity check this is really the same model)\n")

results = []
for width in (2, 3, 4, 5, 6):
    low, high = best_bands(holdout_scaled, width=width)
    hit = (holdout_actual >= low) & (holdout_actual <= high)
    row = {
        "width": width,
        "band_span_runs": width + 1,
        "blended_hit_rate": float(np.mean(hit)),
        "ipl_hit_rate": float(np.mean(hit[holdout_is_ipl])),
        "t20i_hit_rate": float(np.mean(hit[~holdout_is_ipl])),
    }
    results.append(row)
    print(json.dumps(row, indent=2), flush=True)

print("\n=== SUMMARY ===")
print(json.dumps(results, indent=2))
(root / "data/reports/run_range_v4_ipl_width_sweep.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)
