"""Root cause found in diagnose_ipl_t20i_bias.py: run_range_v4_batter_phase
systematically under-predicts IPL runs-per-over (-0.376 bias overall,
-0.971 in the powerplay specifically) while T20I is nearly unbiased
(-0.073). IPL genuinely scores higher (8.41 vs 7.49 runs/over mean,
population-wide) but is trained with EQUAL per-row weight despite T20I
outnumbering it 1.83:1 in the training split -- the shared model's
decision boundaries get pulled toward T20I's lower center. The wicket
v7.x line already found and fixed this exact failure mode by
down-weighting the augmenting T20I population (0.2 weight, "so 4x the
row volume from a different, only partially overlapping population
doesn't swamp the IPL signal") -- never applied to the run-range line,
which has always pooled both at equal weight.

Sweeps T20I sample_weight in {1.0 (current baseline), 0.5, 0.3, 0.15} --
IPL always weight=1.0 -- refitting once per weight (features built once,
shared). Also tests the more direct version of the same idea
(user-suggested): literal separate models, trained on ONLY IPL rows or
ONLY T20I rows, evaluated against their own competition's holdout only --
does full separation beat weighting, or is weighting already enough?
Reports IPL-only, T20I-only, and blended holdout hit rate for each config,
split by competition per
docs/finding_blended_holdout_masks_ipl_accuracy.md's recommendation.
Single holdout evaluation per config, no iterative peeking-and-re-choosing
-- every point reported honestly regardless of which looks best.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
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

root = Path(__file__).resolve().parents[1]

with open("/private/tmp/claude-501/-Users-susha-Downloads-cricket-predictor/6f8f95cc-ff96-474d-a9b7-2f679a19c32a/scratchpad/ipl_files.pkl", "rb") as f:
    ipl_files = pickle.load(f)

eligible = male_source_files(root)
base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
base = base[base["source_file"].isin(eligible)].copy()
base["match_date"] = pd.to_datetime(base["match_date"])

print("Building features once (shared across all weight settings)...", flush=True)
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
print(f"Merged rows: {len(data)}", flush=True)

train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)

calibration_actual = np.minimum(calibration["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
holdout_actual = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
holdout_is_ipl = holdout["is_ipl"].to_numpy()
train_is_ipl = train["is_ipl"].to_numpy()


def fit_and_evaluate(label: str, train_subset: pd.DataFrame, sample_weight: np.ndarray | None) -> dict:
    print(f"\n=== {label} ===", flush=True)
    sub_actual = np.minimum(train_subset["runs_in_over"].to_numpy(), MAX_RUN_CLASS)

    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=MAX_RUN_CLASS + 1,
        n_estimators=60, learning_rate=0.08, num_leaves=25, max_depth=6,
        min_child_samples=100, subsample=0.85, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=42, verbose=-1,
    )
    fit_kwargs = {"categorical_feature": CATEGORICAL}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(features(train_subset, FEATURES, CATEGORICAL), sub_actual, **fit_kwargs)

    calibration_raw = model.predict_proba(features(calibration, FEATURES, CATEGORICAL))
    holdout_raw = model.predict_proba(features(holdout, FEATURES, CATEGORICAL))

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

    holdout_low, holdout_high = best_bands(holdout_scaled)
    hit = (holdout_actual >= holdout_low) & (holdout_actual <= holdout_high)

    run_values = np.arange(MAX_RUN_CLASS + 1)
    predicted_mean = holdout_scaled @ run_values
    bias_ipl = float(np.mean(predicted_mean[holdout_is_ipl] - holdout_actual[holdout_is_ipl])) if holdout_is_ipl.any() else None
    bias_t20i = float(np.mean(predicted_mean[~holdout_is_ipl] - holdout_actual[~holdout_is_ipl])) if (~holdout_is_ipl).any() else None

    row = {
        "config": label,
        "train_rows": int(len(train_subset)),
        "blended_hit_rate": float(np.mean(hit)),
        "ipl_hit_rate": float(np.mean(hit[holdout_is_ipl])) if holdout_is_ipl.any() else None,
        "t20i_hit_rate": float(np.mean(hit[~holdout_is_ipl])) if (~holdout_is_ipl).any() else None,
        "ipl_bias": bias_ipl,
        "t20i_bias": bias_t20i,
        "phase_temperatures": temperatures,
    }
    print(json.dumps(row, indent=2), flush=True)
    return row


results = []
for t20i_weight in (1.0, 0.5, 0.3, 0.15):
    sample_weight = np.where(train_is_ipl, 1.0, t20i_weight)
    results.append(fit_and_evaluate(f"pooled, t20i_weight={t20i_weight}", train, sample_weight))

# Fully separate models -- the more direct version of the same idea.
ipl_only_train = train[train_is_ipl].reset_index(drop=True)
t20i_only_train = train[~train_is_ipl].reset_index(drop=True)
results.append(fit_and_evaluate("IPL-only model (T20I excluded from training)", ipl_only_train, None))
results.append(fit_and_evaluate("T20I-only model (IPL excluded from training)", t20i_only_train, None))

print("\n\n=== SUMMARY (all configs, honest report) ===")
print(json.dumps(results, indent=2))
(root / "data/reports/run_range_v6_ipl_weight_sweep.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)
