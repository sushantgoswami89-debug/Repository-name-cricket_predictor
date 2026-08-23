"""User-flagged (2026-08-22): most stadiums in the training data are no
longer in active IPL rotation -- 328 of 1,243 matches (26.4%) were played
at venues last used 2021 or earlier (UAE relocated seasons: Dubai/Abu
Dhabi/Sharjah, 111 matches; Pune MCA/Mumbai Brabourne/DY Patil, 115
matches; plus the older 2009-2018 defunct venues). Does the currently-live
run_range_v4_batter_phase model's holdout accuracy actually differ between
venues that matter today (used 2023-2026) vs. the discontinued ones it was
still trained and evaluated on?

Reuses v4's exact feature-building pipeline and trained model unchanged --
this is a breakdown of the SAME already-validated holdout evaluation by
venue validity, not a retrain. A second retrain-restricted-to-active-venues
comparison is a natural follow-up if this shows a real gap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from app.ml.ipl_phase_moe_dataset import build_ipl_phase_moe_features
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from app.ml.ipl_venues import normalize_ipl_venue
from app.ml.player_venue_phase_dataset import build_player_venue_phase_features
from train_contract22_rigorous import BOWLER_FEATURES, build_enriched
from train_phase_calibrated_sharp_range_v33 import (
    MAX_RUN_CLASS, best_bands, male_source_files, nll, temperature_scale,
)
from train_run_range_enriched_v3_batting_style import (
    BASE_FEATURES, MOE_EXTRA_FEATURES, VENUE_FEATURES,
    _load_batting_style_lookup, features,
)
from train_run_range_v4_batter_phase import BATTER_PHASE_FEATURES, FEATURES, CATEGORICAL
import lightgbm as lgb

root = Path(__file__).resolve().parents[1]

# Empirically-derived "currently active" venue set: union of venues used
# in real matches 2023-2026 (same check reported to the user).
CURRENT_VENUES = {
    "ahmedabad_narendra_modi", "bengaluru_chinnaswamy", "chennai_chepauk",
    "delhi_arun_jaitley", "dharamsala_hpca", "guwahati_barsapara",
    "hyderabad_rajiv_gandhi", "jaipur_sawai_mansingh", "kolkata_eden_gardens",
    "lucknow_ekana", "mohali_is_bindra", "mumbai_wankhede",
    "new_chandigarh_mullanpur", "raipur_shaheed_veer_narayan",
    "visakhapatnam_acavdca",
}

eligible = male_source_files(root)
base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
base = base[base["source_file"].isin(eligible)].copy()
base["match_date"] = pd.to_datetime(base["match_date"])

print("Building enriched MOE + venue features (scopes=ipl+t20i)...")
moe = build_ipl_phase_moe_features(root, canonical_identities=True, scopes=("ipl", "t20i"))
venue = build_ipl_venue_regime_dataset(root, scopes=("ipl", "t20i"))
print("Building bowler features...")
enriched = build_enriched(root, eligible)
print("Building batter phase features...")
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
print(f"Merged rows: {len(data)} (base was {len(base)})")

style_lookup = _load_batting_style_lookup(root)
data["striker_batting_style"] = data["striker"].map(style_lookup)

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

ipl_files = {p.name for p in (root / "data/raw/cricsheet/ipl").glob("*.json")}

holdout_venue_norm = holdout["venue_name"].astype(str)
is_ipl_scope = holdout["source_file"].isin(ipl_files).to_numpy()
is_active = holdout_venue_norm.isin(CURRENT_VENUES).to_numpy()

# Restrict entirely to IPL-scoped holdout rows -- T20I venues (Botswana,
# Denmark, Japan, etc.) are a separate, already-validated augmentation
# strategy, not what's being asked about here.
ipl_hit = hit[is_ipl_scope]
ipl_active = is_active[is_ipl_scope]

overall_hit = float(np.mean(hit))
ipl_overall_hit = float(np.mean(ipl_hit)) if is_ipl_scope.any() else float("nan")
active_hit = float(np.mean(ipl_hit[ipl_active])) if ipl_active.any() else float("nan")
inactive_hit = float(np.mean(ipl_hit[~ipl_active])) if (~ipl_active).any() else float("nan")

t20i_hit_rate = float(np.mean(hit[~is_ipl_scope])) if (~is_ipl_scope).any() else float("nan")

report = {
    "reference_v4_documented_hit_rate_all_scopes": 0.2860,
    "reproduced_overall_holdout_hit_rate_all_scopes": overall_hit,
    "holdout_rows_total_all_scopes": int(len(holdout)),
    "holdout_rows_ipl_scope_only": int(is_ipl_scope.sum()),
    "holdout_rows_t20i_scope_only": int((~is_ipl_scope).sum()),
    "ipl_scope_overall_hit_rate": ipl_overall_hit,
    "t20i_scope_overall_hit_rate": t20i_hit_rate,
    "ipl_rows_active_venues": int(ipl_active.sum()),
    "ipl_rows_inactive_venues": int((~ipl_active).sum()),
    "hit_rate_ipl_active_venues_only": active_hit,
    "hit_rate_ipl_inactive_venues_only": inactive_hit,
    "gap_active_minus_inactive": active_hit - inactive_hit if not np.isnan(inactive_hit) else None,
}
print(json.dumps(report, indent=2))

venue_breakdown = (
    pd.DataFrame({"venue": holdout_venue_norm[is_ipl_scope], "hit": ipl_hit})
    .groupby("venue")
    .agg(rows=("hit", "size"), hit_rate=("hit", "mean"))
    .sort_values("rows", ascending=False)
)
print()
print("Per-venue holdout breakdown (IPL scope only):")
print(venue_breakdown.to_string())
