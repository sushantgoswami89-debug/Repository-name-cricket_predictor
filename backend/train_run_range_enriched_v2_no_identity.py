"""run_range_enriched_v1 (same enrichment as this file, but including raw
identity categoricals: striker, non_striker, venue_name, batting_team,
known_bowler) rejected: 26.44% holdout hit rate, WORSE than both the
v3.2 baseline (27.28%) and the currently-accepted v3.3 (28.10%). Post-fit
phase-temperature values were sharply higher than v3.3's (1.72-1.87 vs.
~1.05-1.14), pointing at overfitting -- likely from high-cardinality
identity categoricals (1000+ distinct players) added without any
compensating regularization change from v3.3's own hyperparameters.

This variant drops the raw identity categoricals and keeps only the
numeric prior-stat/partnership/chase-pressure/venue-par features, to test
that specific hypothesis in isolation. Same architecture, split, and
population as v1/v3.3 otherwise.

Same architecture as v3.3 (LightGBM multiclass runs classifier,
MAX_RUN_CLASS=30, same hyperparameters, same phase-temperature
calibration), same train/calibration/holdout split (<=2023 / 2024 /
>=2025), same male-only IPL+T20I population (male_source_files()). Only
the FEATURES list changes: adds MOE_FEATURES-style enrichment (player
identity/prior stats, partnership/chase-pressure state) and
STRUCTURAL_VENUE_FEATURES (venue par score/regime) on top of the existing
17. Single holdout evaluation, no peeking. Candidate-only -- does not
touch models/ or the accepted v3.3 candidate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from app.ml.ipl_phase_moe_dataset import build_ipl_phase_moe_features
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from train_phase_calibrated_sharp_range_v33 import (
    MAX_RUN_CLASS,
    best_bands,
    male_source_files,
    nll,
    temperature_scale,
)

VERSION = "run_range_enriched_v2_no_identity"

BASE_FEATURES = [
    "over", "score_before_over", "wkts_down_before_over", "phase",
    "wickets_in_hand", "legal_balls_bowled", "balls_remaining",
    "current_run_rate", "is_chase", "runs_required", "required_run_rate",
    "recent_legal_balls", "recent_runs_per_ball", "recent_dot_rate",
    "recent_single_rate", "recent_boundary_rate", "recent_wicket_rate",
]
MOE_EXTRA_FEATURES = [
    # Raw identity categoricals (striker, non_striker, venue_name,
    # batting_team, known_bowler) dropped for this variant -- v1 (same
    # features + those identities) regressed vs. v3.3, and the sharply
    # higher post-fit temperature-calibration values (1.72-1.87 vs.
    # v3.3's ~1.05-1.14) pointed at overfitting on high-cardinality
    # identity categoricals (1000+ distinct players) without added
    # regularization. This keeps only the numeric prior-stat/state
    # features to test that specific hypothesis.
    "active_batter_state", "new_batter", "partnership_legal_ball_age",
    "wickets_remaining_bucket", "state_regime", "chase_pressure",
    "striker_match_balls", "partner_match_balls", "striker_prior_balls",
    "striker_prior_runs_per_ball", "striker_prior_dot_rate",
    "striker_prior_boundary_rate", "striker_prior_dismissal_rate",
    "partner_prior_balls", "partner_prior_runs_per_ball",
    "partner_prior_dot_rate", "partner_prior_boundary_rate",
]
VENUE_FEATURES = [
    "venue_par_score", "venue_prior_innings", "venue_scoring_regime",
    "venue_par_source", "batting_team_venue_context", "phase_venue_regime",
]
FEATURES = BASE_FEATURES + MOE_EXTRA_FEATURES + VENUE_FEATURES
CATEGORICAL = [
    "phase", "active_batter_state", "wickets_remaining_bucket",
    "state_regime", "chase_pressure", "venue_scoring_regime",
    "venue_par_source", "batting_team_venue_context", "phase_venue_regime",
]


def features(frame: pd.DataFrame, columns: list[str], categorical: list[str]) -> pd.DataFrame:
    result = frame[columns].copy()
    for column in categorical:
        result[column] = result[column].fillna("__UNKNOWN__").astype("category")
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "models/candidates" / VERSION
    output.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    eligible = male_source_files(root)
    base = base[base["source_file"].isin(eligible)].copy()
    base["match_date"] = pd.to_datetime(base["match_date"])

    print("Building enriched MOE + venue features (scopes=ipl+t20i)...")
    moe = build_ipl_phase_moe_features(root, canonical_identities=True, scopes=("ipl", "t20i"))
    venue = build_ipl_venue_regime_dataset(root, scopes=("ipl", "t20i"))
    keys = ["source_file", "innings", "over"]
    moe_columns = [c for c in moe.columns if c not in ("match_date",) and c not in keys]
    venue_columns = [c for c in venue.columns if c not in ("match_date",) and c not in keys]
    data = base.merge(moe[keys + moe_columns], on=keys, how="inner", validate="one_to_one")
    data = data.merge(venue[keys + venue_columns], on=keys, how="inner", validate="one_to_one")
    print(f"Merged rows: {len(data)} (base was {len(base)})")

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

    temperatures: dict[str, float] = {}
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
    calibration_hit = np.mean((calibration_actual >= calibration_low) & (calibration_actual <= calibration_high))
    holdout_hit = np.mean((holdout_actual >= holdout_low) & (holdout_actual <= holdout_high))
    raw_calibration_nll = nll(calibration_raw, calibration_actual)
    scaled_calibration_nll = nll(calibration_scaled, calibration_actual)

    old_hit = 0.2728246539222149  # documented v3.2 baseline, same reference v3.3 uses
    v33_hit = 0.28103713469567126  # documented v3.3 holdout hit rate (this session's re-verified number)
    accepted_vs_baseline = bool(holdout_hit > old_hit and scaled_calibration_nll <= raw_calibration_nll)
    beats_v33 = bool(holdout_hit > v33_hit)

    # phase-level breakdown for honesty beyond the single headline number
    phase_breakdown = {}
    for phase in ("powerplay", "middle", "death"):
        mask = holdout["phase"].astype(str).to_numpy() == phase
        if mask.sum() == 0:
            continue
        phase_breakdown[phase] = {
            "rows": int(mask.sum()),
            "hit_rate": float(np.mean((holdout_actual[mask] >= holdout_low[mask]) & (holdout_actual[mask] <= holdout_high[mask]))),
        }

    report = {
        "candidate_version": VERSION,
        "candidate_only": True,
        "production_changed": False,
        "run_model_changed": False,
        "note": (
            "Same architecture/hyperparameters/split as v3.3, features "
            "extended with MOE (player identity/prior-stat/chase-pressure) "
            "and venue-regime enrichment via the scopes=(ipl,t20i) builders "
            "added this session for the wicket-model line."
        ),
        "split": {
            "train_rows": len(train), "calibration_rows": len(calibration),
            "holdout_rows": len(holdout), "holdout_matches": int(holdout["source_file"].nunique()),
        },
        "phase_temperatures": temperatures,
        "calibration": {
            "raw_nll": raw_calibration_nll, "phase_scaled_nll": scaled_calibration_nll,
            "hit_rate": float(calibration_hit),
        },
        "holdout": {
            "old_v32_baseline_hit_rate": old_hit,
            "v33_accepted_hit_rate": v33_hit,
            "new_hit_rate": float(holdout_hit),
            "relative_improvement_over_v32": float((holdout_hit - old_hit) / old_hit),
            "relative_improvement_over_v33": float((holdout_hit - v33_hit) / v33_hit),
        },
        "holdout_phase_breakdown": phase_breakdown,
        "beats_v32_baseline_and_calibration_gate": accepted_vs_baseline,
        "beats_v33_accepted_candidate": beats_v33,
        "decision": "promote_candidate" if (accepted_vs_baseline and beats_v33) else "reject_keep_research",
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions = holdout[["source_file", "match_date", "innings", "over", "phase", "runs_in_over"]].copy()
    predictions["range_low"] = holdout_low
    predictions["range_high"] = holdout_high
    predictions["hit"] = (holdout_actual >= holdout_low) & (holdout_actual <= holdout_high)
    predictions.to_csv(output / "holdout_predictions.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
