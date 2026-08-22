"""run_range_enriched_v2_no_identity (26.44% -> 28.37%, beating v3.3's
28.10%, by dropping high-cardinality raw player/venue identity categoricals
that caused overfitting) is the best run-range candidate so far -- but it
deliberately omits the "batsman_style"/"bowler_type"/etc. categoricals from
`docs/model_contract.md`'s intended 22-feature contract, since raw player
IDENTITY (striker/non_striker as categoricals) was the proven overfitting
source, not necessarily player ATTRIBUTES.

This tests whether a genuine, static, zero-leakage player attribute --
batting_style (right/left-hand bat) -- adds anything on top of v2. Unlike
raw identity, this is a fixed, known-in-advance attribute (not derived
from the current match), sourced from `data/external/cricsheet_player_styles.csv`
(16,102 players, keyed by the same Cricsheet person UUID used everywhere
else this session) -- no cardinality-overfitting risk, since it only has a
handful of distinct values (right/left-hand bat, unknown).
`bowling_style`/`bowler_type` deliberately NOT added: the bowler for the
upcoming over is not reliably known before it's bowled (see
`build_ipl_phase_moe_features`'s own "next-over bowler is future
information" comment), so a bowler-dependent feature would mostly serve as
noise/missing in live use, same reasoning v1/v2 already applied to
`known_bowler`.

Same architecture as v3.3 (LightGBM multiclass runs classifier,
MAX_RUN_CLASS=30, same hyperparameters, same phase-temperature
calibration), same train/calibration/holdout split (<=2023 / 2024 /
>=2025), same male-only IPL+T20I population (male_source_files()). Single
holdout evaluation, no peeking. Candidate-only -- does not touch models/
or the accepted v3.3 candidate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from app.ml.ipl_phase_moe_dataset import build_ipl_phase_moe_features
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from train_contract22_rigorous import BOWLER_FEATURES, build_enriched
from train_phase_calibrated_sharp_range_v33 import (
    MAX_RUN_CLASS,
    best_bands,
    male_source_files,
    nll,
    temperature_scale,
)

VERSION = "run_range_enriched_v3_batting_style"

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
    "striker_batting_style",
]
VENUE_FEATURES = [
    "venue_par_score", "venue_prior_innings", "venue_scoring_regime",
    "venue_par_source", "batting_team_venue_context", "phase_venue_regime",
]
# This model previously had zero bowler awareness at all -- no economy
# rate, no bowler type, nothing distinguishing a death-over specialist
# from a part-timer. Raw bowler IDENTITY (known_bowler) was tried and
# dropped for overfitting the same way striker/venue identity was (see
# MOE_EXTRA_FEATURES' comment above); these are the same low-cardinality
# NUMERIC bowler-aggregate features already proven safe in the wicket
# model (contract22_wicket_v2_batter_state), never tried here before.
FEATURES = BASE_FEATURES + MOE_EXTRA_FEATURES + VENUE_FEATURES + BOWLER_FEATURES
CATEGORICAL = [
    "phase", "active_batter_state", "wickets_remaining_bucket",
    "state_regime", "chase_pressure", "venue_scoring_regime",
    "venue_par_source", "batting_team_venue_context", "phase_venue_regime",
    "striker_batting_style", "bowler_type",
]


def _load_batting_style_lookup(root: Path) -> dict[str, str]:
    from app.ml.player_style_registry import normalize_batting_style

    styles = pd.read_csv(root / "data/external/cricsheet_player_styles.csv")
    lookup = {}
    for cricsheet_id, style in zip(styles["cricsheet_id"], styles["batting_style"]):
        if pd.notna(cricsheet_id):
            normalized = normalize_batting_style(style)
            if normalized != "unknown":
                lookup[f"player:{cricsheet_id}"] = normalized
    return lookup


def features(
    frame: pd.DataFrame, columns: list[str], categorical: list[str],
    bowler_known: bool = True,
) -> pd.DataFrame:
    result = frame[columns].copy()
    if not bowler_known:
        # The bowler is often not known ahead of the over in real live
        # serving (same reason the wicket model has this same toggle) --
        # zero out the bowler-dependent columns exactly like the wicket
        # model's frame(bowler_known=False) does, so the realistic-case
        # number isn't inflated by ceiling-only evaluation.
        for column in BOWLER_FEATURES:
            if column in CATEGORICAL:
                result[column] = "__UNKNOWN__"
            else:
                result[column] = 0.0
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
    print("Building bowler features (this model previously had none at all)...")
    enriched = build_enriched(root, eligible)
    keys = ["source_file", "innings", "over"]
    moe_columns = [c for c in moe.columns if c not in ("match_date",) and c not in keys]
    venue_columns = [c for c in venue.columns if c not in ("match_date",) and c not in keys]
    data = base.merge(moe[keys + moe_columns], on=keys, how="inner", validate="one_to_one")
    data = data.merge(venue[keys + venue_columns], on=keys, how="inner", validate="one_to_one")
    data = data.merge(
        enriched[keys + BOWLER_FEATURES], on=keys, how="inner", validate="one_to_one"
    )
    print(f"Merged rows: {len(data)} (base was {len(base)})")

    style_lookup = _load_batting_style_lookup(root)
    data["striker_batting_style"] = data["striker"].map(style_lookup)
    print(f"striker_batting_style resolved for {data['striker_batting_style'].notna().mean():.1%} of rows")

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

    # Realistic case: the bowler often isn't known ahead of the over in
    # real live serving. Re-predict the SAME holdout with bowler features
    # zeroed (same model, same fitted temperatures) to see whether the new
    # bowler features actually help under the condition they'll usually be
    # used in, not just an inflated bowler-known ceiling.
    holdout_unknown_raw = model.predict_proba(
        features(holdout, FEATURES, CATEGORICAL, bowler_known=False)
    )
    holdout_unknown_scaled = holdout_unknown_raw.copy()
    for phase in ("powerplay", "middle", "death"):
        mask = holdout["phase"].astype(str).to_numpy() == phase
        holdout_unknown_scaled[mask] = temperature_scale(holdout_unknown_raw[mask], temperatures[phase])
    unknown_low, unknown_high = best_bands(holdout_unknown_scaled)
    holdout_hit_bowler_unknown = float(
        np.mean((holdout_actual >= unknown_low) & (holdout_actual <= unknown_high))
    )

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
            "Same as run_range_enriched_v2_no_identity (numeric prior-stat/"
            "partnership/chase-pressure/venue-par features, no raw player/"
            "venue identity categoricals) plus striker_batting_style -- a "
            "genuine static player attribute (right/left-hand bat) from "
            "data/external/cricsheet_player_styles.csv, zero leakage risk "
            "and low cardinality, unlike raw player identity. 2026-08-22: "
            "added numeric bowler-aggregate features (BOWLER_FEATURES from "
            "train_contract22_rigorous.py) -- this model previously had NO "
            "bowler awareness at all. holdout_hit is the bowler-known "
            "ceiling; holdout_hit_bowler_unknown re-predicts the same "
            "holdout with those features zeroed (the realistic live case)."
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
            "new_hit_rate_bowler_unknown": holdout_hit_bowler_unknown,
            "relative_improvement_over_v32": float((holdout_hit - old_hit) / old_hit),
            "relative_improvement_over_v33": float((holdout_hit - v33_hit) / v33_hit),
        },
        "holdout_phase_breakdown": phase_breakdown,
        "beats_v32_baseline_and_calibration_gate": accepted_vs_baseline,
        "beats_v33_accepted_candidate": beats_v33,
        "decision": "promote_candidate" if (accepted_vs_baseline and beats_v33) else "reject_keep_research",
    }
    joblib.dump(model, output / "sharp_range_model.pkl")
    joblib.dump(temperatures, output / "phase_temperatures.pkl")
    joblib.dump(FEATURES, output / "feature_cols.pkl")
    joblib.dump(CATEGORICAL, output / "categorical_cols.pkl")
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions = holdout[["source_file", "match_date", "innings", "over", "phase", "runs_in_over"]].copy()
    predictions["range_low"] = holdout_low
    predictions["range_high"] = holdout_high
    predictions["hit"] = (holdout_actual >= holdout_low) & (holdout_actual <= holdout_high)
    predictions.to_csv(output / "holdout_predictions.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
