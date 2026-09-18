"""Tests batter/bowler handedness matchup as a wicket-model feature.
See app/ml/handedness_matchup_dataset.py's module docstring: checked
2026-09-18 that this is genuinely new information the model can't
already reconstruct -- the live bowler_type feature is reduced all the
way to just "pace"/"spin", discarding bowling arm entirely, even though
the raw source data has it and a matching normalizer
(normalize_bowling_style) already exists but is used nowhere.

Identical to the live contract22_wicket_v16_team_composition in every
respect except adding bowler_arm/handedness_match as two new categorical
features. Reference is v16's own real numbers (this file's own
validation_report.json), not a stale comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from app.ml.handedness_matchup_dataset import build_handedness_matchup_dataset
from app.ml.ipl_phase_moe_dataset import build_ipl_phase_moe_features
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from app.ml.partnership_dataset import build_partnership_dataset
from app.ml.recency_weighted_prior_dataset import (
    build_batter_phase_recency_dataset, build_recency_weighted_prior_dataset,
)
from app.ml.team_composition_dataset import build_team_composition_dataset
from train_contract22_rigorous import (
    BASE_FEATURES, BATTER_FEATURES, BOWLER_FEATURES, MATCHUP_FEATURES,
    VENUE_FEATURES, build_enriched,
)
from train_contract22_wicket_v2_batter_state import STATE_FEATURES
from train_phase_calibrated_sharp_range_v33 import male_source_files

VERSION = "contract22_wicket_v28_handedness_matchup"
RECENCY_FEATURES = [
    "striker_recency_balls", "striker_recency_runs_per_ball", "striker_recency_dot_rate",
    "striker_recency_boundary_rate", "striker_recency_dismissal_rate",
    "partner_recency_runs_per_ball",
    "bowler_recency_balls", "bowler_recency_economy", "bowler_recency_wicket_rate",
]
PARTNERSHIP_FEATURES = ["partnership_runs", "partnership_balls", "partnership_run_rate"]
BATTER_PHASE_RECENCY_FEATURES = [
    "striker_recency_phase_balls", "striker_recency_phase_runs_per_ball",
    "striker_recency_phase_boundary_rate", "striker_recency_phase_dismissal_rate",
]
TEAM_COMPOSITION_FEATURES = [
    "batting_team_allrounder_count", "batting_team_specialist_batter_count",
    "bowling_team_pace_count", "bowling_team_spin_count", "bowling_team_allrounder_count",
]
HANDEDNESS_FEATURES = ["bowler_arm", "handedness_match"]
FEATURES = (
    BASE_FEATURES + BATTER_FEATURES + BOWLER_FEATURES + MATCHUP_FEATURES
    + VENUE_FEATURES + STATE_FEATURES + RECENCY_FEATURES + PARTNERSHIP_FEATURES
    + BATTER_PHASE_RECENCY_FEATURES + TEAM_COMPOSITION_FEATURES + HANDEDNESS_FEATURES
)
CATEGORICAL = [
    "phase", "striker_batting_style", "bowler_type", "venue_scoring_regime",
    "venue_par_source", "active_batter_state", "bowler_arm", "handedness_match",
]


def frame(d: pd.DataFrame, bowler_known: bool) -> pd.DataFrame:
    result = d[FEATURES].copy()
    if not bowler_known:
        for col in BOWLER_FEATURES + MATCHUP_FEATURES + HANDEDNESS_FEATURES + ["bowler_recency_balls", "bowler_recency_economy", "bowler_recency_wicket_rate"]:
            if col in CATEGORICAL:
                result[col] = "__UNKNOWN__"
            else:
                result[col] = 0.0
    for column in CATEGORICAL:
        result[column] = result[column].fillna("__UNKNOWN__").astype("category")
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "models/candidates" / VERSION
    output.mkdir(parents=True, exist_ok=True)

    eligible = male_source_files(root)
    base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    base = base[base["source_file"].isin(eligible)].copy()
    base["match_date"] = pd.to_datetime(base["match_date"])

    enriched = build_enriched(root, eligible)
    venue = build_ipl_venue_regime_dataset(root, scopes=("ipl", "t20i"))
    moe = build_ipl_phase_moe_features(root, canonical_identities=True, scopes=("ipl", "t20i"))
    recency = build_recency_weighted_prior_dataset(root, scopes=("ipl", "t20i"))
    partnership = build_partnership_dataset(root, scopes=("ipl", "t20i"))
    batter_phase_recency = build_batter_phase_recency_dataset(root, scopes=("ipl", "t20i"))
    team_composition = build_team_composition_dataset(root, scopes=("ipl", "t20i"))
    print("Building handedness matchup features (the new signal being tested)...", flush=True)
    handedness = build_handedness_matchup_dataset(root, scopes=("ipl", "t20i"))
    keys = ["source_file", "innings", "over"]
    data = base.merge(enriched, on=keys, how="inner", validate="one_to_one")
    data = data.merge(
        venue[keys + ["venue_par_score", "venue_prior_innings", "venue_scoring_regime", "venue_par_source"]],
        on=keys, how="inner", validate="one_to_one",
    )
    data = data.merge(moe[keys + STATE_FEATURES], on=keys, how="inner", validate="one_to_one")
    data = data.merge(recency[keys + RECENCY_FEATURES], on=keys, how="inner", validate="one_to_one")
    data = data.merge(partnership[keys + PARTNERSHIP_FEATURES], on=keys, how="inner", validate="one_to_one")
    data = data.merge(batter_phase_recency[keys + BATTER_PHASE_RECENCY_FEATURES], on=keys, how="inner", validate="one_to_one")
    data = data.merge(team_composition[keys + TEAM_COMPOSITION_FEATURES], on=keys, how="inner", validate="one_to_one")
    data = data.merge(handedness[keys + HANDEDNESS_FEATURES], on=keys, how="inner", validate="one_to_one")
    print(f"Merged rows: {len(data)} (base was {len(base)})")

    ipl_files = {p.name for p in (root / "data/raw/cricsheet/ipl").glob("*.json")}
    data["is_ipl"] = data["source_file"].isin(ipl_files)

    train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
    calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
    holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)
    holdout_is_ipl = holdout["is_ipl"].to_numpy()

    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=300, learning_rate=0.03, max_depth=5,
        num_leaves=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    )
    model.fit(frame(train, bowler_known=True), train["wicket_in_over"], categorical_feature=CATEGORICAL)

    def _platt_fit(raw: np.ndarray, actual: np.ndarray) -> LogisticRegression:
        clipped = np.clip(raw, 1e-6, 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        calibrator = LogisticRegression(C=1.0, random_state=42)
        calibrator.fit(logit, actual)
        return calibrator

    def _platt_apply(calibrator: LogisticRegression, raw: np.ndarray) -> np.ndarray:
        clipped = np.clip(raw, 1e-6, 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        return calibrator.predict_proba(logit)[:, 1]

    def _metrics(actual: np.ndarray, proba: np.ndarray) -> dict:
        if len(np.unique(actual)) < 2:
            return {"rows": int(len(actual)), "event_rate": float(actual.mean())}
        return {
            "rows": int(len(actual)),
            "event_rate": float(actual.mean()),
            "auc": float(roc_auc_score(actual, proba)),
            "brier": float(brier_score_loss(actual, proba)),
        }

    def evaluate(bowler_known: bool) -> dict:
        calibration_raw = model.predict_proba(frame(calibration, bowler_known))[:, 1]
        holdout_raw = model.predict_proba(frame(holdout, bowler_known))[:, 1]
        actual_cal = calibration["wicket_in_over"].to_numpy()
        actual = holdout["wicket_in_over"].to_numpy()

        platt = _platt_fit(calibration_raw, actual_cal)
        platt_proba = _platt_apply(platt, holdout_raw)

        return {
            "auc": float(roc_auc_score(actual, holdout_raw)),
            "brier_platt": float(brier_score_loss(actual, platt_proba)),
            "brier_uncalibrated": float(brier_score_loss(actual, holdout_raw)),
            "event_rate": float(actual.mean()),
            "rows": len(actual),
            "ipl_platt": _metrics(actual[holdout_is_ipl], platt_proba[holdout_is_ipl]),
            "t20i_platt": _metrics(actual[~holdout_is_ipl], platt_proba[~holdout_is_ipl]),
        }

    ceiling = evaluate(bowler_known=True)
    realistic = evaluate(bowler_known=False)

    importances = sorted(
        zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]
    )
    importance_rank = {name: rank + 1 for rank, (name, _) in enumerate(importances)}

    reference = {
        "known": {
            "blended_auc": 0.612588, "blended_brier_platt": 0.204349,
            "ipl_auc": 0.615064, "ipl_brier_platt": 0.193320,
            "t20i_auc": 0.611290, "t20i_brier_platt": 0.206345,
        },
        "unknown": {
            "blended_auc": 0.611448, "blended_brier_platt": 0.204541,
            "ipl_auc": 0.613111, "ipl_brier_platt": 0.194369,
            "t20i_auc": 0.611252, "t20i_brier_platt": 0.206382,
        },
    }
    beats_live_known = bool(
        ceiling["auc"] > reference["known"]["blended_auc"]
        and ceiling["brier_platt"] <= reference["known"]["blended_brier_platt"]
    )
    beats_live_unknown = bool(
        realistic["auc"] > reference["unknown"]["blended_auc"]
        and realistic["brier_platt"] <= reference["unknown"]["blended_brier_platt"]
    )

    report = {
        "candidate_version": VERSION,
        "candidate_only": True,
        "production_changed": False,
        "note": (
            "contract22_wicket_v16_team_composition (currently live) + "
            "batter/bowler handedness matchup (bowler_arm, "
            "handedness_match) -- genuinely new information, not "
            "reconstructible from the existing pace/spin-only bowler_type. "
            "Reference is v16's own real validation_report.json numbers."
        ),
        "split": {"train_rows": len(train), "holdout_rows": len(holdout)},
        "reference_contract22_wicket_v16_team_composition_live": reference,
        "bowler_known_ceiling": ceiling,
        "bowler_unknown_realistic": realistic,
        "beats_live_known": beats_live_known,
        "beats_live_unknown": beats_live_unknown,
        "handedness_feature_importance_rank": {
            f: importance_rank[f] for f in HANDEDNESS_FEATURES
        },
        "total_features": len(FEATURES),
        "decision": "promote_candidate" if (beats_live_known and beats_live_unknown) else "reject_keep_research",
    }

    calibration_raw_prod = model.predict_proba(frame(calibration, bowler_known=True))[:, 1]
    production_calibrator = _platt_fit(calibration_raw_prod, calibration["wicket_in_over"].to_numpy())
    joblib.dump(model, output / "wicket_model.pkl")
    joblib.dump(production_calibrator, output / "wicket_calibrator.pkl")
    joblib.dump(FEATURES, output / "feature_cols.pkl")
    joblib.dump(CATEGORICAL, output / "categorical_cols.pkl")
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
