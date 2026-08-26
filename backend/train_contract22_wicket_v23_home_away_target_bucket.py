"""Follow-up to the home/away re-test (2026-08-25, user-requested): adds
a target-size bucket alongside batting_team_venue_context, letting the
GBM find any real interaction between home/away and how big a chase
target is (e.g. maybe home advantage matters more under bigger-target
pressure) rather than testing venue context alone, which was already
rejected (rank 44/62, IPL regression -0.0022/-0.0034). Same architecture,
split, bowler-known/unknown evaluation, and IPL/T20I split methodology as
every other wicket candidate.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from app.ml.cascade_features import BOWLER_FEATURES, MATCHUP_FEATURES, WICKET_CATEGORICAL, WICKET_FEATURES, build_wicket_base_dataset
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset

VERSION = "contract22_wicket_v23_home_away_target_bucket"
VENUE_FEATURE = "batting_team_venue_context"
TARGET_BUCKET_FEATURE = "target_bucket"
FEATURES = WICKET_FEATURES + [VENUE_FEATURE, TARGET_BUCKET_FEATURE]
CATEGORICAL = WICKET_CATEGORICAL + [VENUE_FEATURE, TARGET_BUCKET_FEATURE]


def _target_bucket(row) -> str:
    if not row["is_chase"]:
        return "not_chase"
    target = row["score_before_over"] + row["runs_required"]
    if target < 140:
        return "low_target"
    if target <= 170:
        return "medium_target"
    return "high_target"


def frame(d, bowler_known: bool):
    result = d[FEATURES].copy()
    if not bowler_known:
        for col in BOWLER_FEATURES + MATCHUP_FEATURES + ["bowler_recency_balls", "bowler_recency_economy", "bowler_recency_wicket_rate"]:
            if col in CATEGORICAL:
                result[col] = "__UNKNOWN__"
            else:
                result[col] = 0.0
    for column in CATEGORICAL:
        result[column] = result[column].fillna("__UNKNOWN__").astype("category")
    return result


def _platt_fit(raw, actual):
    clipped = np.clip(raw, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, random_state=42)
    calibrator.fit(logit, actual)
    return calibrator


def _platt_apply(calibrator, raw):
    clipped = np.clip(raw, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logit)[:, 1]


def _metrics(actual, proba):
    if len(np.unique(actual)) < 2:
        return {"rows": int(len(actual)), "event_rate": float(actual.mean())}
    return {
        "rows": int(len(actual)), "event_rate": float(actual.mean()),
        "auc": float(roc_auc_score(actual, proba)), "brier": float(brier_score_loss(actual, proba)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "models/candidates" / VERSION
    output.mkdir(parents=True, exist_ok=True)

    print("Building wicket base dataset...", flush=True)
    base = build_wicket_base_dataset(root)
    print("Building venue regime dataset (home/away context)...", flush=True)
    venue = build_ipl_venue_regime_dataset(root, scopes=("ipl", "t20i"))
    keys = ["source_file", "innings", "over"]
    data = base.merge(venue[keys + [VENUE_FEATURE]], on=keys, how="inner", validate="one_to_one")
    data[TARGET_BUCKET_FEATURE] = data.apply(_target_bucket, axis=1)
    print(f"Merged rows: {len(data)} (base was {len(base)}, venue {len(venue)})")
    print("target_bucket distribution:", data[TARGET_BUCKET_FEATURE].value_counts().to_dict())

    train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
    calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
    holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)
    holdout_is_ipl = holdout["is_ipl"].to_numpy()

    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=300, learning_rate=0.03, max_depth=5,
        num_leaves=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    )
    model.fit(frame(train, bowler_known=True), train["wicket_in_over"], categorical_feature=CATEGORICAL)

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
            "event_rate": float(actual.mean()), "rows": len(actual),
            "ipl_platt": _metrics(actual[holdout_is_ipl], platt_proba[holdout_is_ipl]),
            "t20i_platt": _metrics(actual[~holdout_is_ipl], platt_proba[~holdout_is_ipl]),
        }

    ceiling = evaluate(bowler_known=True)
    realistic = evaluate(bowler_known=False)

    importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    importance_rank = {name: rank + 1 for rank, (name, _) in enumerate(importances)}

    report = {
        "candidate_version": VERSION,
        "candidate_only": True,
        "production_changed": False,
        "note": (
            "contract22_wicket_v16_team_composition + batting_team_venue_context "
            "+ target_bucket (low/medium/high chase target, or not_chase for "
            "innings 1), letting the GBM find a real interaction rather than "
            "testing venue context alone (already rejected, rank 44/62)."
        ),
        "split": {"train_rows": len(train), "holdout_rows": len(holdout)},
        "reference_contract22_wicket_v16_team_composition": {
            "auc_known_blended": 0.6126, "auc_unknown_blended": 0.6114,
            "auc_known_ipl": 0.6151, "auc_unknown_ipl": 0.6131,
            "auc_known_t20i": 0.6113, "auc_unknown_t20i": 0.6113,
        },
        "reference_v22_venue_context_alone": {
            "auc_known_blended": 0.6129, "auc_known_ipl": 0.6129, "auc_known_t20i": 0.6118,
            "auc_unknown_blended": 0.6112, "auc_unknown_ipl": 0.6097, "auc_unknown_t20i": 0.6113,
        },
        "bowler_known_ceiling": ceiling,
        "bowler_unknown_realistic": realistic,
        "new_feature_importance_rank": {
            VENUE_FEATURE: importance_rank[VENUE_FEATURE],
            TARGET_BUCKET_FEATURE: importance_rank[TARGET_BUCKET_FEATURE],
        },
        "total_features": len(FEATURES),
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
