"""Final check before closing the batter-pressure-response line
(2026-08-25): combines both pressure formulations tested today --
one-over stall response (v20) and sustained multi-over survival (v24) --
in a single candidate. Both showed nearly identical individual behavior
(importance rank ~17-18, IPL regression ~-0.0016/-0.0024 both cuts),
suggesting they capture largely the same underlying signal rather than
complementary ones. This confirms (or overturns) that directly, per this
project's standing practice of testing a combination before fully
closing an investigation (same as the matchup-score closure). Same
architecture, split, bowler-known/unknown evaluation, and IPL/T20I split
methodology as every other wicket candidate.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from app.ml.batter_pressure_response_dataset import build_batter_pressure_response_dataset
from app.ml.cascade_features import BOWLER_FEATURES, MATCHUP_FEATURES, WICKET_CATEGORICAL, WICKET_FEATURES, build_wicket_base_dataset
from app.ml.pressure_survival_dataset import build_pressure_survival_dataset

VERSION = "contract22_wicket_v25_combined_pressure"
V20_FEATURES = ["pressure_trigger", "striker_pressure_dismissal_rate_shrunk"]
V24_FEATURES = ["overs_under_pressure_this_spell", "striker_extended_pressure_dismissal_rate_shrunk"]
NEW_FEATURES = V20_FEATURES + V24_FEATURES
FEATURES = WICKET_FEATURES + NEW_FEATURES
CATEGORICAL = WICKET_CATEGORICAL

REFERENCE_V16 = {
    "auc_known_blended": 0.6126, "auc_unknown_blended": 0.6114,
    "auc_known_ipl": 0.6151, "auc_unknown_ipl": 0.6131,
    "auc_known_t20i": 0.6113, "auc_unknown_t20i": 0.6113,
}


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
    print("Building both pressure datasets...", flush=True)
    pressure_v20 = build_batter_pressure_response_dataset(root)
    pressure_v24 = build_pressure_survival_dataset(root)
    keys = ["source_file", "innings", "over"]
    data = base.merge(pressure_v20[keys + V20_FEATURES], on=keys, how="inner", validate="one_to_one")
    data = data.merge(pressure_v24[keys + V24_FEATURES], on=keys, how="inner", validate="one_to_one")
    print(f"Merged rows: {len(data)} (base was {len(base)})")

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
            "contract22_wicket_v16_team_composition + BOTH pressure "
            "formulations tested today (one-over stall response + "
            "sustained multi-over survival response) combined -- final "
            "check before closing the batter-pressure-response "
            "investigation."
        ),
        "split": {"train_rows": len(train), "holdout_rows": len(holdout)},
        "reference_contract22_wicket_v16_team_composition": REFERENCE_V16,
        "reference_v20_pressure_trigger_alone": {
            "auc_known_blended": 0.6128, "auc_known_ipl": 0.6135, "auc_known_t20i": 0.6117,
            "auc_unknown_blended": 0.6116, "auc_unknown_ipl": 0.6106, "auc_unknown_t20i": 0.6119,
        },
        "reference_v24_pressure_survival_alone": {
            "auc_known_blended": 0.6130, "auc_known_ipl": 0.6135, "auc_known_t20i": 0.6119,
            "auc_unknown_blended": 0.6111, "auc_unknown_ipl": 0.6107, "auc_unknown_t20i": 0.6110,
        },
        "bowler_known_ceiling": ceiling,
        "bowler_unknown_realistic": realistic,
        "new_feature_importance_rank": {f: importance_rank[f] for f in NEW_FEATURES},
        "total_features": len(FEATURES),
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
