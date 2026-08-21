"""Train a live-feature candidate and validate it on unseen recent matches."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from app.ml.calibrated_model import CalibratedBinaryClassifier
from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS


LIVE_NUMERIC_FEATURES = [
    "over",
    "score_before_over",
    "wkts_down_before_over",
    "balls_faced_before_over",
]
LIVE_CATEGORICAL_FEATURES = ["phase"]
FEATURE_COLUMNS = LIVE_NUMERIC_FEATURES + LIVE_CATEGORICAL_FEATURES


def _source_metadata(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    match_id = 0
    for archive in ("ipl", "t20i"):
        pattern = project_root / "data/raw/cricsheet" / archive / "*.json"
        for filename in glob.glob(str(pattern)):
            match_id += 1
            path = Path(filename)
            with path.open(encoding="utf-8") as handle:
                raw = json.load(handle)
            info = raw["info"]
            rows.append(
                {
                    "match_id": match_id,
                    "source_file": path.name,
                    "match_date": str(info["dates"][0]),
                    "source_venue": info.get("venue", "Unknown"),
                    "competition_scope": (
                        "approved_domestic_ipl"
                        if archive == "ipl"
                        else "icc_recognized_international"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _load_dataset(project_root: Path) -> pd.DataFrame:
    data = pd.read_csv(project_root / "data/real_overs.csv")
    metadata = _source_metadata(project_root)
    data = data.merge(metadata, on="match_id", how="left", validate="many_to_one")
    if data["match_date"].isna().any():
        raise ValueError("Could not map every training match to source metadata.")
    venue_matches = data["venue"].fillna("Unknown") == data["source_venue"].fillna(
        "Unknown"
    )
    if float(venue_matches.mean()) < 0.99:
        raise ValueError("Training match order no longer matches the source archive.")
    excluded = set(KNOWN_RULE_ANOMALY_EXCLUSIONS)
    data = data[~data["source_file"].str.removesuffix(".json").isin(excluded)].copy()
    data["match_date"] = pd.to_datetime(data["match_date"])
    return data.sort_values(["match_date", "match_id", "over"]).reset_index(drop=True)


def _prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data[FEATURE_COLUMNS].copy()
    features["phase"] = features["phase"].astype("category")
    return features


def _interval_profiles(
    calibration: pd.DataFrame,
    prediction: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    residuals = calibration[["phase"]].copy()
    residuals["residual"] = calibration["runs_in_over"].to_numpy() - prediction
    profiles: dict[str, dict[str, float | int]] = {}
    for phase, frame in residuals.groupby("phase", observed=True):
        profiles[str(phase)] = {
            # Calibrate to 92% on the calibration year to provide a small
            # distribution-shift buffer for the independent 90% holdout target.
            "lower_residual": float(frame["residual"].quantile(0.04)),
            "upper_residual": float(frame["residual"].quantile(0.96)),
            "calibration_rows": int(len(frame)),
        }
    profiles["default"] = {
        "lower_residual": float(residuals["residual"].quantile(0.04)),
        "upper_residual": float(residuals["residual"].quantile(0.96)),
        "calibration_rows": int(len(residuals)),
    }
    return profiles


def _interval_metrics(
    holdout: pd.DataFrame,
    prediction: np.ndarray,
    profiles: dict[str, dict[str, float | int]],
) -> dict[str, float]:
    lows: list[float] = []
    highs: list[float] = []
    for phase, center in zip(holdout["phase"], prediction, strict=True):
        profile = profiles.get(str(phase), profiles["default"])
        lows.append(max(0.0, center + float(profile["lower_residual"])))
        highs.append(center + float(profile["upper_residual"]))
    actual = holdout["runs_in_over"].to_numpy()
    low_values = np.asarray(lows)
    high_values = np.asarray(highs)
    return {
        "coverage": float(np.mean((actual >= low_values) & (actual <= high_values))),
        "average_width": float(np.mean(high_values - low_values)),
    }


def train_and_validate(project_root: Path, output_dir: Path) -> dict[str, Any]:
    data = _load_dataset(project_root)
    train = data[data["match_date"].dt.year <= 2023]
    calibration = data[data["match_date"].dt.year == 2024]
    holdout = data[data["match_date"].dt.year >= 2025]
    if min(len(train), len(calibration), len(holdout)) == 0:
        raise ValueError("Chronological train, calibration, and holdout sets are required.")

    x_train = _prepare_features(train)
    x_calibration = _prepare_features(calibration)
    x_holdout = _prepare_features(holdout)

    runs_model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=600,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=7,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    runs_model.fit(
        x_train,
        train["runs_in_over"],
        categorical_feature=LIVE_CATEGORICAL_FEATURES,
    )

    wicket_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.025,
        num_leaves=25,
        max_depth=6,
        min_child_samples=100,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    wicket_model.fit(
        x_train,
        train["wicket_in_over"],
        categorical_feature=LIVE_CATEGORICAL_FEATURES,
    )
    calibration_probability = wicket_model.predict_proba(x_calibration)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        calibration_probability, calibration["wicket_in_over"]
    )

    runs_prediction = np.clip(runs_model.predict(x_holdout), 0, None)
    calibration_runs_prediction = np.clip(
        runs_model.predict(x_calibration), 0, None
    )
    interval_profiles = _interval_profiles(calibration, calibration_runs_prediction)
    interval_metrics = _interval_metrics(
        holdout, runs_prediction, interval_profiles
    )
    raw_wicket_probability = wicket_model.predict_proba(x_holdout)[:, 1]
    wicket_probability = calibrator.predict(raw_wicket_probability)
    actual_runs = holdout["runs_in_over"].to_numpy()
    actual_wickets = holdout["wicket_in_over"].to_numpy()

    baseline_runs = pd.read_csv(
        project_root / "data/reports/accuracy_diagnostic/calibration_candidates.csv"
    )
    diagnostic = json.loads(
        (
            project_root / "data/reports/accuracy_diagnostic/accuracy_diagnostic.json"
        ).read_text(encoding="utf-8")
    )
    baseline_brier = float(
        baseline_runs.loc[
            baseline_runs["candidate"] == "uncalibrated", "brier_score"
        ].iloc[0]
    )
    baseline_runs_mae = float(
        diagnostic["runs_linear_calibration"]["raw_holdout_mae"]
    )

    candidate_metrics = {
        "runs_mae": float(mean_absolute_error(actual_runs, runs_prediction)),
        "runs_rmse": float(mean_squared_error(actual_runs, runs_prediction) ** 0.5),
        "runs_bias": float(np.mean(runs_prediction - actual_runs)),
        "wicket_brier": float(brier_score_loss(actual_wickets, wicket_probability)),
        "wicket_log_loss": float(log_loss(actual_wickets, wicket_probability)),
        "wicket_auc": float(roc_auc_score(actual_wickets, wicket_probability)),
    }
    comparison = {
        "runs_mae_baseline": baseline_runs_mae,
        "runs_mae_candidate": candidate_metrics["runs_mae"],
        "runs_mae_improvement": (
            baseline_runs_mae - candidate_metrics["runs_mae"]
        )
        / baseline_runs_mae,
        "wicket_brier_baseline": baseline_brier,
        "wicket_brier_candidate": candidate_metrics["wicket_brier"],
        "wicket_brier_improvement": (
            baseline_brier - candidate_metrics["wicket_brier"]
        )
        / baseline_brier,
    }
    promotion_gates = {
        "runs_mae_improves": candidate_metrics["runs_mae"] < baseline_runs_mae,
        "wicket_brier_improves": candidate_metrics["wicket_brier"] < baseline_brier,
        "runs_bias_below_one": abs(candidate_metrics["runs_bias"]) < 1.0,
    }
    promoted = all(promotion_gates.values())

    output_dir.mkdir(parents=True, exist_ok=True)
    calibrated_wicket_model = CalibratedBinaryClassifier(wicket_model, calibrator)
    joblib.dump(runs_model, output_dir / "runs_model.pkl")
    joblib.dump(calibrated_wicket_model, output_dir / "wkt_model.pkl")
    joblib.dump(wicket_model, output_dir / "wicket_model_raw_candidate.pkl")
    joblib.dump(calibrator, output_dir / "wicket_calibrator_candidate.pkl")
    joblib.dump(FEATURE_COLUMNS, output_dir / "feature_cols.pkl")
    joblib.dump(LIVE_CATEGORICAL_FEATURES, output_dir / "cat_cols.pkl")

    result = {
        "candidate_version": "v2_live_features_time_split",
        "model_scope": "candidate_only",
        "production_models_changed": False,
        "dataset": {
            "eligible_rows": len(data),
            "train_rows_through_2023": len(train),
            "calibration_rows_2024": len(calibration),
            "unseen_holdout_rows_2025_plus": len(holdout),
        },
        "features": FEATURE_COLUMNS,
        "candidate_metrics": candidate_metrics,
        "prediction_interval_90": interval_metrics,
        "comparison": comparison,
        "promotion_gates": promotion_gates,
        "promotion_recommended": promoted,
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_dir / "interval_profiles.json").write_text(
        json.dumps(interval_profiles, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "models/candidates/v2_live_features_time_split",
    )
    args = parser.parse_args()
    print(json.dumps(train_and_validate(project_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
