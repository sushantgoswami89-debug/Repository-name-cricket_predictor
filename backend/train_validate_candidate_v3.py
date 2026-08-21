"""Backtest phase-impact features and train CricketBaba Candidate v3 in shadow."""

from __future__ import annotations

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
    roc_auc_score,
)

from app.ml.calibrated_model import CalibratedBinaryClassifier
from app.ml.candidate_v3_dataset import build_verified_dataset

BASE_FEATURES = ["over", "score_before_over", "wkts_down_before_over", "phase"]
PHASE_IMPACT_FEATURES = [
    "wickets_in_hand",
    "legal_balls_bowled",
    "balls_remaining",
    "current_run_rate",
    "is_chase",
    "runs_required",
    "required_run_rate",
    "recent_legal_balls",
    "recent_runs_per_ball",
    "recent_dot_rate",
    "recent_single_rate",
    "recent_boundary_rate",
    "recent_wicket_rate",
]
V3_FEATURES = BASE_FEATURES + PHASE_IMPACT_FEATURES
CATEGORICAL_FEATURES = ["phase"]


def _features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame[columns].copy()
    if "phase" in result:
        result["phase"] = result["phase"].astype("category")
    return result


def _models() -> tuple[lgb.LGBMRegressor, lgb.LGBMClassifier]:
    runs = lgb.LGBMRegressor(
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
    wickets = lgb.LGBMClassifier(
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
    return runs, wickets


def _fit(
    train: pd.DataFrame, calibration: pd.DataFrame, columns: list[str]
) -> tuple[Any, Any, Any]:
    runs, raw_wickets = _models()
    runs.fit(
        _features(train, columns),
        train["runs_in_over"],
        categorical_feature=CATEGORICAL_FEATURES,
    )
    raw_wickets.fit(
        _features(train, columns),
        train["wicket_in_over"],
        categorical_feature=CATEGORICAL_FEATURES,
    )
    calibration_probability = raw_wickets.predict_proba(
        _features(calibration, columns)
    )[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        calibration_probability, calibration["wicket_in_over"]
    )
    return runs, raw_wickets, calibrator


def _metrics(
    frame: pd.DataFrame, runs: Any, wickets: Any, calibrator: Any, columns: list[str]
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    x = _features(frame, columns)
    run_prediction = np.clip(runs.predict(x), 0, None)
    wicket_probability = calibrator.predict(wickets.predict_proba(x)[:, 1])
    actual_runs = frame["runs_in_over"].to_numpy()
    actual_wickets = frame["wicket_in_over"].to_numpy()
    result = {
        "rows": float(len(frame)),
        "runs_mae": float(mean_absolute_error(actual_runs, run_prediction)),
        "runs_rmse": float(np.sqrt(np.mean(np.square(run_prediction - actual_runs)))),
        "runs_bias": float(np.mean(run_prediction - actual_runs)),
        "wicket_brier": float(brier_score_loss(actual_wickets, wicket_probability)),
        "wicket_log_loss": float(log_loss(actual_wickets, wicket_probability)),
        "wicket_auc": float(roc_auc_score(actual_wickets, wicket_probability)),
    }
    return result, run_prediction, wicket_probability


def _intervals(
    calibration: pd.DataFrame,
    calibration_prediction: np.ndarray,
    holdout: pd.DataFrame,
    holdout_prediction: np.ndarray,
) -> tuple[dict[str, dict[str, float | int]], dict[str, float]]:
    residual = calibration["runs_in_over"].to_numpy() - calibration_prediction
    profiles: dict[str, dict[str, float | int]] = {}
    for phase in sorted(calibration["phase"].unique()):
        values = residual[calibration["phase"].to_numpy() == phase]
        profiles[str(phase)] = {
            "lower_residual": float(np.quantile(values, 0.04)),
            "upper_residual": float(np.quantile(values, 0.96)),
            "calibration_rows": int(len(values)),
        }
    profiles["default"] = {
        "lower_residual": float(np.quantile(residual, 0.04)),
        "upper_residual": float(np.quantile(residual, 0.96)),
        "calibration_rows": int(len(residual)),
    }
    lows, highs = [], []
    for phase, center in zip(holdout["phase"], holdout_prediction, strict=True):
        profile = profiles.get(str(phase), profiles["default"])
        lows.append(max(0.0, center + float(profile["lower_residual"])))
        highs.append(center + float(profile["upper_residual"]))
    actual = holdout["runs_in_over"].to_numpy()
    return profiles, {
        "coverage": float(
            np.mean((actual >= np.asarray(lows)) & (actual <= np.asarray(highs)))
        ),
        "average_width": float(np.mean(np.asarray(highs) - np.asarray(lows))),
    }


def train_and_validate(project_root: Path, output_dir: Path) -> dict[str, Any]:
    dataset = build_verified_dataset(project_root)
    data = dataset.frame.copy()
    data["match_date"] = pd.to_datetime(data["match_date"])
    train = data[data["match_date"].dt.year <= 2023]
    calibration = data[data["match_date"].dt.year == 2024]
    holdout = data[data["match_date"].dt.year >= 2025]
    if min(map(len, (train, calibration, holdout))) == 0:
        raise ValueError(
            "Chronological train, calibration, and holdout sets are required."
        )

    base_runs, base_wickets, base_calibrator = _fit(train, calibration, BASE_FEATURES)
    v3_runs, v3_wickets, v3_calibrator = _fit(train, calibration, V3_FEATURES)
    base_metrics, _, _ = _metrics(
        holdout, base_runs, base_wickets, base_calibrator, BASE_FEATURES
    )
    v3_metrics, v3_run_prediction, _ = _metrics(
        holdout, v3_runs, v3_wickets, v3_calibrator, V3_FEATURES
    )
    calibration_run_prediction = np.clip(
        v3_runs.predict(_features(calibration, V3_FEATURES)), 0, None
    )
    profiles, interval_metrics = _intervals(
        calibration, calibration_run_prediction, holdout, v3_run_prediction
    )

    phase_backtest: dict[str, Any] = {}
    phase_non_regression = True
    for phase in ("powerplay", "middle", "death"):
        segment = holdout[holdout["phase"] == phase]
        base_phase, _, _ = _metrics(
            segment, base_runs, base_wickets, base_calibrator, BASE_FEATURES
        )
        v3_phase, _, _ = _metrics(
            segment, v3_runs, v3_wickets, v3_calibrator, V3_FEATURES
        )
        mae_delta = (base_phase["runs_mae"] - v3_phase["runs_mae"]) / base_phase[
            "runs_mae"
        ]
        brier_delta = (
            base_phase["wicket_brier"] - v3_phase["wicket_brier"]
        ) / base_phase["wicket_brier"]
        phase_non_regression &= mae_delta >= -0.02 and brier_delta >= -0.02
        phase_backtest[phase] = {
            "base": base_phase,
            "candidate_v3": v3_phase,
            "runs_mae_improvement": mae_delta,
            "wicket_brier_improvement": brier_delta,
        }

    v2_report = json.loads(
        (
            project_root
            / "models/candidates/v2_live_features_time_split/validation_report.json"
        ).read_text()
    )
    v2 = v2_report["candidate_metrics"]
    gates = {
        "dataset_verified": bool(dataset.report["verified"]),
        "phase_impact_runs_mae_improves": v3_metrics["runs_mae"]
        < base_metrics["runs_mae"],
        "phase_impact_wicket_brier_improves": v3_metrics["wicket_brier"]
        < base_metrics["wicket_brier"],
        "beats_candidate_v2_runs_mae": v3_metrics["runs_mae"] < float(v2["runs_mae"]),
        "beats_candidate_v2_wicket_brier": v3_metrics["wicket_brier"]
        < float(v2["wicket_brier"]),
        "runs_bias_below_one": abs(v3_metrics["runs_bias"]) < 1.0,
        "interval_coverage_at_least_90pct": interval_metrics["coverage"] >= 0.90,
        "phase_non_regression_within_2pct": bool(phase_non_regression),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(v3_runs, output_dir / "runs_model.pkl")
    joblib.dump(
        CalibratedBinaryClassifier(v3_wickets, v3_calibrator),
        output_dir / "wkt_model.pkl",
    )
    joblib.dump(v3_wickets, output_dir / "wicket_model_raw_candidate.pkl")
    joblib.dump(v3_calibrator, output_dir / "wicket_calibrator_candidate.pkl")
    joblib.dump(V3_FEATURES, output_dir / "feature_cols.pkl")
    joblib.dump(CATEGORICAL_FEATURES, output_dir / "cat_cols.pkl")
    (output_dir / "interval_profiles.json").write_text(json.dumps(profiles, indent=2))
    report = {
        "candidate_version": "v3_phase_impact_time_split",
        "model_scope": "candidate_only",
        "production_models_changed": False,
        "dataset": dataset.report,
        "split": {
            "train_through_2023": len(train),
            "calibration_2024": len(calibration),
            "holdout_2025_plus": len(holdout),
        },
        "features": V3_FEATURES,
        "ablation_base_metrics": base_metrics,
        "candidate_metrics": v3_metrics,
        "phase_impact_backtest": phase_backtest,
        "prediction_interval_90": interval_metrics,
        "promotion_gates": gates,
        "complete_match_test": {"status": "pending"},
        "promotion_recommended": False,
    }
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            train_and_validate(
                root, root / "models/candidates/v3_phase_impact_time_split"
            ),
            indent=2,
        )
    )
