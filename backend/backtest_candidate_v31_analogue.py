"""Calibrate and backtest the Candidate v3.1 historical analogue layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error

from app.ml.historical_analogue import AnalogueBatch, HistoricalAnalogueEngine


def _model_predictions(
    model_dir: Path, frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    features: list[str] = joblib.load(model_dir / "feature_cols.pkl")
    categorical: list[str] = joblib.load(model_dir / "cat_cols.pkl")
    x = frame[features].copy()
    for column in categorical:
        x[column] = x[column].astype("category")
    runs = np.clip(joblib.load(model_dir / "runs_model.pkl").predict(x), 0, None)
    wickets = joblib.load(model_dir / "wkt_model.pkl").predict_proba(x)[:, 1]
    return runs, wickets


def _reliability(batch: AnalogueBatch) -> np.ndarray:
    sample = np.clip(batch.effective_sample_size / 180.0, 0.0, 1.0)
    consistency = np.clip(1.0 - batch.outcome_std / 10.0, 0.20, 1.0)
    return batch.similarity * np.sqrt(sample) * consistency


def _select_blend(
    actual: np.ndarray,
    model: np.ndarray,
    analogue: np.ndarray,
    reliability: np.ndarray,
    metric: str,
) -> tuple[float, list[dict[str, float]]]:
    trials: list[dict[str, float]] = []
    for maximum_weight in np.arange(0.0, 0.51, 0.05):
        weight = np.clip(reliability * maximum_weight, 0.0, maximum_weight)
        prediction = model * (1.0 - weight) + analogue * weight
        score = (
            float(mean_absolute_error(actual, prediction))
            if metric == "mae"
            else float(brier_score_loss(actual, prediction))
        )
        trials.append({"maximum_analogue_weight": float(maximum_weight), metric: score})
    selected = min(trials, key=lambda trial: trial[metric])
    return float(selected["maximum_analogue_weight"]), trials


def _blend(
    model: np.ndarray,
    analogue: np.ndarray,
    reliability: np.ndarray,
    maximum_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    weight = np.clip(reliability * maximum_weight, 0.0, maximum_weight)
    return model * (1.0 - weight) + analogue * weight, weight


def _select_padding(
    actual: np.ndarray,
    center: np.ndarray,
    analogue: AnalogueBatch,
    target_coverage: float = 0.915,
) -> tuple[float, list[dict[str, float]]]:
    shift = center - analogue.expected_runs
    trials: list[dict[str, float]] = []
    for padding in np.arange(0.0, 4.01, 0.25):
        low = np.maximum(0.0, analogue.lower_runs + shift - padding)
        high = analogue.upper_runs + shift + padding
        coverage = float(np.mean((actual >= low) & (actual <= high)))
        trials.append(
            {
                "padding": float(padding),
                "coverage": coverage,
                "average_width": float(np.mean(high - low)),
            }
        )
    passing = [trial for trial in trials if trial["coverage"] >= target_coverage]
    selected = min(passing, key=lambda trial: trial["average_width"])
    return float(selected["padding"]), trials


def _intervals(
    center: np.ndarray, analogue: AnalogueBatch, padding: float
) -> tuple[np.ndarray, np.ndarray]:
    shift = center - analogue.expected_runs
    return (
        np.maximum(0.0, analogue.lower_runs + shift - padding),
        analogue.upper_runs + shift + padding,
    )


def _confidence_features(
    frame: pd.DataFrame,
    model_runs: np.ndarray,
    analogue: AnalogueBatch,
    low: np.ndarray,
    high: np.ndarray,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "similarity": analogue.similarity,
            "log_effective_sample": np.log1p(analogue.effective_sample_size),
            "outcome_std": analogue.outcome_std,
            "model_analogue_gap": np.abs(model_runs - analogue.expected_runs),
            "range_width": high - low,
            "recent_wicket_rate": frame["recent_wicket_rate"].to_numpy(),
            "required_run_rate": frame["required_run_rate"].to_numpy(),
        }
    )
    phase = pd.get_dummies(frame["phase"], prefix="phase", dtype=float)
    return pd.concat([result, phase.reset_index(drop=True)], axis=1)


def _confidence_report(
    actual_runs: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, Any]:
    bins = np.linspace(0.0, 1.0, 11)
    bucket_rows: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(10):
        mask = (confidence >= bins[index]) & (
            confidence <= bins[index + 1]
            if index == 9
            else confidence < bins[index + 1]
        )
        if not mask.any():
            continue
        predicted = float(np.mean(confidence[mask]))
        observed = float(np.mean(target[mask]))
        weight = float(np.mean(mask))
        ece += weight * abs(predicted - observed)
        bucket_rows.append(
            {
                "band": f"{bins[index]:.1f}-{bins[index + 1]:.1f}",
                "rows": int(mask.sum()),
                "mean_confidence": predicted,
                "observed_success_rate": observed,
                "runs_mae": float(
                    mean_absolute_error(actual_runs[mask], prediction[mask])
                ),
            }
        )
    order = np.argsort(confidence)
    third = max(1, len(order) // 3)
    low_indices = order[:third]
    high_indices = order[-third:]
    return {
        "brier": float(brier_score_loss(target, confidence)),
        "expected_calibration_error": ece,
        "low_confidence_runs_mae": float(
            mean_absolute_error(actual_runs[low_indices], prediction[low_indices])
        ),
        "high_confidence_runs_mae": float(
            mean_absolute_error(actual_runs[high_indices], prediction[high_indices])
        ),
        "bands": bucket_rows,
    }


def _phase_intervals(
    frame: pd.DataFrame, actual: np.ndarray, low: np.ndarray, high: np.ndarray
) -> dict[str, dict[str, float | int]]:
    report: dict[str, dict[str, float | int]] = {}
    phases = frame["phase"].astype(str).to_numpy()
    for phase in ("powerplay", "middle", "death"):
        mask = phases == phase
        report[phase] = {
            "rows": int(mask.sum()),
            "coverage": float(
                np.mean((actual[mask] >= low[mask]) & (actual[mask] <= high[mask]))
            ),
            "average_width": float(np.mean(high[mask] - low[mask])),
        }
    return report


def _sequence_failures(frame: pd.DataFrame) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for source_file, match in frame.groupby("source_file"):
        for innings_number, innings in match.groupby("innings"):
            innings = innings.sort_values("over")
            scores = innings["score_before_over"].astype(int).tolist()
            actuals = innings["runs_in_over"].astype(int).tolist()
            overs = innings["over"].astype(int).tolist()
            if any(b <= a for a, b in zip(overs, overs[1:], strict=False)):
                failures.append(
                    {
                        "source_file": source_file,
                        "innings": int(innings_number),
                        "error": "non-increasing over sequence",
                    }
                )
            for index in range(1, len(scores)):
                if scores[index] != scores[index - 1] + actuals[index - 1]:
                    failures.append(
                        {
                            "source_file": source_file,
                            "innings": int(innings_number),
                            "over": overs[index],
                            "error": "score does not reconcile with prior over",
                        }
                    )
    return failures


def backtest(project_root: Path, output_dir: Path) -> dict[str, Any]:
    data = pd.read_csv(project_root / "data/candidates/v3/verified_training_overs.csv")
    data["match_date"] = pd.to_datetime(data["match_date"])
    train = data[data["match_date"].dt.year <= 2023].reset_index(drop=True)
    calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
    holdout = data[data["match_date"].dt.year >= 2025].reset_index(drop=True)
    model_dir = project_root / "models/locked/cricketbaba_candidate_v3"
    calibration_model_runs, calibration_model_wickets = _model_predictions(
        model_dir, calibration
    )
    holdout_model_runs, holdout_model_wickets = _model_predictions(model_dir, holdout)

    calibration_engine = HistoricalAnalogueEngine().fit(train)
    calibration_analogues = calibration_engine.query(
        calibration, lower_quantile=0.05, upper_quantile=0.95
    )
    calibration_reliability = _reliability(calibration_analogues)
    run_weight, run_trials = _select_blend(
        calibration["runs_in_over"].to_numpy(),
        calibration_model_runs,
        calibration_analogues.expected_runs,
        calibration_reliability,
        "mae",
    )
    wicket_weight, wicket_trials = _select_blend(
        calibration["wicket_in_over"].to_numpy(),
        calibration_model_wickets,
        calibration_analogues.wicket_probability,
        calibration_reliability,
        "brier",
    )
    calibration_runs, _ = _blend(
        calibration_model_runs,
        calibration_analogues.expected_runs,
        calibration_reliability,
        run_weight,
    )
    padding, padding_trials = _select_padding(
        calibration["runs_in_over"].to_numpy(),
        calibration_runs,
        calibration_analogues,
    )
    calibration_low, calibration_high = _intervals(
        calibration_runs, calibration_analogues, padding
    )
    calibration_success = (
        np.abs(calibration_runs - calibration["runs_in_over"].to_numpy()) <= 4.0
    ).astype(int)
    confidence_model = LogisticRegression(max_iter=1000, random_state=42)
    confidence_model.fit(
        _confidence_features(
            calibration,
            calibration_model_runs,
            calibration_analogues,
            calibration_low,
            calibration_high,
        ),
        calibration_success,
    )

    historical = pd.concat([train, calibration], ignore_index=True)
    holdout_engine = HistoricalAnalogueEngine().fit(historical)
    holdout_analogues = holdout_engine.query(
        holdout, lower_quantile=0.05, upper_quantile=0.95
    )
    holdout_reliability = _reliability(holdout_analogues)
    predicted_runs, run_blend = _blend(
        holdout_model_runs,
        holdout_analogues.expected_runs,
        holdout_reliability,
        run_weight,
    )
    wicket_probability, wicket_blend = _blend(
        holdout_model_wickets,
        holdout_analogues.wicket_probability,
        holdout_reliability,
        wicket_weight,
    )
    low, high = _intervals(predicted_runs, holdout_analogues, padding)
    confidence = confidence_model.predict_proba(
        _confidence_features(
            holdout,
            holdout_model_runs,
            holdout_analogues,
            low,
            high,
        )
    )[:, 1]
    actual_runs = holdout["runs_in_over"].to_numpy()
    actual_wickets = holdout["wicket_in_over"].to_numpy()
    success = (np.abs(predicted_runs - actual_runs) <= 4.0).astype(int)
    coverage = float(np.mean((actual_runs >= low) & (actual_runs <= high)))
    average_width = float(np.mean(high - low))
    candidate_metrics = {
        "runs_mae": float(mean_absolute_error(actual_runs, predicted_runs)),
        "runs_bias": float(np.mean(predicted_runs - actual_runs)),
        "wicket_brier": float(brier_score_loss(actual_wickets, wicket_probability)),
        "interval_coverage": coverage,
        "interval_average_width": average_width,
    }
    confidence_metrics = _confidence_report(
        actual_runs, predicted_runs, success, confidence
    )
    phase_intervals = _phase_intervals(holdout, actual_runs, low, high)
    sequence_failures = _sequence_failures(holdout)
    locked = json.loads((model_dir / "validation_report.json").read_text())
    gates = {
        "runs_mae_not_worse": candidate_metrics["runs_mae"]
        <= locked["candidate_metrics"]["runs_mae"],
        "wicket_brier_not_worse": candidate_metrics["wicket_brier"]
        <= locked["candidate_metrics"]["wicket_brier"],
        "overall_coverage_at_least_90pct": coverage >= 0.90,
        "all_phase_coverage_at_least_88pct": all(
            values["coverage"] >= 0.88 for values in phase_intervals.values()
        ),
        "range_narrower_than_locked": average_width
        < locked["prediction_interval_90"]["average_width"],
        "confidence_ece_below_5pct": confidence_metrics["expected_calibration_error"]
        < 0.05,
        "high_confidence_more_accurate": confidence_metrics["high_confidence_runs_mae"]
        < confidence_metrics["low_confidence_runs_mae"],
        "complete_match_sample_at_least_100": holdout["source_file"].nunique() >= 100,
        "complete_match_sequence_failures_zero": not sequence_failures,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "candidate_version": "v3.1_historical_analogue",
        "model_scope": "candidate_only",
        "production_models_changed": False,
        "method": {
            "neighbors": calibration_engine.neighbors,
            "same_phase_required": True,
            "same_chase_status_required": True,
            "reference_for_calibration": "through_2023",
            "reference_for_holdout": "through_2024",
            "analogue_quantiles": [0.05, 0.95],
            "selected_run_maximum_weight": run_weight,
            "selected_wicket_maximum_weight": wicket_weight,
            "selected_interval_padding": padding,
            "confidence_target": "absolute run error at most four",
        },
        "split": {
            "historical_reference_rows": len(historical),
            "calibration_rows_2024": len(calibration),
            "unseen_holdout_rows_2025_plus": len(holdout),
            "unseen_complete_matches": int(holdout["source_file"].nunique()),
        },
        "locked_v3_metrics": {
            **locked["candidate_metrics"],
            **{
                "interval_coverage": locked["prediction_interval_90"]["coverage"],
                "interval_average_width": locked["prediction_interval_90"][
                    "average_width"
                ],
            },
        },
        "candidate_metrics": candidate_metrics,
        "phase_intervals": phase_intervals,
        "confidence_calibration": confidence_metrics,
        "complete_match_test": {
            "matches_tested": int(holdout["source_file"].nunique()),
            "innings_tested": int(
                holdout[["source_file", "innings"]].drop_duplicates().shape[0]
            ),
            "overs_predicted": len(holdout),
            "sequence_failures": len(sequence_failures),
            "failures": sequence_failures,
        },
        "analogue_diagnostics": {
            "mean_similarity": float(np.mean(holdout_analogues.similarity)),
            "mean_effective_sample_size": float(
                np.mean(holdout_analogues.effective_sample_size)
            ),
            "mean_run_blend_weight": float(np.mean(run_blend)),
            "mean_wicket_blend_weight": float(np.mean(wicket_blend)),
        },
        "calibration_search": {
            "run_blend": run_trials,
            "wicket_blend": wicket_trials,
            "interval_padding": padding_trials,
        },
        "promotion_gates": gates,
        "promotion_recommended": all(gates.values()),
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    predictions = holdout[
        ["source_file", "match_date", "innings", "over", "phase"]
    ].copy()
    predictions["model_runs"] = holdout_model_runs
    predictions["analogue_runs"] = holdout_analogues.expected_runs
    predictions["predicted_runs"] = predicted_runs
    predictions["actual_runs"] = actual_runs
    predictions["range_low"] = low
    predictions["range_high"] = high
    predictions["range_covered"] = (actual_runs >= low) & (actual_runs <= high)
    predictions["wicket_probability"] = wicket_probability
    predictions["actual_wicket"] = actual_wickets
    predictions["confidence"] = confidence
    predictions["analogue_similarity"] = holdout_analogues.similarity
    predictions["effective_sample_size"] = holdout_analogues.effective_sample_size
    predictions.to_csv(output_dir / "holdout_predictions.csv", index=False)
    joblib.dump(confidence_model, output_dir / "confidence_calibrator.pkl")
    joblib.dump(holdout_engine, output_dir / "historical_analogue_engine.pkl")
    (output_dir / "analogue_config.json").write_text(
        json.dumps(report["method"], indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            backtest(root, root / "models/candidates/v3.1_historical_analogue"),
            indent=2,
        )
    )
