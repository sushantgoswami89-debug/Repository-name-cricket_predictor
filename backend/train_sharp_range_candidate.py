"""Train a direct next-over distribution model for one/two-run sharp ranges."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from train_validate_candidate_v3 import CATEGORICAL_FEATURES, V3_FEATURES, _features

MAX_RUN_CLASS = 30


def _best_bands(
    probabilities: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the highest-probability inclusive integer band of a fixed width."""

    class_count = probabilities.shape[1]
    window_size = width + 1
    window_probability = np.column_stack(
        [
            probabilities[:, start : start + window_size].sum(axis=1)
            for start in range(class_count - width)
        ]
    )
    low = np.argmax(window_probability, axis=1)
    probability = window_probability[np.arange(len(probabilities)), low]
    return low, low + width, probability


def _naive_bands(
    point_prediction: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray]:
    low = np.maximum(0, np.rint(point_prediction - width / 2).astype(int))
    high = low + width
    return low, high


def _phase_report(
    frame: pd.DataFrame,
    actual: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    phases = frame["phase"].astype(str).to_numpy()
    for phase in ("powerplay", "middle", "death"):
        mask = phases == phase
        result[phase] = {
            "rows": int(mask.sum()),
            "hit_rate": float(
                np.mean((actual[mask] >= low[mask]) & (actual[mask] <= high[mask]))
            ),
        }
    return result


def _sequence_failures(frame: pd.DataFrame) -> int:
    failures = 0
    for _, match in frame.groupby("source_file"):
        for _, innings in match.groupby("innings"):
            innings = innings.sort_values("over")
            scores = innings["score_before_over"].astype(int).to_numpy()
            runs = innings["runs_in_over"].astype(int).to_numpy()
            failures += int(np.any(scores[1:] != scores[:-1] + runs[:-1]))
    return failures


def train_and_backtest(project_root: Path, output_dir: Path) -> dict[str, Any]:
    data = pd.read_csv(project_root / "data/candidates/v3/verified_training_overs.csv")
    data["match_date"] = pd.to_datetime(data["match_date"])
    train = data[data["match_date"].dt.year <= 2023].reset_index(drop=True)
    calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
    holdout = data[data["match_date"].dt.year >= 2025].reset_index(drop=True)
    target = np.minimum(train["runs_in_over"].to_numpy(), MAX_RUN_CLASS)

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=MAX_RUN_CLASS + 1,
        n_estimators=60,
        learning_rate=0.08,
        num_leaves=25,
        max_depth=6,
        min_child_samples=100,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        _features(train, V3_FEATURES),
        target,
        categorical_feature=CATEGORICAL_FEATURES,
    )
    calibration_probability = model.predict_proba(_features(calibration, V3_FEATURES))
    holdout_probability = model.predict_proba(_features(holdout, V3_FEATURES))
    if list(model.classes_) != list(range(MAX_RUN_CLASS + 1)):
        raise ValueError("Sharp-range model did not learn every required run class.")

    locked_dir = project_root / "models/locked/cricketbaba_candidate_v3"
    feature_columns: list[str] = joblib.load(locked_dir / "feature_cols.pkl")
    categories: list[str] = joblib.load(locked_dir / "cat_cols.pkl")
    x_holdout = holdout[feature_columns].copy()
    for column in categories:
        x_holdout[column] = x_holdout[column].astype("category")
    point_prediction = joblib.load(locked_dir / "runs_model.pkl").predict(x_holdout)

    actual = holdout["runs_in_over"].to_numpy()
    clipped_actual = np.minimum(actual, MAX_RUN_CLASS)
    results: dict[str, Any] = {}
    prediction_columns: dict[str, np.ndarray] = {}
    for width in (1, 2):
        calibration_low, calibration_high, calibration_band_probability = _best_bands(
            calibration_probability, width
        )
        low, high, band_probability = _best_bands(holdout_probability, width)
        naive_low, naive_high = _naive_bands(point_prediction, width)
        hit = (clipped_actual >= low) & (clipped_actual <= high)
        naive_hit = (actual >= naive_low) & (actual <= naive_high)
        calibration_actual = np.minimum(
            calibration["runs_in_over"].to_numpy(), MAX_RUN_CLASS
        )
        calibration_hit = (calibration_actual >= calibration_low) & (
            calibration_actual <= calibration_high
        )
        results[f"width_{width}"] = {
            "display_example": "8-9" if width == 1 else "8-10",
            "calibration_hit_rate": float(np.mean(calibration_hit)),
            "holdout_hit_rate": float(np.mean(hit)),
            "naive_point_centered_hit_rate": float(np.mean(naive_hit)),
            "relative_improvement_over_naive": float(
                (np.mean(hit) - np.mean(naive_hit)) / np.mean(naive_hit)
            ),
            "mean_selected_band_probability": float(np.mean(band_probability)),
            "mean_calibration_band_probability": float(
                np.mean(calibration_band_probability)
            ),
            "phase": _phase_report(holdout, clipped_actual, low, high),
        }
        prediction_columns[f"sharp_{width}_low"] = low
        prediction_columns[f"sharp_{width}_high"] = high
        prediction_columns[f"sharp_{width}_probability"] = band_probability
        prediction_columns[f"sharp_{width}_hit"] = hit

    sequence_failures = _sequence_failures(holdout)
    width_two = results["width_2"]
    gates = {
        "two_run_width_is_exact": True,
        "two_run_hit_rate_beats_naive": width_two["holdout_hit_rate"]
        > width_two["naive_point_centered_hit_rate"],
        "two_run_holdout_hit_rate_at_least_25pct": width_two["holdout_hit_rate"]
        >= 0.25,
        "calibration_to_holdout_drop_below_3pct": width_two["calibration_hit_rate"]
        - width_two["holdout_hit_rate"]
        < 0.03,
        "complete_match_sequence_failures_zero": sequence_failures == 0,
        "complete_match_sample_at_least_100": holdout["source_file"].nunique() >= 100,
    }
    report = {
        "candidate_version": "v3.2_two_run_sharp_range",
        "purpose": "narrow_range_only_confidence_unchanged",
        "model_scope": "candidate_only",
        "production_models_changed": False,
        "range_definition": (
            "Inclusive integer band. Width two means, for example, 8-10 runs."
        ),
        "tail_definition": f"Class {MAX_RUN_CLASS} represents {MAX_RUN_CLASS}+ runs.",
        "split": {
            "train_through_2023": len(train),
            "calibration_2024": len(calibration),
            "holdout_2025_plus": len(holdout),
            "complete_holdout_matches": int(holdout["source_file"].nunique()),
        },
        "sharp_range_results": results,
        "complete_match_test": {
            "matches_tested": int(holdout["source_file"].nunique()),
            "innings_tested": int(
                holdout[["source_file", "innings"]].drop_duplicates().shape[0]
            ),
            "overs_tested": len(holdout),
            "sequence_failures": sequence_failures,
        },
        "promotion_gates": gates,
        "ready_for_shadow": all(gates.values()),
        "confidence_changed": False,
        "safety_range_changed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "sharp_range_model.pkl")
    joblib.dump(V3_FEATURES, output_dir / "feature_cols.pkl")
    joblib.dump(CATEGORICAL_FEATURES, output_dir / "cat_cols.pkl")
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    predictions = holdout[
        ["source_file", "match_date", "innings", "over", "phase", "runs_in_over"]
    ].copy()
    for name, values in prediction_columns.items():
        predictions[name] = values
    predictions.to_csv(output_dir / "holdout_sharp_ranges.csv", index=False)
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            train_and_backtest(root, root / "models/candidates/v3.2_sharp_range"),
            indent=2,
        )
    )
