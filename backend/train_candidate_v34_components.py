"""Train component distributions and convolve them for Candidate v3.4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from train_sharp_range_candidate import MAX_RUN_CLASS, _best_bands, _phase_report
from train_validate_candidate_v3 import CATEGORICAL_FEATURES, V3_FEATURES, _features

KEYS = ["source_file", "match_date", "innings", "over"]
COMPONENTS = {
    "boundary_runs": 24,
    "running_batter_runs": 12,
    "extras_runs": 8,
}


def _new_model(class_count: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=class_count,
        n_estimators=55,
        learning_rate=0.08,
        num_leaves=25,
        max_depth=6,
        min_child_samples=100,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )


def _convolve(probabilities: list[np.ndarray], values: list[np.ndarray]) -> np.ndarray:
    rows = len(probabilities[0])
    distribution = np.zeros((rows, MAX_RUN_CLASS + 1), dtype=np.float32)
    for first_index, first_value in enumerate(values[0]):
        for second_index, second_value in enumerate(values[1]):
            pair = probabilities[0][:, first_index] * probabilities[1][:, second_index]
            for third_index, third_value in enumerate(values[2]):
                total = min(
                    MAX_RUN_CLASS,
                    int(first_value + second_value + third_value),
                )
                distribution[:, total] += pair * probabilities[2][:, third_index]
    distribution /= distribution.sum(axis=1, keepdims=True)
    return distribution


def train(project_root: Path, output_dir: Path) -> dict[str, Any]:
    base = pd.read_csv(project_root / "data/candidates/v3/verified_training_overs.csv")
    sequence = pd.read_csv(
        project_root / "data/candidates/v3.3/sequence_response_features.csv"
    )
    components = pd.read_csv(project_root / "data/candidates/v3.4/over_components.csv")
    for frame in (base, sequence, components):
        frame["match_date"] = frame["match_date"].astype(str)
    data = base.merge(sequence, on=KEYS, validate="one_to_one").merge(
        components,
        on=KEYS,
        validate="one_to_one",
        suffixes=("", "_component"),
    )
    if len(data) != len(base):
        raise ValueError("Component rows do not reconcile with verified overs.")
    data["match_date"] = pd.to_datetime(data["match_date"])
    train_data = data[data["match_date"].dt.year <= 2023].reset_index(drop=True)
    holdout = data[data["match_date"].dt.year >= 2025].reset_index(drop=True)
    ignored = set(KEYS + ["striker", "bowler"])
    additions = [
        column
        for column in sequence.columns
        if column not in ignored
        and not column.startswith(("matchup_response_", "pressure_matchup_response_"))
    ]
    feature_columns = V3_FEATURES + additions
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities: list[np.ndarray] = []
    class_values: list[np.ndarray] = []
    component_reports: dict[str, Any] = {}

    for target, cap in COMPONENTS.items():
        train_target = np.minimum(train_data[target].to_numpy(), cap)
        classes = np.unique(train_target)
        encoded = np.searchsorted(classes, train_target)
        model = _new_model(len(classes))
        model.fit(
            _features(train_data, feature_columns),
            encoded,
            categorical_feature=CATEGORICAL_FEATURES,
        )
        probability = model.predict_proba(_features(holdout, feature_columns))
        probabilities.append(probability)
        class_values.append(classes)
        predicted = classes[np.argmax(probability, axis=1)]
        actual = np.minimum(holdout[target].to_numpy(), cap)
        component_reports[target] = {
            "classes": classes.astype(int).tolist(),
            "exact_accuracy": float(np.mean(predicted == actual)),
            "mae": float(np.mean(np.abs(predicted - actual))),
        }
        joblib.dump(model, output_dir / f"{target}_model.pkl")
        joblib.dump(classes, output_dir / f"{target}_classes.pkl")

    distribution = _convolve(probabilities, class_values)
    low, high, band_probability = _best_bands(distribution, 2)
    actual_runs = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
    hit = (actual_runs >= low) & (actual_runs <= high)
    hit_rate = float(hit.mean())
    phase = _phase_report(holdout, actual_runs, low, high)
    sequence_failures = 0
    for _, match in holdout.groupby("source_file"):
        for _, innings in match.groupby("innings"):
            innings = innings.sort_values("over")
            scores = innings["score_before_over"].to_numpy()
            runs = innings["runs_in_over"].to_numpy()
            sequence_failures += int(np.any(scores[1:] != scores[:-1] + runs[:-1]))
    gates = {
        "beats_v32": hit_rate > 0.31376664261821996,
        "beats_required_32_38pct": hit_rate >= 0.3238,
        "no_phase_regresses_over_2pct": all(
            phase[name]["hit_rate"] >= baseline - 0.02
            for name, baseline in {
                "powerplay": 0.2774744366127271,
                "middle": 0.34074231628203766,
                "death": 0.31140663342040686,
            }.items()
        ),
        "complete_match_sequence_failures_zero": sequence_failures == 0,
    }
    report = {
        "candidate_version": "v3.4_over_component_simulator",
        "model_scope": "candidate_only",
        "production_models_changed": False,
        "verified_rows": len(data),
        "holdout_rows": len(holdout),
        "holdout_matches": int(holdout["source_file"].nunique()),
        "components": component_reports,
        "sharp_range": {
            "width": 2,
            "hit_rate": hit_rate,
            "improvement_over_v32_percentage_points": (hit_rate - 0.31376664261821996)
            * 100,
            "mean_model_band_mass": float(band_probability.mean()),
            "phase": phase,
        },
        "complete_match_sequence_failures": sequence_failures,
        "promotion_gates": gates,
        "integration_recommended": all(gates.values()),
    }
    joblib.dump(feature_columns, output_dir / "feature_cols.pkl")
    joblib.dump(CATEGORICAL_FEATURES, output_dir / "cat_cols.pkl")
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            train(root, root / "models/candidates/v3.4_over_components"),
            indent=2,
        )
    )
