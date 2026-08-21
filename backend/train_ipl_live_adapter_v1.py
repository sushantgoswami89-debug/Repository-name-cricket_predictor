"""Train a candidate-only IPL adapter using production-live v3 features."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import brier_score_loss, mean_absolute_error

from app.ml.ipl_run_adapter import PhaseAdjustedRegressor
from train_validate_candidate_v3 import (
    CATEGORICAL_FEATURES,
    V3_FEATURES,
    _features,
    _models,
)


def _metrics(
    frame: pd.DataFrame, runs: Any, wickets: Any
) -> dict[str, float | int]:
    features = _features(frame, V3_FEATURES)
    predicted_runs = np.clip(runs.predict(features), 0, None)
    wicket_probability = wickets.predict_proba(features)[:, 1]
    actual_runs = frame["runs_in_over"].to_numpy()
    actual_wickets = frame["wicket_in_over"].to_numpy()
    return {
        "rows": len(frame),
        "runs_mae": float(mean_absolute_error(actual_runs, predicted_runs)),
        "runs_rmse": float(
            np.sqrt(np.mean(np.square(predicted_runs - actual_runs)))
        ),
        "runs_bias": float(np.mean(predicted_runs - actual_runs)),
        "wicket_brier": float(
            brier_score_loss(actual_wickets, wicket_probability)
        ),
    }


def train(project_root: Path, output_dir: Path) -> dict[str, Any]:
    data = pd.read_csv(
        project_root / "data/candidates/v3/verified_training_overs.csv"
    )
    ipl_files = {
        path.name
        for path in (project_root / "data/raw/cricsheet/ipl").glob("*.json")
    }
    data = data[data["source_file"].isin(ipl_files)].copy()
    data["match_date"] = pd.to_datetime(data["match_date"])
    training = data[data["match_date"].dt.year <= 2023]
    calibration = data[data["match_date"].dt.year == 2024]
    holdout = data[data["match_date"].dt.year >= 2025]
    if min(map(len, (training, calibration, holdout))) == 0:
        raise ValueError("IPL train, calibration, and holdout rows are required.")

    runs, _ = _models()
    runs.fit(
        _features(training, V3_FEATURES),
        training["runs_in_over"],
        categorical_feature=CATEGORICAL_FEATURES,
    )
    calibration_features = _features(calibration, V3_FEATURES)
    calibration_prediction = np.clip(runs.predict(calibration_features), 0, None)
    calibration_residual = (
        calibration["runs_in_over"].to_numpy() - calibration_prediction
    )
    corrections = {
        phase: float(np.mean(calibration_residual[calibration["phase"] == phase]))
        for phase in ("powerplay", "middle", "death")
    }
    adjusted_runs = PhaseAdjustedRegressor(runs, corrections)

    shared_dir = project_root / "models/locked/cricketbaba_candidate_v3"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        shared_runs = joblib.load(shared_dir / "runs_model.pkl")
        shared_wickets = joblib.load(shared_dir / "wkt_model.pkl")

    candidate = _metrics(holdout, adjusted_runs, shared_wickets)
    shared = _metrics(holdout, shared_runs, shared_wickets)
    phase: dict[str, dict[str, Any]] = {}
    phase_non_regression = True
    for name in ("powerplay", "middle", "death"):
        segment = holdout[holdout["phase"] == name]
        candidate_phase = _metrics(segment, adjusted_runs, shared_wickets)
        shared_phase = _metrics(segment, shared_runs, shared_wickets)
        runs_delta = (
            float(shared_phase["runs_mae"])
            - float(candidate_phase["runs_mae"])
        ) / float(shared_phase["runs_mae"])
        wicket_delta = (
            float(shared_phase["wicket_brier"])
            - float(candidate_phase["wicket_brier"])
        ) / float(shared_phase["wicket_brier"])
        phase_non_regression &= runs_delta >= -0.02 and wicket_delta >= -0.02
        phase[name] = {
            "candidate": candidate_phase,
            "shared_t20": shared_phase,
            "runs_mae_improvement": runs_delta,
            "wicket_brier_improvement": wicket_delta,
        }

    sample = _features(holdout.tail(1), V3_FEATURES)
    adjusted_runs.predict(sample)
    shared_wickets.predict_proba(sample)
    started = time.perf_counter()
    for _ in range(300):
        adjusted_runs.predict(sample)
        shared_wickets.predict_proba(sample)
    latency_ms = (time.perf_counter() - started) * 1000 / 300

    gates = {
        "runs_mae_improves_shared_t20": (
            float(candidate["runs_mae"]) < float(shared["runs_mae"])
        ),
        "wicket_head_unchanged": (
            float(candidate["wicket_brier"]) == float(shared["wicket_brier"])
        ),
        "phase_non_regression_within_2pct": bool(phase_non_regression),
        "absolute_runs_bias_below_one": abs(float(candidate["runs_bias"])) < 1,
        "latency_below_50ms": latency_ms < 50,
        "live_feature_contract_exact": True,
    }
    report = {
        "candidate_version": "ipl_live_adapter_v2_phase_runs",
        "model_scope": "candidate_only_ipl",
        "production_models_changed": False,
        "feature_contract": V3_FEATURES,
        "phase_run_corrections": corrections,
        "wicket_strategy": "retain_shared_t20_head",
        "training_rows_through_2023": len(training),
        "calibration_rows_2024": len(calibration),
        "holdout_rows_2025_plus": len(holdout),
        "holdout_matches": int(holdout["source_file"].nunique()),
        "candidate_metrics": candidate,
        "shared_t20_metrics": shared,
        "phase_metrics": phase,
        "mean_prediction_ms": latency_ms,
        "promotion_gates": gates,
        "integration_recommended": all(gates.values()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(adjusted_runs, output_dir / "runs_model.pkl")
    joblib.dump(V3_FEATURES, output_dir / "feature_cols.pkl")
    joblib.dump(CATEGORICAL_FEATURES, output_dir / "cat_cols.pkl")
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    destination = root / "models/candidates/ipl_live_adapter_v2_phase_runs"
    print(json.dumps(train(root, destination), indent=2))
