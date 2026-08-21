"""Train the candidate-only global CatBoost model on canonical IPL v2."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_ipl_catboost_phase_moe_v1 import (
    CATEGORICAL,
    MOE_FEATURES,
    SEED,
    _apply_point,
    _calibrate_fixed_range,
    _calibrate_point,
    _fit_family,
    _matrix,
    _metrics,
    _predict_family,
    _replays,
)

VERSION = "ipl_catboost_global_canonical_v2"


def train(root: Path, output: Path) -> dict:
    data = pd.read_csv(
        root / "data/candidates/ipl_canonical_v2/training_overs.csv"
    )
    data["match_date"] = pd.to_datetime(data["match_date"])
    data = data.sort_values(["match_date", "source_file", "innings", "over"])
    training = data[data["match_date"].dt.year <= 2023].copy()
    calibration = data[data["match_date"].dt.year == 2024].copy()
    holdout = data[data["match_date"].dt.year >= 2025].copy()

    models = _fit_family(training)
    calibration_prediction = _predict_family(models, calibration)
    corrections = _calibrate_point(
        calibration, calibration_prediction["runs"]
    )
    calibration_runs = _apply_point(
        calibration, calibration_prediction["runs"], corrections
    )
    offsets = _calibrate_fixed_range(calibration, calibration_runs)

    prediction = _predict_family(models, holdout)
    runs = _apply_point(holdout, prediction["runs"], corrections)
    metrics = _metrics(
        holdout,
        runs,
        offsets,
        prediction["wicket"],
        prediction["lower"],
        prediction["upper"],
    )
    previous = json.loads(
        (
            root
            / "models/candidates/ipl_catboost_phase_moe_v1/validation_report.json"
        ).read_text()
    )
    report = {
        "candidate_version": VERSION,
        "candidate_only": True,
        "production_changed": False,
        "seed": SEED,
        "dataset": "ipl_canonical_v2",
        "rows": {
            "training": len(training),
            "calibration": len(calibration),
            "holdout": len(holdout),
        },
        "features": MOE_FEATURES,
        "categorical_features": CATEGORICAL,
        "metrics": metrics,
        "previous_global_metrics": previous["catboost_global"],
        "mae_change_vs_previous_global": (
            metrics["mae"] - previous["catboost_global"]["mae"]
        ),
        "coverage_change_vs_previous_global": (
            metrics["fixed_three_run_coverage"]
            - previous["catboost_global"]["fixed_three_run_coverage"]
        ),
        "beats_previous_global": (
            metrics["mae"] < previous["catboost_global"]["mae"]
        ),
        "beats_venue_candidate": metrics["mae"] < 3.893075851448594,
        "beats_locked_coverage": (
            metrics["fixed_three_run_coverage"] >= 0.3366212338593974
        ),
        "corrections": corrections,
        "range_offsets": offsets,
        "error_matrix": _matrix(holdout, runs, offsets),
        "five_calibration_replays": _replays(
            calibration, calibration_runs, offsets
        ),
        "five_holdout_replays": _replays(holdout, runs, offsets),
    }
    report["decision"] = (
        "retain_for_research"
        if not report["beats_previous_global"]
        else "advance_to_live_parity_validation"
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.save_model(output / f"global_{name}.cbm")
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    train(project_root, project_root / f"models/candidates/{VERSION}")
