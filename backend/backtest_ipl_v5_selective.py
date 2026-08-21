"""Calibrate frozen IPL v5 on 2025 and test selective bands on 2026."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from train_ipl_phase_intelligence_v5 import _load, _x
from train_sharp_range_candidate import MAX_RUN_CLASS, _best_bands

COVERAGE_TARGETS = [1.0, 0.75, 0.50, 0.30, 0.20]


def _metrics(
    hit: np.ndarray,
    calibrated: np.ndarray,
    wicket_actual: np.ndarray,
    wicket_probability: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    kept = score >= threshold
    return {
        "rows": int(kept.sum()),
        "coverage": float(kept.mean()),
        "band_accuracy": float(hit[kept].mean()),
        "mean_displayed_confidence": float(calibrated[kept].mean()),
        "calibration_gap": float(abs(calibrated[kept].mean() - hit[kept].mean())),
        "conditional_wicket_brier": float(
            brier_score_loss(wicket_actual[kept], wicket_probability[kept])
        ),
    }


def backtest(root: Path, output_dir: Path) -> dict[str, Any]:
    data, _ = _load(root)
    holdout = data[data["match_date"].dt.year >= 2025].reset_index(drop=True)
    model_dir = root / "models/candidates/ipl_engine_v5_phase_intelligence"
    columns = joblib.load(model_dir / "feature_cols.pkl")
    probability = np.zeros((len(holdout), MAX_RUN_CLASS + 1))
    wicket_probability = np.zeros(len(holdout))
    models: dict[str, Any] = {}
    for phase in ("powerplay", "middle", "death"):
        mask = holdout["phase"] == phase
        run_model = joblib.load(model_dir / f"{phase}_runs_model.pkl")
        wicket_model = joblib.load(model_dir / f"{phase}_wicket_model.pkl")
        phase_probability = run_model.predict_proba(
            _x(holdout.loc[mask], columns)
        )
        probability[
            np.ix_(np.flatnonzero(mask.to_numpy()), run_model.classes_.astype(int))
        ] = phase_probability
        wicket_probability[mask] = wicket_model.predict_proba(
            _x(holdout.loc[mask], columns)
        )[:, 1]
        models[f"{phase}_runs"] = run_model
        models[f"{phase}_wicket"] = wicket_model
    low, high, raw_mass = _best_bands(probability, 2)
    actual = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
    hit = (actual >= low) & (actual <= high)
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=1)
    year = holdout["match_date"].dt.year.to_numpy()
    calibration_mask = year == 2025
    test_mask = year == 2026
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        raw_mass[calibration_mask], hit[calibration_mask].astype(int)
    )
    calibrated = calibrator.predict(raw_mass)
    scores = {
        "raw_band_mass": raw_mass,
        "negative_entropy": -entropy,
        "calibrated_hit_probability": calibrated,
    }
    candidates: dict[str, Any] = {}
    wicket_actual = holdout["wicket_in_over"].astype(int).to_numpy()
    for score_name, score in scores.items():
        coverage_report: dict[str, Any] = {}
        for target in COVERAGE_TARGETS:
            threshold = (
                float(np.min(score[calibration_mask]) - 1.0)
                if target == 1.0
                else float(np.quantile(score[calibration_mask], 1.0 - target))
            )
            coverage_report[f"{int(target * 100)}pct"] = {
                "threshold_selected_on_2025": threshold,
                "validation_2025": _metrics(
                    hit[calibration_mask],
                    calibrated[calibration_mask],
                    wicket_actual[calibration_mask],
                    wicket_probability[calibration_mask],
                    score[calibration_mask],
                    threshold,
                ),
                "test_2026": _metrics(
                    hit[test_mask],
                    calibrated[test_mask],
                    wicket_actual[test_mask],
                    wicket_probability[test_mask],
                    score[test_mask],
                    threshold,
                ),
            }
        candidates[score_name] = coverage_report
    sample = holdout.tail(1)
    phase = str(sample.iloc[0]["phase"])
    started = time.perf_counter()
    for _ in range(300):
        run_probability = models[f"{phase}_runs"].predict_proba(_x(sample, columns))
        models[f"{phase}_wicket"].predict_proba(_x(sample, columns))
        sample_mass = _best_bands(run_probability, 2)[2]
        calibrator.predict(sample_mass)
    latency = (time.perf_counter() - started) * 1000 / 300
    test_50 = candidates["calibrated_hit_probability"]["50pct"]["test_2026"]
    gates = {
        "accuracy_at_least_32_38pct_at_50pct_coverage": bool(
            test_50["band_accuracy"] >= 0.3238 and test_50["coverage"] >= 0.45
        ),
        "calibration_gap_below_5pct_at_50pct_coverage": bool(
            test_50["calibration_gap"] < 0.05
        ),
        "conditional_wicket_brier_below_v5": bool(
            test_50["conditional_wicket_brier"] < 0.19519474572613377
        ),
        "latency_below_50ms": bool(latency < 50),
    }
    report = {
        "candidate_version": "ipl_v5_selective_calibration",
        "production_models_changed": False,
        "frozen_model": "ipl_engine_v5_phase_intelligence",
        "calibration_season": 2025,
        "untouched_test_season": 2026,
        "calibration_rows": int(calibration_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "selection_results": candidates,
        "mean_prediction_ms": latency,
        "promotion_gates": gates,
        "integration_recommended": all(gates.values()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, output_dir / "band_hit_isotonic_calibrator.pkl")
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    destination = project_root / "models/candidates/ipl_v5_selective"
    print(json.dumps(backtest(project_root, destination), indent=2))
