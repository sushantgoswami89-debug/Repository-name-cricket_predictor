"""Train a candidate-only venue-aware IPL runs adapter."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import mean_absolute_error

from app.ml.ipl_run_adapter import PhaseAdjustedRegressor
from train_validate_candidate_v3 import V3_FEATURES

KEYS = ["source_file", "match_date", "innings", "over"]
VENUE_FEATURES = [
    "venue_par_score",
    "venue_prior_innings",
    "venue_scoring_regime",
    "venue_par_source",
    "batting_team_venue_context",
    "phase_venue_regime",
    "momentum_score",
    "current_pair_strike_rate",
    "opening_batters_prior_average",
    "opening_batters_prior_dismissals",
]
FEATURES = V3_FEATURES + VENUE_FEATURES
CATEGORICAL = [
    "phase",
    "venue_scoring_regime",
    "venue_par_source",
    "batting_team_venue_context",
    "phase_venue_regime",
]


def _x(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame[columns].copy()
    for column in CATEGORICAL:
        if column in result:
            result[column] = result[column].astype("category")
    return result


def _model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=0.025,
        num_leaves=25,
        max_depth=6,
        min_child_samples=100,
        reg_lambda=1.5,
        random_state=42,
        verbose=-1,
    )


def _ranges(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    offsets: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    selected = []
    for row in frame.itertuples():
        combined = f"{row.phase}|{row.venue_scoring_regime}"
        selected.append(
            (offsets or {}).get(combined, (offsets or {}).get(str(row.phase), -1))
        )
    low = np.maximum(
        0, np.floor(predicted).astype(int) + np.asarray(selected, dtype=int)
    )
    high = low + 3
    return low, high


def _report(
    frame: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    offsets: dict[str, int] | None = None,
) -> dict[str, float]:
    low, high = _ranges(frame, predicted, offsets)
    return {
        "runs_mae": float(mean_absolute_error(actual, predicted)),
        "runs_bias": float(np.mean(predicted - actual)),
        "three_run_range_coverage": float(
            np.mean((actual >= low) & (actual <= high))
        ),
        "average_range_width": float(np.mean(high - low)),
    }


def train(project_root: Path, output_dir: Path) -> dict:
    base = pd.read_csv(
        project_root / "data/candidates/v3/verified_training_overs.csv"
    )
    venue = pd.read_csv(
        project_root / "data/candidates/ipl_venue_regime/features.csv"
    )
    for frame in (base, venue):
        frame["match_date"] = frame["match_date"].astype(str)
    data = base.merge(venue[KEYS + VENUE_FEATURES], on=KEYS, validate="one_to_one")
    data["match_date"] = pd.to_datetime(data["match_date"])
    training = data[data["match_date"].dt.year <= 2023]
    calibration = data[data["match_date"].dt.year == 2024]
    holdout = data[data["match_date"].dt.year >= 2025]

    model = _model()
    model.fit(
        _x(training, FEATURES),
        training["runs_in_over"],
        categorical_feature=CATEGORICAL,
    )
    calibration_raw = model.predict(_x(calibration, FEATURES))
    residual = calibration["runs_in_over"].to_numpy() - calibration_raw
    residual_by_index = pd.Series(residual, index=calibration.index)
    corrections: dict[str, float] = {}
    phase_corrections: dict[str, float] = {}
    for phase, segment in calibration.groupby("phase"):
        phase_corrections[str(phase)] = float(
            residual_by_index.loc[segment.index].mean()
        )
    for (phase, regime), segment in calibration.groupby(
        ["phase", "venue_scoring_regime"]
    ):
        if len(segment) >= 100:
            corrections[f"{phase}|{regime}"] = float(
                residual_by_index.loc[segment.index].mean()
            )
    corrections.update(phase_corrections)
    adapter = PhaseAdjustedRegressor(
        model,
        corrections,
        correction_columns=("phase", "venue_scoring_regime"),
    )
    calibration_adjusted = adapter.predict(_x(calibration, FEATURES))
    range_offsets: dict[str, int] = {}
    for group_columns in (["phase"], ["phase", "venue_scoring_regime"]):
        for key, segment in calibration.groupby(group_columns):
            if len(group_columns) > 1 and len(segment) < 100:
                continue
            labels = key if isinstance(key, tuple) else (key,)
            name = "|".join(map(str, labels))
            positions = calibration.index.get_indexer(segment.index)
            actual_segment = segment["runs_in_over"].to_numpy()
            prediction_segment = calibration_adjusted[positions]
            range_offsets[name] = max(
                range(-3, 4),
                key=lambda offset: float(
                    np.mean(
                        (
                            actual_segment
                            >= np.maximum(
                                0,
                                np.floor(prediction_segment).astype(int) + offset,
                            )
                        )
                        & (
                            actual_segment
                            <= np.maximum(
                                0,
                                np.floor(prediction_segment).astype(int) + offset,
                            )
                            + 3
                        )
                    )
                ),
            )

    candidate_prediction = adapter.predict(_x(holdout, FEATURES))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        shared = joblib.load(
            project_root
            / "models/locked/cricketbaba_candidate_v3/runs_model.pkl"
        )
    shared_prediction = shared.predict(_x(holdout, V3_FEATURES))
    actual = holdout["runs_in_over"].to_numpy()
    candidate = _report(holdout, actual, candidate_prediction, range_offsets)
    baseline = _report(holdout, actual, shared_prediction)
    by_regime = {}
    non_regression = True
    for regime, segment in holdout.groupby("venue_scoring_regime"):
        indexes = holdout.index.get_indexer(segment.index)
        candidate_segment = _report(
            segment,
            actual[indexes],
            candidate_prediction[indexes],
            range_offsets,
        )
        baseline_segment = _report(
            segment, actual[indexes], shared_prediction[indexes]
        )
        non_regression &= (
            candidate_segment["runs_mae"] <= baseline_segment["runs_mae"] * 1.02
        )
        by_regime[str(regime)] = {
            "rows": len(segment),
            "candidate": candidate_segment,
            "shared_t20": baseline_segment,
        }
    gates = {
        "mae_improves": candidate["runs_mae"] < baseline["runs_mae"],
        "range_coverage_improves": (
            candidate["three_run_range_coverage"]
            > baseline["three_run_range_coverage"]
        ),
        "range_not_widened": candidate["average_range_width"] <= 3,
        "regime_non_regression_within_2pct": bool(non_regression),
        "absolute_bias_below_one": abs(candidate["runs_bias"]) < 1,
    }
    report = {
        "candidate_version": "ipl_venue_adapter_v1",
        "production_models_changed": False,
        "features": FEATURES,
        "candidate": candidate,
        "shared_t20": baseline,
        "by_regime": by_regime,
        "corrections": corrections,
        "range_offsets": range_offsets,
        "promotion_gates": gates,
        "integration_recommended": all(gates.values()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(adapter, output_dir / "runs_model.pkl")
    joblib.dump(FEATURES, output_dir / "feature_cols.pkl")
    joblib.dump(CATEGORICAL, output_dir / "cat_cols.pkl")
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            train(root, root / "models/candidates/ipl_venue_adapter_v1"),
            indent=2,
        )
    )
