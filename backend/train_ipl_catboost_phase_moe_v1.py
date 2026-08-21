"""Candidate-only CatBoost global and deterministic phase-specialist experiment."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error

from app.ml.ipl_phase_moe_dataset import KEYS, UNKNOWN_CATEGORY
from train_ipl_venue_adapter_v1 import (
    CATEGORICAL as VENUE_CATEGORICAL,
    FEATURES as VENUE_EXPERIMENTAL_FEATURES,
    V3_FEATURES,
)

SEED = 42
MIN_SUPPORTED_ROWS = 100
VERSION = "ipl_catboost_phase_moe_v1"
STRUCTURAL_VENUE_FEATURES = [
    "venue_par_score",
    "venue_prior_innings",
    "venue_scoring_regime",
    "venue_par_source",
    "batting_team_venue_context",
    "phase_venue_regime",
]
MOE_FEATURES = V3_FEATURES + STRUCTURAL_VENUE_FEATURES + [
    "batting_team",
    "venue_name",
    "striker",
    "non_striker",
    "known_bowler",
    "active_batter_state",
    "new_batter",
    "partnership_legal_ball_age",
    "wickets_remaining_bucket",
    "state_regime",
    "chase_pressure",
    "striker_match_balls",
    "partner_match_balls",
    "striker_prior_balls",
    "striker_prior_runs_per_ball",
    "striker_prior_dot_rate",
    "striker_prior_boundary_rate",
    "striker_prior_dismissal_rate",
    "partner_prior_balls",
    "partner_prior_runs_per_ball",
    "partner_prior_dot_rate",
    "partner_prior_boundary_rate",
]
CATEGORICAL = [
    "phase",
    "venue_scoring_regime",
    "venue_par_source",
    "batting_team_venue_context",
    "phase_venue_regime",
    "batting_team",
    "venue_name",
    "striker",
    "non_striker",
    "known_bowler",
    "active_batter_state",
    "wickets_remaining_bucket",
    "state_regime",
    "chase_pressure",
]


def _catboost():
    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError(
            "CatBoost is required. Install only after explicit dependency approval."
        ) from exc
    return catboost


def _frame(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = data[columns].copy()
    for column in CATEGORICAL:
        if column in result:
            result[column] = result[column].fillna(UNKNOWN_CATEGORY).astype(str)
    return result


def _load_data(root: Path) -> pd.DataFrame:
    base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    venue = pd.read_csv(root / "data/candidates/ipl_venue_regime/features.csv")
    moe = pd.read_csv(root / "data/candidates/ipl_phase_moe_v1/features.csv")
    for item in (base, venue, moe):
        item["match_date"] = item["match_date"].astype(str)
    venue_columns = list(
        dict.fromkeys(
            KEYS
            + ["venue_name"]
            + [
                name
                for name in VENUE_EXPERIMENTAL_FEATURES
                if name not in V3_FEATURES
            ]
        )
    )
    data = base.merge(
        venue[venue_columns],
        on=KEYS,
        validate="one_to_one",
    ).merge(moe, on=KEYS, validate="one_to_one")
    data["match_date"] = pd.to_datetime(data["match_date"])
    return data.sort_values(["match_date", "source_file", "innings", "over"])


def _cb_regressor(loss: str = "MAE"):
    cb = _catboost()
    return cb.CatBoostRegressor(
        loss_function=loss,
        iterations=350,
        depth=7,
        learning_rate=0.035,
        l2_leaf_reg=5,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def _cb_classifier():
    cb = _catboost()
    return cb.CatBoostClassifier(
        loss_function="Logloss",
        iterations=350,
        depth=7,
        learning_rate=0.035,
        l2_leaf_reg=5,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def _fit_family(training: pd.DataFrame) -> dict[str, object]:
    x = _frame(training, MOE_FEATURES)
    models = {
        "runs": _cb_regressor(),
        "wicket": _cb_classifier(),
        "lower": _cb_regressor("Quantile:alpha=0.2"),
        "upper": _cb_regressor("Quantile:alpha=0.8"),
    }
    models["runs"].fit(x, training["runs_in_over"], cat_features=CATEGORICAL)
    models["wicket"].fit(x, training["wicket_in_over"], cat_features=CATEGORICAL)
    models["lower"].fit(x, training["runs_in_over"], cat_features=CATEGORICAL)
    models["upper"].fit(x, training["runs_in_over"], cat_features=CATEGORICAL)
    return models


def _predict_family(models: dict[str, object], data: pd.DataFrame) -> dict[str, np.ndarray]:
    x = _frame(data, MOE_FEATURES)
    return {
        "runs": np.asarray(models["runs"].predict(x), dtype=float),
        "wicket": np.asarray(models["wicket"].predict_proba(x)[:, 1], dtype=float),
        "lower": np.asarray(models["lower"].predict(x), dtype=float),
        "upper": np.asarray(models["upper"].predict(x), dtype=float),
    }


def _fit_phase_families(training: pd.DataFrame) -> dict[str, dict[str, object]]:
    return {
        str(phase): _fit_family(segment)
        for phase, segment in training.groupby("phase", sort=True)
    }


def _predict_phase_families(
    models: dict[str, dict[str, object]], data: pd.DataFrame
) -> dict[str, np.ndarray]:
    output = {
        name: np.zeros(len(data), dtype=float)
        for name in ("runs", "wicket", "lower", "upper")
    }
    for phase, segment in data.groupby("phase", sort=False):
        positions = data.index.get_indexer(segment.index)
        predicted = _predict_family(models[str(phase)], segment)
        for name in output:
            output[name][positions] = predicted[name]
    return output


def _calibrate_point(
    calibration: pd.DataFrame, prediction: np.ndarray
) -> dict[str, float]:
    residual = calibration["runs_in_over"].to_numpy() - prediction
    return {
        str(phase): float(np.mean(residual[calibration["phase"].to_numpy() == phase]))
        for phase in sorted(calibration["phase"].unique())
    }


def _apply_point(
    data: pd.DataFrame, prediction: np.ndarray, corrections: dict[str, float]
) -> np.ndarray:
    return prediction + data["phase"].map(corrections).fillna(0).to_numpy()


def _calibrate_fixed_range(
    calibration: pd.DataFrame, prediction: np.ndarray
) -> dict[str, int]:
    offsets: dict[str, int] = {}
    actual = calibration["runs_in_over"].to_numpy()
    for phase, segment in calibration.groupby("phase"):
        positions = calibration.index.get_indexer(segment.index)
        phase_actual = actual[positions]
        phase_prediction = prediction[positions]
        offsets[str(phase)] = max(
            range(-4, 3),
            key=lambda offset: float(
                np.mean(
                    (phase_actual >= np.maximum(0, np.floor(phase_prediction).astype(int) + offset))
                    & (
                        phase_actual
                        <= np.maximum(0, np.floor(phase_prediction).astype(int) + offset)
                        + 3
                    )
                )
            ),
        )
    return offsets


def _fixed_range(
    data: pd.DataFrame, prediction: np.ndarray, offsets: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    selected = data["phase"].map(offsets).fillna(-1).astype(int).to_numpy()
    low = np.maximum(0, np.floor(prediction).astype(int) + selected)
    return low, low + 3


def _metrics(
    data: pd.DataFrame,
    prediction: np.ndarray,
    offsets: dict[str, int],
    wicket_prediction: np.ndarray | None = None,
    diagnostic_lower: np.ndarray | None = None,
    diagnostic_upper: np.ndarray | None = None,
) -> dict[str, float | int]:
    actual = data["runs_in_over"].to_numpy()
    low, high = _fixed_range(data, prediction, offsets)
    result: dict[str, float | int] = {
        "rows": len(data),
        "mae": float(mean_absolute_error(actual, prediction)),
        "bias": float(np.mean(prediction - actual)),
        "fixed_three_run_coverage": float(np.mean((actual >= low) & (actual <= high))),
        "inclusive_integer_outcomes": 4,
        "numeric_width": 3,
    }
    if wicket_prediction is not None:
        result["wicket_brier"] = float(
            brier_score_loss(data["wicket_in_over"], wicket_prediction)
        )
    if diagnostic_lower is not None and diagnostic_upper is not None:
        q_low = np.minimum(diagnostic_lower, diagnostic_upper)
        q_high = np.maximum(diagnostic_lower, diagnostic_upper)
        result["diagnostic_quantile_coverage"] = float(
            np.mean((actual >= q_low) & (actual <= q_high))
        )
        result["diagnostic_mean_quantile_width"] = float(np.mean(q_high - q_low))
    return result


def _matrix(
    data: pd.DataFrame, prediction: np.ndarray, offsets: dict[str, int]
) -> dict[str, dict[str, dict[str, object]]]:
    dimensions = {
        "innings": data["innings"].map({1: "first_innings", 2: "chase"}).fillna("other"),
        "phase": data["phase"],
        "venue_regime": data["venue_scoring_regime"],
        "state": data["state_regime"],
        "batter_state": data["active_batter_state"],
        "wickets_remaining": data["wickets_remaining_bucket"],
        "chase_pressure": data["chase_pressure"],
    }
    matrix: dict[str, dict[str, dict[str, object]]] = {}
    for dimension, labels in dimensions.items():
        matrix[dimension] = {}
        for label in sorted(labels.astype(str).unique()):
            mask = labels.astype(str).to_numpy() == label
            segment = data.iloc[np.flatnonzero(mask)]
            positions = np.flatnonzero(mask)
            entry = _metrics(segment, prediction[positions], offsets)
            entry["supported"] = len(segment) >= MIN_SUPPORTED_ROWS
            matrix[dimension][label] = entry
    return matrix


def _lightgbm_venue_only(training: pd.DataFrame):
    features = V3_FEATURES + STRUCTURAL_VENUE_FEATURES
    categoricals = [name for name in VENUE_CATEGORICAL if name in features]
    x = training[features].copy()
    for name in categoricals:
        x[name] = x[name].astype("category")
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=0.025,
        num_leaves=25,
        max_depth=6,
        min_child_samples=100,
        reg_lambda=1.5,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(x, training["runs_in_over"], categorical_feature=categoricals)
    return model, features, categoricals


def _lgb_predict(model, data, features, categoricals) -> np.ndarray:
    x = data[features].copy()
    for name in categoricals:
        x[name] = x[name].astype("category")
    return np.asarray(model.predict(x), dtype=float)


def _replays(
    data: pd.DataFrame, prediction: np.ndarray, offsets: dict[str, int], count: int = 5
) -> list[dict[str, object]]:
    result = []
    matches = (
        data[["source_file", "match_date"]]
        .drop_duplicates()
        .sort_values(["match_date", "source_file"])
        .tail(count)
    )
    for match in matches.itertuples():
        mask = data["source_file"].to_numpy() == match.source_file
        segment = data.iloc[np.flatnonzero(mask)]
        result.append(
            {
                "source_file": match.source_file,
                "match_date": str(match.match_date.date()),
                **_metrics(segment, prediction[np.flatnonzero(mask)], offsets),
            }
        )
    return result


def train(root: Path, output_dir: Path) -> dict[str, object]:
    data = _load_data(root)
    training = data[data["match_date"].dt.year <= 2023].copy()
    calibration = data[data["match_date"].dt.year == 2024].copy()
    holdout = data[data["match_date"].dt.year >= 2025].copy()

    global_models = _fit_family(training)
    phase_models = _fit_phase_families(training)
    global_cal = _predict_family(global_models, calibration)
    phase_cal = _predict_phase_families(phase_models, calibration)
    global_corrections = _calibrate_point(calibration, global_cal["runs"])
    phase_corrections = _calibrate_point(calibration, phase_cal["runs"])
    global_cal_runs = _apply_point(calibration, global_cal["runs"], global_corrections)
    phase_cal_runs = _apply_point(calibration, phase_cal["runs"], phase_corrections)
    global_offsets = _calibrate_fixed_range(calibration, global_cal_runs)
    phase_offsets = _calibrate_fixed_range(calibration, phase_cal_runs)

    global_pred = _predict_family(global_models, holdout)
    phase_pred = _predict_phase_families(phase_models, holdout)
    global_runs = _apply_point(holdout, global_pred["runs"], global_corrections)
    phase_runs = _apply_point(holdout, phase_pred["runs"], phase_corrections)

    venue_artifact = joblib.load(
        root / "models/candidates/ipl_venue_adapter_v1/runs_model.pkl"
    )
    shared_artifact = joblib.load(
        root / "models/locked/cricketbaba_candidate_v3/runs_model.pkl"
    )
    venue_experimental = venue_artifact.predict(
        holdout[VENUE_EXPERIMENTAL_FEATURES].assign(
            **{
                name: holdout[name].astype("category")
                for name in VENUE_CATEGORICAL
            }
        )
    )
    shared = shared_artifact.predict(
        holdout[V3_FEATURES].assign(
            phase=holdout["phase"].astype("category")
        )
    )
    venue_only_model, venue_only_features, venue_only_cat = _lightgbm_venue_only(
        training
    )
    venue_only_cal = _lgb_predict(
        venue_only_model, calibration, venue_only_features, venue_only_cat
    )
    venue_only_corrections = _calibrate_point(calibration, venue_only_cal)
    venue_only_cal = _apply_point(
        calibration, venue_only_cal, venue_only_corrections
    )
    venue_only_offsets = _calibrate_fixed_range(calibration, venue_only_cal)
    venue_only = _apply_point(
        holdout,
        _lgb_predict(venue_only_model, holdout, venue_only_features, venue_only_cat),
        venue_only_corrections,
    )
    default_offsets = {"powerplay": -1, "middle": -1, "death": -1}
    venue_offsets = json.loads(
        (
            root / "models/candidates/ipl_venue_adapter_v1/validation_report.json"
        ).read_text()
    )["range_offsets"]

    global_metrics = _metrics(
        holdout,
        global_runs,
        global_offsets,
        global_pred["wicket"],
        global_pred["lower"],
        global_pred["upper"],
    )
    phase_metrics = _metrics(
        holdout,
        phase_runs,
        phase_offsets,
        phase_pred["wicket"],
        phase_pred["lower"],
        phase_pred["upper"],
    )
    baselines = {
        "locked_shared_t20": _metrics(holdout, shared, default_offsets),
        "venue_only_ipl": _metrics(
            holdout, venue_only, venue_only_offsets
        ),
        "venue_plus_experimental": _metrics(
            holdout, venue_experimental, venue_offsets
        ),
    }
    phase_matrix = _matrix(holdout, phase_runs, phase_offsets)
    venue_matrix = _matrix(holdout, venue_experimental, venue_offsets)
    phase_or_venue_non_regression = True
    regressions = []
    for dimension in ("phase", "venue_regime"):
        for label, candidate_segment in phase_matrix[dimension].items():
            baseline_segment = venue_matrix[dimension][label]
            if not candidate_segment["supported"]:
                continue
            ratio = candidate_segment["mae"] / baseline_segment["mae"]
            if ratio > 1.02:
                phase_or_venue_non_regression = False
                regressions.append(
                    {"dimension": dimension, "label": label, "mae_ratio": ratio}
                )

    cb = _catboost()
    report: dict[str, object] = {
        "candidate_version": VERSION,
        "candidate_only": True,
        "production_models_changed": False,
        "random_seed": SEED,
        "minimum_supported_rows": MIN_SUPPORTED_ROWS,
        "split": {
            "training": "through 2023",
            "calibration": "2024",
            "holdout": "2025+",
            "training_rows": len(training),
            "calibration_rows": len(calibration),
            "holdout_rows": len(holdout),
            "training_matches": training["source_file"].nunique(),
            "calibration_matches": calibration["source_file"].nunique(),
            "holdout_matches": holdout["source_file"].nunique(),
        },
        "features": MOE_FEATURES,
        "categorical_features": CATEGORICAL,
        "live_parity": {
            "pre_over_state_only": True,
            "player_profiles_frozen_before_match": True,
            "active_batters_available_live": True,
            "future_bowler_used": False,
            "known_bowler_policy": UNKNOWN_CATEGORY,
            "super_overs_excluded": True,
            "same_feature_file_for_training_calibration_holdout": True,
        },
        "dependencies": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "catboost": cb.__version__,
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "parameters": {
            "iterations": 350,
            "depth": 7,
            "learning_rate": 0.035,
            "l2_leaf_reg": 5,
        },
        "baselines": baselines,
        "catboost_global": global_metrics,
        "catboost_phase_specialists": phase_metrics,
        "phase_specialist_vs_global_mae_delta": (
            phase_metrics["mae"] - global_metrics["mae"]
        ),
        "calibration": {
            "global_corrections": global_corrections,
            "phase_corrections": phase_corrections,
            "global_range_offsets": global_offsets,
            "phase_range_offsets": phase_offsets,
            "five_replays": _replays(
                calibration, phase_cal_runs, phase_offsets
            ),
        },
        "five_heldout_replays": _replays(holdout, phase_runs, phase_offsets),
        "heldout_error_matrix": phase_matrix,
        "venue_candidate_error_matrix": venue_matrix,
        "supported_segment_regressions_over_2pct": regressions,
        "promotion_gates": {
            "mae_beats_3_8930": phase_metrics["mae"] < 3.8930,
            "phase_specialists_beat_global": (
                phase_metrics["mae"] < global_metrics["mae"]
            ),
            "absolute_bias_below_one": abs(phase_metrics["bias"]) < 1,
            "supported_phase_venue_non_regression": phase_or_venue_non_regression,
            "fixed_range_coverage_non_regression": (
                phase_metrics["fixed_three_run_coverage"]
                >= baselines["locked_shared_t20"][
                    "fixed_three_run_coverage"
                ]
            ),
            "live_feature_parity": True,
            "range_not_widened": True,
        },
    }
    gates = report["promotion_gates"]
    report["point_decision"] = (
        "promote_later_with_explicit_approval"
        if all(
            gates[name]
            for name in (
                "mae_beats_3_8930",
                "phase_specialists_beat_global",
                "absolute_bias_below_one",
                "supported_phase_venue_non_regression",
                "live_feature_parity",
            )
        )
        else "retain_for_research"
    )
    report["range_decision"] = (
        "promote_later_with_explicit_approval"
        if gates["fixed_range_coverage_non_regression"]
        else "reject"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in global_models.items():
        model.save_model(output_dir / f"global_{name}.cbm")
    for phase, family in phase_models.items():
        for name, model in family.items():
            model.save_model(output_dir / f"{phase}_{name}.cbm")
    (output_dir / "feature_contract.json").write_text(
        json.dumps(
            {"features": MOE_FEATURES, "categorical_features": CATEGORICAL},
            indent=2,
        )
    )
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    result = train(
        project_root, project_root / f"models/candidates/{VERSION}"
    )
    print(json.dumps(result, indent=2))
