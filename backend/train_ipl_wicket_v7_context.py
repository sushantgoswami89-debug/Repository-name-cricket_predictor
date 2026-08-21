"""Train and evaluate a leakage-safe, wicket-only context candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_ipl_catboost_phase_moe_v1 import CATEGORICAL, MOE_FEATURES, SEED


VERSION = "ipl_wicket_v7_context"
TARGET_PRECISION = 0.42
EXTRA_FEATURES = [
    "batter_ball_age_bucket",
    "partner_ball_age_bucket",
    "partnership_age_bucket",
    "recent_wicket_flag",
    "wickets_pressure_bucket",
    "run_rate_gap",
    "phase_wicket_state",
    "phase_batter_age",
    "chase_wicket_state",
]
EXTRA_CATEGORICAL = [
    item for item in EXTRA_FEATURES if item != "run_rate_gap"
]
FEATURES = MOE_FEATURES + EXTRA_FEATURES
CATEGORICAL_V7 = CATEGORICAL + EXTRA_CATEGORICAL


def _bucket_age(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-1, 5, 11, 23, np.inf],
        labels=["0_5", "6_11", "12_23", "24_plus"],
    ).astype(str)


def _features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["batter_ball_age_bucket"] = _bucket_age(
        result["striker_match_balls"]
    )
    result["partner_ball_age_bucket"] = _bucket_age(
        result["partner_match_balls"]
    )
    result["partnership_age_bucket"] = _bucket_age(
        result["partnership_legal_ball_age"]
    )
    result["recent_wicket_flag"] = np.where(
        result["recent_wicket_rate"] > 0, "RECENT_WICKET", "NO_RECENT_WICKET"
    )
    result["wickets_pressure_bucket"] = pd.cut(
        result["wickets_in_hand"],
        bins=[-1, 3, 6, 10],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)
    result["run_rate_gap"] = (
        result["required_run_rate"] - result["current_run_rate"]
    ).clip(-10, 15)
    result["phase_wicket_state"] = (
        result["phase"].astype(str)
        + "|"
        + result["wickets_pressure_bucket"].astype(str)
    )
    result["phase_batter_age"] = (
        result["phase"].astype(str)
        + "|"
        + result["batter_ball_age_bucket"].astype(str)
    )
    result["chase_wicket_state"] = (
        result["chase_pressure"].astype(str)
        + "|"
        + result["wickets_pressure_bucket"].astype(str)
    )
    return result


def _frame(data: pd.DataFrame, features, categoricals) -> pd.DataFrame:
    result = data[features].copy()
    for column in categoricals:
        result[column] = result[column].fillna("__UNKNOWN__").astype(str)
    return result


def _model(depth: int, iterations: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        iterations=iterations,
        depth=depth,
        learning_rate=0.03,
        l2_leaf_reg=8,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def _fit_platt(probability: np.ndarray, actual: np.ndarray):
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1.0, random_state=SEED)
    model.fit(logit, actual)
    return model


def _apply_platt(model, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return model.predict_proba(logit)[:, 1]


def _probability_metrics(actual: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "rows": len(actual),
        "event_rate": float(np.mean(actual)),
        "brier": float(brier_score_loss(actual, probability)),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "pr_auc": float(average_precision_score(actual, probability)),
        "log_loss": float(log_loss(actual, probability)),
    }


def _select_threshold(actual: np.ndarray, probability: np.ndarray) -> dict:
    choices = []
    for threshold in np.linspace(0.05, 0.75, 141):
        predicted = probability >= threshold
        precision = precision_score(actual, predicted, zero_division=0)
        recall = recall_score(actual, predicted, zero_division=0)
        choices.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "accuracy": float(accuracy_score(actual, predicted)),
                "alert_rate": float(np.mean(predicted)),
            }
        )
    eligible = [
        item for item in choices if item["precision"] >= TARGET_PRECISION
    ]
    return max(
        eligible or choices,
        key=lambda item: (item["recall"], item["precision"], -item["alert_rate"]),
    )


def _segments(data, actual, probability) -> dict:
    definitions = {
        "powerplay": data["phase"].to_numpy() == "powerplay",
        "middle": data["phase"].to_numpy() == "middle",
        "death": data["phase"].to_numpy() == "death",
        "new_batter_0_5": data["striker_match_balls"].to_numpy() <= 5,
        "recent_wicket": data["recent_wicket_rate"].to_numpy() > 0,
        "high_chase_pressure": data["chase_pressure"].to_numpy() == "high",
        "three_wickets_or_less": data["wickets_in_hand"].to_numpy() <= 3,
    }
    return {
        name: _probability_metrics(actual[mask], probability[mask])
        for name, mask in definitions.items()
        if int(mask.sum()) >= 50 and len(np.unique(actual[mask])) == 2
    }


def train(root: Path) -> dict:
    data = _features(
        pd.read_csv(
            root
            / "data/candidates/ipl_cold_start_v5_t20_priors/training_overs.csv"
        )
    )
    data["match_date"] = pd.to_datetime(data["match_date"])
    data = data.sort_values(["match_date", "source_file", "innings", "over"])
    training = data[data["match_date"].dt.year <= 2023].copy()
    calibration = data[data["match_date"].dt.year == 2024].copy()
    holdout = data[data["match_date"].dt.year.isin([2025, 2026])].copy()
    split = len(calibration) // 2
    platt_fit = calibration.iloc[:split]
    selection = calibration.iloc[split:]

    specifications = [
        ("baseline_depth7", MOE_FEATURES, CATEGORICAL, 7, 350),
        ("context_depth4", FEATURES, CATEGORICAL_V7, 4, 450),
        ("context_depth5", FEATURES, CATEGORICAL_V7, 5, 450),
        ("context_depth6", FEATURES, CATEGORICAL_V7, 6, 350),
    ]
    fitted = {}
    selection_reports = {}
    for name, features, categoricals, depth, iterations in specifications:
        model = _model(depth, iterations)
        model.fit(
            _frame(training, features, categoricals),
            training["wicket_in_over"],
            cat_features=categoricals,
        )
        fit_raw = model.predict_proba(
            _frame(platt_fit, features, categoricals)
        )[:, 1]
        calibrator = _fit_platt(
            fit_raw, platt_fit["wicket_in_over"].to_numpy()
        )
        selection_raw = model.predict_proba(
            _frame(selection, features, categoricals)
        )[:, 1]
        selection_probability = _apply_platt(calibrator, selection_raw)
        metrics = _probability_metrics(
            selection["wicket_in_over"].to_numpy(), selection_probability
        )
        selection_reports[name] = metrics
        fitted[name] = (model, features, categoricals)
    winner = min(
        [name for name in selection_reports if name != "baseline_depth7"],
        key=lambda name: (
            selection_reports[name]["brier"],
            -selection_reports[name]["roc_auc"],
            -selection_reports[name]["pr_auc"],
        ),
    )

    probabilities = {}
    models = {}
    for name in ("baseline_depth7", winner):
        _, features, categoricals = fitted[name]
        model = _model(
            next(item[3] for item in specifications if item[0] == name),
            next(item[4] for item in specifications if item[0] == name),
        )
        model.fit(
            _frame(training, features, categoricals),
            training["wicket_in_over"],
            cat_features=categoricals,
        )
        calibration_raw = model.predict_proba(
            _frame(calibration, features, categoricals)
        )[:, 1]
        calibrator = _fit_platt(
            calibration_raw, calibration["wicket_in_over"].to_numpy()
        )
        holdout_raw = model.predict_proba(
            _frame(holdout, features, categoricals)
        )[:, 1]
        probabilities[name] = _apply_platt(calibrator, holdout_raw)
        models[name] = (model, calibrator)

    threshold_model, threshold_features, threshold_categoricals = fitted[winner]
    fit_raw = threshold_model.predict_proba(
        _frame(platt_fit, threshold_features, threshold_categoricals)
    )[:, 1]
    threshold_calibrator = _fit_platt(
        fit_raw, platt_fit["wicket_in_over"].to_numpy()
    )
    selection_probability = _apply_platt(
        threshold_calibrator,
        threshold_model.predict_proba(
            _frame(selection, threshold_features, threshold_categoricals)
        )[:, 1],
    )
    threshold_policy = _select_threshold(
        selection["wicket_in_over"].to_numpy(), selection_probability
    )

    actual = holdout["wicket_in_over"].to_numpy()
    baseline_probability = probabilities["baseline_depth7"]
    candidate_probability = probabilities[winner]
    threshold = float(threshold_policy["threshold"])
    predicted = candidate_probability >= threshold
    alert_metrics = {
        "threshold": threshold,
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "alert_rate": float(np.mean(predicted)),
    }
    temporal = {}
    for year in (2025, 2026):
        mask = (holdout["match_date"].dt.year == year).to_numpy()
        temporal[str(year)] = {
            "baseline": _probability_metrics(
                actual[mask], baseline_probability[mask]
            ),
            "candidate": _probability_metrics(
                actual[mask], candidate_probability[mask]
            ),
        }
    gates = {
        "brier_improves": (
            brier_score_loss(actual, candidate_probability)
            < brier_score_loss(actual, baseline_probability)
        ),
        "roc_auc_improves": (
            roc_auc_score(actual, candidate_probability)
            > roc_auc_score(actual, baseline_probability)
        ),
        "pr_auc_improves": (
            average_precision_score(actual, candidate_probability)
            > average_precision_score(actual, baseline_probability)
        ),
        "holdout_precision_at_least_42pct": alert_metrics["precision"] >= 0.42,
        "holdout_recall_above_current_27_86pct": alert_metrics["recall"] > 0.2786,
    }
    for year, values in temporal.items():
        gates[f"{year}_brier_improves"] = (
            values["candidate"]["brier"] < values["baseline"]["brier"]
        )
        gates[f"{year}_roc_auc_not_lower"] = (
            values["candidate"]["roc_auc"] >= values["baseline"]["roc_auc"]
        )
    report = {
        "candidate_version": VERSION,
        "candidate_only": True,
        "production_changed": False,
        "run_model_changed": False,
        "leakage_safe": True,
        "selection_split": {
            "training_through": 2023,
            "platt_fit": "first chronological half of 2024",
            "candidate_and_threshold_selection": "second chronological half of 2024",
            "locked_holdout": [2025, 2026],
        },
        "candidate_selection": selection_reports,
        "selected_model": winner,
        "baseline": _probability_metrics(actual, baseline_probability),
        "candidate": _probability_metrics(actual, candidate_probability),
        "threshold_selected_pre_holdout": threshold_policy,
        "holdout_alert_metrics": alert_metrics,
        "temporal_holdout": temporal,
        "baseline_segments": _segments(
            holdout, actual, baseline_probability
        ),
        "candidate_segments": _segments(
            holdout, actual, candidate_probability
        ),
        "promotion_gates": gates,
        "decision": (
            "promote_candidate" if all(gates.values()) else "reject_keep_research"
        ),
    }
    output = root / f"models/candidates/{VERSION}"
    output.mkdir(parents=True, exist_ok=True)
    models[winner][0].save_model(output / "wicket.cbm")
    pd.DataFrame(
        {
            "source_file": holdout["source_file"],
            "match_date": holdout["match_date"].astype(str),
            "innings": holdout["innings"],
            "over": holdout["over"],
            "actual": actual,
            "baseline_probability": baseline_probability,
            "candidate_probability": candidate_probability,
            "candidate_alert": predicted.astype(int),
        }
    ).to_csv(output / "holdout_predictions.csv", index=False)
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    train(Path(__file__).resolve().parents[1])
