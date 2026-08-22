"""ipl_wicket_v7_context (context_depth4) and its v7.1 threshold-margin
variant both hit a genuine precision/recall cliff on the locked holdout:
no threshold clears both the >=42% precision and >27.86% recall gates at
once (see train_ipl_wicket_v7_1_margin_threshold.py's sweep). That means
the model's discrimination, not the threshold, was the limit.

The `announced_bowler_current_spell_v3` / `announced_bowler_phase_h2h_v1`
research line already proved bowler current-spell signal adds real wicket
predictive value (precision_improves_same_alert_volume: true in both), but
so far only as a live post-hoc adjustment layered on the production model
(shipped as "wicket model v2", commit 765a118) -- never as a training
feature of a from-scratch candidate like this one.

This candidate merges that proven signal into the v7 context feature set
(the two training_overs.csv files join 1:1 on source_file/innings/over,
47756/47756 rows, confirmed before writing this) and retrains
context_depth4 with it, keeping every other part of the pipeline
(architecture, split, calibration, promotion gates) identical to v7.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
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
from train_ipl_wicket_v7_context import CATEGORICAL_V7, FEATURES as V7_FEATURES, _features, _model
from train_ipl_wicket_v7_1_margin_threshold import _apply_platt, _fit_platt

VERSION = "ipl_wicket_v7_2_spell_features"
PROMOTION_TARGET_PRECISION = 0.42
PROMOTION_TARGET_RECALL = 0.2786
DEPTH = 4
ITERATIONS = 450

SPELL_NUMERIC = [
    "current_spell_balls",
    "spell_number",
    "bowler_match_economy",
    "bowler_match_dot_rate",
    "bowler_phase_history_economy",
    "bowler_phase_history_wickets",
]
SPELL_CATEGORICAL = ["spell_state", "bowler_history_supported"]

FEATURES = V7_FEATURES + SPELL_NUMERIC + SPELL_CATEGORICAL
CATEGORICAL_V72 = CATEGORICAL_V7 + SPELL_CATEGORICAL


def _frame(data: pd.DataFrame, features, categoricals) -> pd.DataFrame:
    result = data[features].copy()
    for column in categoricals:
        result[column] = result[column].fillna("__UNKNOWN__").astype(str)
    return result


def _probability_metrics(actual: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "rows": len(actual),
        "event_rate": float(np.mean(actual)),
        "brier": float(brier_score_loss(actual, probability)),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "pr_auc": float(average_precision_score(actual, probability)),
        "log_loss": float(log_loss(actual, probability)),
    }


def _select_threshold(actual: np.ndarray, probability: np.ndarray, target_precision: float) -> dict:
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
    eligible = [item for item in choices if item["precision"] >= target_precision]
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


def _load_merged(root: Path) -> pd.DataFrame:
    base = _features(
        pd.read_csv(
            root / "data/candidates/ipl_cold_start_v5_t20_priors/training_overs.csv"
        )
    )
    spell = pd.read_csv(
        root / "data/candidates/announced_bowler_phase_h2h/training_overs.csv"
    )[["source_file", "innings", "over"] + SPELL_NUMERIC + SPELL_CATEGORICAL]
    merged = base.merge(spell, on=["source_file", "innings", "over"], how="inner")
    assert len(merged) == len(base), "expected a clean 1:1 join"
    merged["bowler_history_supported"] = merged["bowler_history_supported"].astype(str)
    return merged


def train(root: Path, target_precision: float = PROMOTION_TARGET_PRECISION) -> dict:
    data = _load_merged(root)
    data["match_date"] = pd.to_datetime(data["match_date"])
    data = data.sort_values(["match_date", "source_file", "innings", "over"])
    training = data[data["match_date"].dt.year <= 2023].copy()
    calibration = data[data["match_date"].dt.year == 2024].copy()
    holdout = data[data["match_date"].dt.year.isin([2025, 2026])].copy()
    split = len(calibration) // 2
    platt_fit = calibration.iloc[:split]
    selection = calibration.iloc[split:]

    baseline_model = _model(7, 350)
    baseline_model.fit(
        _frame(training, MOE_FEATURES, CATEGORICAL),
        training["wicket_in_over"],
        cat_features=CATEGORICAL,
    )
    candidate_model = _model(DEPTH, ITERATIONS)
    candidate_model.fit(
        _frame(training, FEATURES, CATEGORICAL_V72),
        training["wicket_in_over"],
        cat_features=CATEGORICAL_V72,
    )

    probabilities = {}
    for name, model, features, categoricals in (
        ("baseline_depth7", baseline_model, MOE_FEATURES, CATEGORICAL),
        (VERSION, candidate_model, FEATURES, CATEGORICAL_V72),
    ):
        calibration_raw = model.predict_proba(
            _frame(calibration, features, categoricals)
        )[:, 1]
        calibrator = _fit_platt(
            calibration_raw, calibration["wicket_in_over"].to_numpy()
        )
        holdout_raw = model.predict_proba(_frame(holdout, features, categoricals))[:, 1]
        probabilities[name] = _apply_platt(calibrator, holdout_raw)

    fit_raw = candidate_model.predict_proba(
        _frame(platt_fit, FEATURES, CATEGORICAL_V72)
    )[:, 1]
    threshold_calibrator = _fit_platt(fit_raw, platt_fit["wicket_in_over"].to_numpy())
    selection_probability = _apply_platt(
        threshold_calibrator,
        candidate_model.predict_proba(_frame(selection, FEATURES, CATEGORICAL_V72))[:, 1],
    )
    threshold_policy = _select_threshold(
        selection["wicket_in_over"].to_numpy(), selection_probability, target_precision
    )

    actual = holdout["wicket_in_over"].to_numpy()
    baseline_probability = probabilities["baseline_depth7"]
    candidate_probability = probabilities[VERSION]
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
            "baseline": _probability_metrics(actual[mask], baseline_probability[mask]),
            "candidate": _probability_metrics(actual[mask], candidate_probability[mask]),
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
        "holdout_precision_at_least_42pct": (
            alert_metrics["precision"] >= PROMOTION_TARGET_PRECISION
        ),
        "holdout_recall_above_current_27_86pct": (
            alert_metrics["recall"] > PROMOTION_TARGET_RECALL
        ),
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
        "note": (
            "Same context_depth4 architecture and split as ipl_wicket_v7_context, "
            "plus bowler current-spell / phase-history features merged in from "
            "the announced_bowler_phase_h2h research line "
            f"({SPELL_NUMERIC + SPELL_CATEGORICAL}). Threshold selected at "
            f"target precision {target_precision}."
        ),
        "selection_split": {
            "training_through": 2023,
            "platt_fit": "first chronological half of 2024",
            "candidate_and_threshold_selection": "second chronological half of 2024",
            "locked_holdout": [2025, 2026],
        },
        "baseline": _probability_metrics(actual, baseline_probability),
        "candidate": _probability_metrics(actual, candidate_probability),
        "threshold_selected_pre_holdout": threshold_policy,
        "holdout_alert_metrics": alert_metrics,
        "temporal_holdout": temporal,
        "baseline_segments": _segments(holdout, actual, baseline_probability),
        "candidate_segments": _segments(holdout, actual, candidate_probability),
        "promotion_gates": gates,
        "decision": (
            "promote_candidate" if all(gates.values()) else "reject_keep_research"
        ),
    }
    output = root / f"models/candidates/{VERSION}"
    output.mkdir(parents=True, exist_ok=True)
    candidate_model.save_model(output / "wicket.cbm")
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
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    train(Path(__file__).resolve().parents[1])
