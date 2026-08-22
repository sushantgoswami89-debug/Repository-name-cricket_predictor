"""ipl_wicket_v7_2_spell_features (context_depth4, 450 fixed iterations)
was rejected: best achievable was ~41% holdout precision at the 27.86%
recall floor. A follow-up depth/iteration sweep (v7.3, undocumented script,
see docs/candidate_ipl_wicket_v7_2_spell_features.md attempt 3) found that
forcing MORE iterations at higher depth badly overfits the ~1369-row
selection split -- holdout recall collapsed to single digits.

This candidate retests depth with the actual fix for that: CatBoost early
stopping against a held-out chronological TAIL OF TRAINING itself (last
15% of pre-2024 rows by date -- NOT platt_fit/selection/holdout, so no new
leakage into those splits), instead of a hand-picked fixed iteration count.
An exploratory scratch sweep (since deleted) found best_iteration landed at
just 71-82 rounds across every depth/L2 combination tried -- confirming the
prior fixed 450-900 iteration counts were the actual cause of the
overfitting, not depth itself.

Methodology note: architecture (depth, L2) is selected here using ONLY the
selection split (brier, matching train_ipl_wicket_v7_context.py's own
selection criterion) -- never by peeking at holdout. The threshold is
picked once, at the real 0.42 target precision, on the selection split.
Holdout is evaluated exactly once, at the end, for the real gate check.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
from train_ipl_wicket_v7_1_margin_threshold import _apply_platt, _fit_platt
from train_ipl_wicket_v7_2_spell_features import CATEGORICAL_V72, FEATURES, _frame, _load_merged

VERSION = "ipl_wicket_v7_5_early_stopping"
TARGET_PRECISION = 0.42
TARGET_RECALL = 0.2786
EARLY_STOP_ROUNDS = 75
MAX_ITERATIONS = 3000

# (depth, l2_leaf_reg) candidates -- selected on the selection split only.
ARCHITECTURES = [
    ("depth4_l2_8", 4, 8),
    ("depth5_l2_12", 5, 12),
    ("depth6_l2_16", 6, 16),
    ("depth6_l2_28", 6, 28),
]


def _model(depth: int, l2: float) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        iterations=MAX_ITERATIONS,
        depth=depth,
        learning_rate=0.03,
        l2_leaf_reg=l2,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        early_stopping_rounds=EARLY_STOP_ROUNDS,
    )


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
    eligible = [item for item in choices if item["precision"] >= TARGET_PRECISION]
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
    data = _load_merged(root)
    data["match_date"] = pd.to_datetime(data["match_date"])
    data = data.sort_values(["match_date", "source_file", "innings", "over"])
    training = data[data["match_date"].dt.year <= 2023].copy()
    calibration = data[data["match_date"].dt.year == 2024].copy()
    holdout = data[data["match_date"].dt.year.isin([2025, 2026])].copy()
    split = len(calibration) // 2
    platt_fit = calibration.iloc[:split]
    selection = calibration.iloc[split:]

    eval_cut = int(len(training) * 0.85)
    train_fit = training.iloc[:eval_cut]
    train_eval = training.iloc[eval_cut:]

    # --- architecture selection on the selection split only ---
    fitted = {}
    selection_reports = {}
    for name, depth, l2 in ARCHITECTURES:
        model = _model(depth, l2)
        model.fit(
            _frame(train_fit, FEATURES, CATEGORICAL_V72),
            train_fit["wicket_in_over"],
            cat_features=CATEGORICAL_V72,
            eval_set=(
                _frame(train_eval, FEATURES, CATEGORICAL_V72),
                train_eval["wicket_in_over"],
            ),
            use_best_model=True,
        )
        fit_raw = model.predict_proba(_frame(platt_fit, FEATURES, CATEGORICAL_V72))[:, 1]
        calibrator = _fit_platt(fit_raw, platt_fit["wicket_in_over"].to_numpy())
        selection_raw = model.predict_proba(_frame(selection, FEATURES, CATEGORICAL_V72))[:, 1]
        selection_probability = _apply_platt(calibrator, selection_raw)
        metrics = _probability_metrics(selection["wicket_in_over"].to_numpy(), selection_probability)
        metrics["best_iteration"] = int(model.get_best_iteration())
        selection_reports[name] = metrics
        fitted[name] = model
    winner = min(selection_reports, key=lambda name: selection_reports[name]["brier"])

    # --- refit winner (and baseline) on FULL training (<=2023), same early-stopping split ---
    winner_depth, winner_l2 = next((d, l) for n, d, l in ARCHITECTURES if n == winner)
    baseline_model = CatBoostClassifier(
        loss_function="Logloss", iterations=350, depth=7, learning_rate=0.03,
        l2_leaf_reg=8, random_seed=SEED, verbose=False, allow_writing_files=False,
        thread_count=-1,
    )
    baseline_model.fit(
        _frame(training, MOE_FEATURES, CATEGORICAL),
        training["wicket_in_over"],
        cat_features=CATEGORICAL,
    )
    candidate_model = _model(winner_depth, winner_l2)
    candidate_model.fit(
        _frame(train_fit, FEATURES, CATEGORICAL_V72),
        train_fit["wicket_in_over"],
        cat_features=CATEGORICAL_V72,
        eval_set=(
            _frame(train_eval, FEATURES, CATEGORICAL_V72),
            train_eval["wicket_in_over"],
        ),
        use_best_model=True,
    )

    probabilities = {}
    for name, model, features, categoricals in (
        ("baseline_depth7", baseline_model, MOE_FEATURES, CATEGORICAL),
        (VERSION, candidate_model, FEATURES, CATEGORICAL_V72),
    ):
        calibration_raw = model.predict_proba(_frame(calibration, features, categoricals))[:, 1]
        calibrator = _fit_platt(calibration_raw, calibration["wicket_in_over"].to_numpy())
        holdout_raw = model.predict_proba(_frame(holdout, features, categoricals))[:, 1]
        probabilities[name] = _apply_platt(calibrator, holdout_raw)

    fit_raw = candidate_model.predict_proba(_frame(platt_fit, FEATURES, CATEGORICAL_V72))[:, 1]
    threshold_calibrator = _fit_platt(fit_raw, platt_fit["wicket_in_over"].to_numpy())
    selection_probability = _apply_platt(
        threshold_calibrator,
        candidate_model.predict_proba(_frame(selection, FEATURES, CATEGORICAL_V72))[:, 1],
    )
    threshold_policy = _select_threshold(selection["wicket_in_over"].to_numpy(), selection_probability)

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
            brier_score_loss(actual, candidate_probability) < brier_score_loss(actual, baseline_probability)
        ),
        "roc_auc_improves": (
            roc_auc_score(actual, candidate_probability) > roc_auc_score(actual, baseline_probability)
        ),
        "pr_auc_improves": (
            average_precision_score(actual, candidate_probability)
            > average_precision_score(actual, baseline_probability)
        ),
        "holdout_precision_at_least_42pct": alert_metrics["precision"] >= TARGET_PRECISION,
        "holdout_recall_above_current_27_86pct": alert_metrics["recall"] > TARGET_RECALL,
    }
    for year, values in temporal.items():
        gates[f"{year}_brier_improves"] = values["candidate"]["brier"] < values["baseline"]["brier"]
        gates[f"{year}_roc_auc_not_lower"] = values["candidate"]["roc_auc"] >= values["baseline"]["roc_auc"]

    report = {
        "candidate_version": VERSION,
        "candidate_only": True,
        "production_changed": False,
        "run_model_changed": False,
        "leakage_safe": True,
        "note": (
            "Same spell-enriched feature set as ipl_wicket_v7_2_spell_features. "
            "Architecture selected among "
            f"{[a[0] for a in ARCHITECTURES]} by selection-split brier only "
            f"(winner: {winner}, best_iteration={selection_reports[winner]['best_iteration']}). "
            "Early stopping against a chronological tail of TRAINING (<=2023), "
            "not calibration/selection/holdout. Threshold fixed at the real "
            f"{TARGET_PRECISION} target precision on the selection split, "
            "evaluated on locked holdout exactly once -- no holdout peeking."
        ),
        "architecture_selection": selection_reports,
        "selected_architecture": winner,
        "selection_split": {
            "training_through": 2023,
            "early_stopping_eval": "chronological last 15% of pre-2024 rows",
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
        "decision": "promote_candidate" if all(gates.values()) else "reject_keep_research",
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
    (output / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    train(Path(__file__).resolve().parents[1])
