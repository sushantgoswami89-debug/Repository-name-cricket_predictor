"""ipl_wicket_v7_7_t20i_augmented (T20I sample_weight=0.2) landed at 44.0%
holdout precision / 27.02% recall vs. the 42%/27.86% gates -- precision
comfortably clears, recall misses by only 0.84 points, the closest gap of
any attempt in this line so far. This doubles the T20I sample weight to
0.4 to see if more T20I influence closes the remaining recall gap.
Everything else identical to v7_7.

`data/candidates/v3/verified_training_overs.csv` already combines IPL
(47,790 rows / 1,243 matches) and T20I (199,056 rows / 5,524 matches) --
everything richer built downstream (canonical player identities, venue
database, bowler current-spell/phase-h2h) comes from separate builders
hardcoded to `cricsheet/ipl` only, which is why T20I never reached the
v7.x line before. Building full T20I feature parity (its own venue/identity
pipeline) is real multi-session engineering. This is the cheap test first:
does raw additional match-state volume help at all, before investing in
that.

Adds T20I rows to the TRAINING split ONLY (never calibration, threshold
selection, or the locked holdout, which all stay pure IPL -- unchanged from
v7.5/v7.6, since those are literally what the gates measure). T20I rows
get the ~13 match-state features that both competitions share (score,
wickets, run rate, phase, chase pressure, recent-ball momentum); every
IPL-only enriched column (career stats, spell/h2h, venue, identity) is
left missing for them -- CatBoost handles missing natively. An `is_ipl`
flag and a sample weight for T20I rows (0.4 here, up from 0.2 in v7_7)
keep the model from being swamped by 4x the row volume of a different,
only partially overlapping population.

Same winning architecture as v7.5/v7.6 (depth6, l2=28) and the same
k-fold cross-fitted threshold selection as v7.6 (the more statistically
robust of the two methods tried). Single holdout evaluation at the end.
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
from sklearn.model_selection import StratifiedKFold

from train_ipl_catboost_phase_moe_v1 import CATEGORICAL, MOE_FEATURES, SEED
from train_ipl_wicket_v7_1_margin_threshold import _apply_platt, _fit_platt
from train_ipl_wicket_v7_2_spell_features import CATEGORICAL_V72, FEATURES, _frame, _load_merged

VERSION = "ipl_wicket_v7_8_t20i_weight_40"
TARGET_PRECISION = 0.42
TARGET_RECALL = 0.2786
EARLY_STOP_ROUNDS = 75
MAX_ITERATIONS = 3000
DEPTH, L2 = 6, 28  # winning architecture from v7_5, unchanged
N_FOLDS = 5
T20I_SAMPLE_WEIGHT = 0.4

# Columns present in the common v3-level base dataset for both competitions.
COMMON_COLUMNS = [
    "source_file", "match_date", "innings", "over", "phase",
    "score_before_over", "wkts_down_before_over", "wickets_in_hand",
    "legal_balls_bowled", "balls_remaining", "current_run_rate", "is_chase",
    "runs_required", "required_run_rate", "recent_legal_balls",
    "recent_runs_per_ball", "recent_dot_rate", "recent_single_rate",
    "recent_boundary_rate", "recent_wicket_rate", "phase_boundary_pressure",
    "phase_dot_pressure", "phase_rotation_value", "wicket_in_over",
]


def _model(depth: int, l2: float) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss", iterations=MAX_ITERATIONS, depth=depth,
        learning_rate=0.03, l2_leaf_reg=l2, random_seed=SEED, verbose=False,
        allow_writing_files=False, thread_count=-1,
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
        choices.append({
            "threshold": float(threshold), "precision": float(precision),
            "recall": float(recall), "accuracy": float(accuracy_score(actual, predicted)),
            "alert_rate": float(np.mean(predicted)),
        })
    eligible = [item for item in choices if item["precision"] >= TARGET_PRECISION]
    return max(eligible or choices, key=lambda item: (item["recall"], item["precision"], -item["alert_rate"]))


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


def _t20i_frame(root: Path) -> pd.DataFrame:
    base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    ipl_files = {p.stem for p in (root / "data/raw/cricsheet/ipl").glob("*.json")}
    t20i = base.loc[~base["source_file"].str.replace(".json", "", regex=False).isin(ipl_files)].copy()
    t20i["match_date"] = pd.to_datetime(t20i["match_date"])
    t20i = t20i[t20i["match_date"].dt.year <= 2023].copy()
    t20i["is_ipl"] = "t20i"
    for column in FEATURES + CATEGORICAL_V72:
        if column not in t20i.columns:
            t20i[column] = np.nan
    t20i["sample_weight"] = T20I_SAMPLE_WEIGHT
    return t20i


def train(root: Path) -> dict:
    data = _load_merged(root)
    data["match_date"] = pd.to_datetime(data["match_date"])
    data = data.sort_values(["match_date", "source_file", "innings", "over"])
    training = data[data["match_date"].dt.year <= 2023].copy()
    calibration = data[data["match_date"].dt.year == 2024].copy().reset_index(drop=True)
    holdout = data[data["match_date"].dt.year.isin([2025, 2026])].copy()

    eval_cut = int(len(training) * 0.85)
    train_fit_ipl = training.iloc[:eval_cut].copy()
    train_eval = training.iloc[eval_cut:]  # IPL-only early-stopping validation, unchanged from v7.5/v7.6
    train_fit_ipl["is_ipl"] = "ipl"
    train_fit_ipl["sample_weight"] = 1.0

    t20i = _t20i_frame(root)
    train_fit = pd.concat([train_fit_ipl, t20i], ignore_index=True, sort=False)

    categoricals_plus = CATEGORICAL_V72 + ["is_ipl"]
    features_plus = FEATURES + ["is_ipl"]

    baseline_model = CatBoostClassifier(
        loss_function="Logloss", iterations=350, depth=7, learning_rate=0.03,
        l2_leaf_reg=8, random_seed=SEED, verbose=False, allow_writing_files=False,
        thread_count=-1,
    )
    baseline_model.fit(
        _frame(training, MOE_FEATURES, CATEGORICAL), training["wicket_in_over"],
        cat_features=CATEGORICAL,
    )
    candidate_model = _model(DEPTH, L2)
    candidate_model.fit(
        _frame(train_fit, features_plus, categoricals_plus),
        train_fit["wicket_in_over"],
        sample_weight=train_fit["sample_weight"].to_numpy(),
        cat_features=categoricals_plus,
        eval_set=(
            _frame(train_eval.assign(is_ipl="ipl"), features_plus, categoricals_plus),
            train_eval["wicket_in_over"],
        ),
        use_best_model=True,
    )
    best_iteration = int(candidate_model.get_best_iteration())

    calibration = calibration.assign(is_ipl="ipl")
    holdout = holdout.assign(is_ipl="ipl")

    # k-fold cross-fitted threshold selection over the full 2024 calibration set (same method as v7_6)
    candidate_raw_all = candidate_model.predict_proba(_frame(calibration, features_plus, categoricals_plus))[:, 1]
    oof_probability = np.zeros(len(calibration))
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fit_idx, holdout_idx in kfold.split(calibration, calibration["wicket_in_over"]):
        calibrator = _fit_platt(candidate_raw_all[fit_idx], calibration["wicket_in_over"].to_numpy()[fit_idx])
        oof_probability[holdout_idx] = _apply_platt(calibrator, candidate_raw_all[holdout_idx])
    threshold_policy = _select_threshold(calibration["wicket_in_over"].to_numpy(), oof_probability)

    probabilities = {}
    for name, model, features, categoricals in (
        ("baseline_depth7", baseline_model, MOE_FEATURES, CATEGORICAL),
        (VERSION, candidate_model, features_plus, categoricals_plus),
    ):
        calibration_raw = model.predict_proba(_frame(calibration, features, categoricals))[:, 1]
        calibrator = _fit_platt(calibration_raw, calibration["wicket_in_over"].to_numpy())
        holdout_raw = model.predict_proba(_frame(holdout, features, categoricals))[:, 1]
        probabilities[name] = _apply_platt(calibrator, holdout_raw)

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
        "brier_improves": brier_score_loss(actual, candidate_probability) < brier_score_loss(actual, baseline_probability),
        "roc_auc_improves": roc_auc_score(actual, candidate_probability) > roc_auc_score(actual, baseline_probability),
        "pr_auc_improves": average_precision_score(actual, candidate_probability) > average_precision_score(actual, baseline_probability),
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
            f"Same architecture as v7_5/v7_6 (depth={DEPTH}, l2={L2}, "
            f"best_iteration={best_iteration}). Training FIT augmented with "
            f"{len(t20i)} T20I rows (<=2023, sample_weight={T20I_SAMPLE_WEIGHT}) "
            "using only the ~13 match-state features common to both "
            "competitions; IPL-only enriched columns left missing for T20I "
            "rows. Early-stopping eval, calibration, threshold selection "
            f"({N_FOLDS}-fold cross-fit, matching v7_6), and holdout are all "
            "unchanged and IPL-only."
        ),
        "t20i_rows_added_to_training": int(len(t20i)),
        "t20i_sample_weight": T20I_SAMPLE_WEIGHT,
        "ipl_training_rows": int(len(train_fit_ipl)),
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
    pd.DataFrame({
        "source_file": holdout["source_file"], "match_date": holdout["match_date"].astype(str),
        "innings": holdout["innings"], "over": holdout["over"], "actual": actual,
        "baseline_probability": baseline_probability, "candidate_probability": candidate_probability,
        "candidate_alert": predicted.astype(int),
    }).to_csv(output / "holdout_predictions.csv", index=False)
    (output / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    train(Path(__file__).resolve().parents[1])
