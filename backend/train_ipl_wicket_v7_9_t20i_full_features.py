"""ipl_wicket_v7_7_t20i_augmented (T20I rows added to training with only
the ~13 match-state features common to both competitions, everything else
NaN) got the closest result of the whole line: 44.0% precision / 27.02%
recall vs. the 42%/27.86% gates -- misses by 0.84 points. This gives T20I
rows real values for the batter/bowler prior stats, chase pressure,
partnership age, and venue par score instead of NaN placeholders, by
extending `build_ipl_phase_moe_features` and `build_ipl_venue_regime_dataset`
with an opt-in `scopes` parameter (default unchanged, IPL-only -- existing
callers untouched) and calling them with `scopes=("ipl", "t20i")`. Their
core logic was already format-agnostic: `canonical_player_id` is
Cricsheet's own cross-format person UUID, and chase-pressure/partnership/
par-score math doesn't reference anything IPL-specific. Only
`canonical_team_id` (IPL franchise aliases) and `team_venue_context`
(home/away vs. IPL home grounds) degrade gracefully to an
unresolved-but-consistent fallback for national T20I teams.

Bowler current-spell/phase-h2h features (SPELL_NUMERIC/SPELL_CATEGORICAL)
are NOT extended this round -- that builder merges onto an already
IPL-only downstream file, a bigger job. Still NaN for T20I rows here.

IPL-side training/calibration/threshold-selection/holdout data is
UNCHANGED from v7.5/v7.6/v7.7/v7.8 (still the existing IPL-only enriched
dataset) -- only T20I training rows get the richer features. This keeps
every gate-defining split exactly apples-to-apples comparable to every
prior result in this line. Same architecture (depth6, l2=28), same T20I
sample weight (0.2, the value v7.7/v7.8 showed works best), same k-fold
threshold selection. Single holdout evaluation.
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

from app.ml.ipl_phase_moe_dataset import build_ipl_phase_moe_features
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from train_ipl_catboost_phase_moe_v1 import CATEGORICAL, MOE_FEATURES, SEED
from train_ipl_wicket_v7_1_margin_threshold import _apply_platt, _fit_platt
from train_ipl_wicket_v7_2_spell_features import CATEGORICAL_V72, FEATURES, _frame, _load_merged

VERSION = "ipl_wicket_v7_9_t20i_full_features"
TARGET_PRECISION = 0.42
TARGET_RECALL = 0.2786
EARLY_STOP_ROUNDS = 75
MAX_ITERATIONS = 3000
DEPTH, L2 = 6, 28
N_FOLDS = 5
T20I_SAMPLE_WEIGHT = 0.2


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


def _t20i_full_feature_frame(root: Path) -> pd.DataFrame:
    base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    ipl_files = {p.stem for p in (root / "data/raw/cricsheet/ipl").glob("*.json")}
    t20i = base.loc[
        ~base["source_file"].str.replace(".json", "", regex=False).isin(ipl_files)
    ].copy()
    t20i["match_date"] = pd.to_datetime(t20i["match_date"])
    t20i = t20i[t20i["match_date"].dt.year <= 2023].copy()

    moe = build_ipl_phase_moe_features(root, canonical_identities=True, scopes=("ipl", "t20i"))
    venue = build_ipl_venue_regime_dataset(root, scopes=("ipl", "t20i"))
    moe["match_date"] = pd.to_datetime(moe["match_date"])
    venue["match_date"] = pd.to_datetime(venue["match_date"])
    moe_t20i = moe.loc[~moe["source_file"].str.replace(".json", "", regex=False).isin(ipl_files)]
    venue_t20i = venue.loc[~venue["source_file"].str.replace(".json", "", regex=False).isin(ipl_files)]

    keys = ["source_file", "innings", "over"]
    moe_columns = [c for c in moe_t20i.columns if c not in ("match_date",) and c not in keys]
    venue_columns = [c for c in venue_t20i.columns if c not in ("match_date",) and c not in keys]
    t20i = t20i.merge(moe_t20i[keys + moe_columns], on=keys, how="left", validate="one_to_one")
    t20i = t20i.merge(venue_t20i[keys + venue_columns], on=keys, how="left", validate="one_to_one")

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
    train_eval = training.iloc[eval_cut:]
    train_fit_ipl["is_ipl"] = "ipl"
    train_fit_ipl["sample_weight"] = 1.0

    t20i = _t20i_full_feature_frame(root)
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
            f"Same as v7_7/v7_8 (depth={DEPTH}, l2={L2}, "
            f"best_iteration={best_iteration}, T20I sample_weight={T20I_SAMPLE_WEIGHT}) "
            "except T20I training rows now get real batter/bowler prior-stat, "
            "chase-pressure/partnership, and venue-par-score features "
            "(via new opt-in scopes=(ipl,t20i) params on "
            "build_ipl_phase_moe_features / build_ipl_venue_regime_dataset) "
            "instead of NaN placeholders. Bowler current-spell/phase-h2h "
            "features still NaN for T20I (not extended this round). "
            "IPL-side data (training/calibration/threshold/holdout) unchanged "
            "from v7_5 onward."
        ),
        "t20i_rows_added_to_training": int(len(t20i)),
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
