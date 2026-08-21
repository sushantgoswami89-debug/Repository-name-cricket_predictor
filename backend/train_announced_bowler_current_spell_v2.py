"""Train a bounded two-stage current-spell bowler adjustment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_ipl_wicket_v7_context import (
    CATEGORICAL_V7,
    FEATURES,
    _apply_platt,
    _features,
    _fit_platt,
    _frame,
    _model,
)


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
BOWLER_NUMERIC = [
    "bowler_match_balls",
    "bowler_match_runs_conceded",
    "bowler_match_wickets",
    "bowler_match_dot_rate",
    "bowler_match_boundary_concession_rate",
    "bowler_match_strike_rate",
    "bowler_match_economy",
    "overs_since_previous",
    "consecutive_overs",
    "spell_number",
    "current_spell_balls",
    "bowler_phase_history_balls",
    "bowler_phase_history_runs_conceded",
    "bowler_phase_history_wickets",
    "bowler_phase_history_dot_rate",
    "bowler_phase_history_boundary_concession_rate",
    "bowler_phase_history_strike_rate",
    "bowler_phase_history_economy",
    "h2h_balls",
    "h2h_runs",
    "h2h_wickets",
    "h2h_dot_rate",
    "h2h_boundary_rate",
]
ADJUSTMENT_CONTEXT = [
    "base_prediction",
    "over",
    "wickets_in_hand",
    "partnership_legal_ball_age",
    "recent_wicket_rate",
    "current_run_rate",
    "required_run_rate",
] + BOWLER_NUMERIC
ADJUSTMENT_CATEGORICAL = [
    "phase",
    "spell_state",
    "bowler_history_supported",
    "h2h_supported",
]
MAX_RUN_DELTA = 1.5
MAX_WICKET_LOGIT_DELTA = 0.45


def _regressor(iterations: int = 450, depth: int = 5) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE",
        iterations=iterations,
        depth=depth,
        learning_rate=0.035,
        l2_leaf_reg=12,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def _adjustment_frame(data: pd.DataFrame, base: np.ndarray) -> pd.DataFrame:
    frame = data[ADJUSTMENT_CONTEXT[1:] + ADJUSTMENT_CATEGORICAL].copy()
    frame.insert(0, "base_prediction", base)
    for column in ADJUSTMENT_CATEGORICAL:
        frame[column] = frame[column].fillna("__UNKNOWN__").astype(str)
    return frame


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def _logistic(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _blend_runs(
    base: np.ndarray, proposed: np.ndarray, alpha: float
) -> np.ndarray:
    delta = np.clip(proposed - base, -MAX_RUN_DELTA, MAX_RUN_DELTA)
    return base + alpha * delta


def _blend_wickets(
    base: np.ndarray, proposed: np.ndarray, alpha: float
) -> np.ndarray:
    delta = np.clip(
        _logit(proposed) - _logit(base),
        -MAX_WICKET_LOGIT_DELTA,
        MAX_WICKET_LOGIT_DELTA,
    )
    return _logistic(_logit(base) + alpha * delta)


def _fixed_range(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = np.maximum(0, np.floor(prediction).astype(int) - 1)
    return low, low + 3


def _calibration(actual: np.ndarray, probability: np.ndarray) -> list[dict]:
    labels = pd.cut(
        probability,
        bins=[0, 0.15, 0.25, 0.35, 0.45, 0.60, 1.0],
        include_lowest=True,
    )
    frame = pd.DataFrame({"actual": actual, "p": probability, "band": labels})
    return [
        {
            "band": str(band),
            "rows": len(group),
            "mean_probability": float(group["p"].mean()),
            "event_rate": float(group["actual"].mean()),
        }
        for band, group in frame.groupby("band", observed=True)
    ]


def _metrics(
    actual_runs: np.ndarray,
    predicted_runs: np.ndarray,
    actual_wicket: np.ndarray,
    probability: np.ndarray,
    alert_count: int,
) -> dict:
    low, high = _fixed_range(predicted_runs)
    alerts = np.zeros(len(probability), dtype=bool)
    count = min(alert_count, len(probability))
    if count:
        alerts[np.argsort(-probability, kind="stable")[:count]] = True
    return {
        "rows": len(probability),
        "run_mae": float(np.mean(np.abs(predicted_runs - actual_runs))),
        "run_bias": float(np.mean(predicted_runs - actual_runs)),
        "fixed_width_3_run_coverage": float(
            np.mean((actual_runs >= low) & (actual_runs <= high))
        ),
        "wicket_brier": float(brier_score_loss(actual_wicket, probability)),
        "wicket_roc_auc": float(roc_auc_score(actual_wicket, probability)),
        "wicket_pr_auc": float(
            average_precision_score(actual_wicket, probability)
        ),
        "alert_count": int(alerts.sum()),
        "alert_volume": float(alerts.mean()),
        "precision_at_existing_alert_volume": float(
            precision_score(actual_wicket, alerts, zero_division=0)
        ),
        "recall_at_existing_alert_volume": float(
            recall_score(actual_wicket, alerts, zero_division=0)
        ),
        "wicket_calibration": _calibration(actual_wicket, probability),
    }


def _tune_alpha(
    actual: np.ndarray,
    base: np.ndarray,
    proposed: np.ndarray,
    *,
    wicket: bool,
) -> tuple[float, list[dict]]:
    trials = []
    for alpha in np.linspace(0, 1, 21):
        prediction = (
            _blend_wickets(base, proposed, float(alpha))
            if wicket
            else _blend_runs(base, proposed, float(alpha))
        )
        score = (
            brier_score_loss(actual, prediction)
            if wicket
            else np.mean(np.abs(prediction - actual))
        )
        trials.append({"alpha": float(alpha), "score": float(score)})
    winner = min(trials, key=lambda item: (item["score"], item["alpha"]))
    return winner["alpha"], trials


def _chronological_oof(data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate adjustment-training inputs from strictly earlier seasons."""

    parts = []
    run_predictions = []
    wicket_predictions = []
    for start, end in ((2016, 2019), (2020, 2021), (2022, 2023)):
        fit = data[data["match_date"].dt.year < start]
        validation = data[
            data["match_date"].dt.year.between(start, end)
        ]
        run_model = _regressor(350, 5)
        run_model.fit(
            _frame(fit, FEATURES, CATEGORICAL_V7),
            fit["runs_in_over"],
            cat_features=CATEGORICAL_V7,
        )
        wicket_model = _model(4, 350)
        wicket_model.fit(
            _frame(fit, FEATURES, CATEGORICAL_V7),
            fit["wicket_in_over"],
            cat_features=CATEGORICAL_V7,
        )
        parts.append(validation)
        run_predictions.append(
            run_model.predict(_frame(validation, FEATURES, CATEGORICAL_V7))
        )
        wicket_predictions.append(
            wicket_model.predict_proba(
                _frame(validation, FEATURES, CATEGORICAL_V7)
            )[:, 1]
        )
    return (
        pd.concat(parts, ignore_index=True),
        np.concatenate(run_predictions),
        np.concatenate(wicket_predictions),
    )


def _evidence_strength(data: pd.DataFrame) -> np.ndarray:
    match = data["bowler_match_balls"].to_numpy() / (
        data["bowler_match_balls"].to_numpy() + 18.0
    )
    history = data["bowler_phase_history_balls"].to_numpy() / (
        data["bowler_phase_history_balls"].to_numpy() + 120.0
    )
    h2h = data["h2h_balls"].to_numpy() / (data["h2h_balls"].to_numpy() + 24.0)
    return np.clip(0.55 * match + 0.35 * history + 0.10 * h2h, 0.0, 1.0)


def _phase_alphas(
    data: pd.DataFrame,
    actual: np.ndarray,
    base: np.ndarray,
    proposed: np.ndarray,
    *,
    wicket: bool,
    smooth: bool,
) -> dict[str, float]:
    result = {}
    strength = _evidence_strength(data) if smooth else np.ones(len(data))
    for phase in ("powerplay", "middle", "death"):
        mask = data["phase"].eq(phase).to_numpy()
        adjusted = (
            _blend_wickets(
                base[mask],
                _logistic(
                    _logit(base[mask])
                    + strength[mask]
                    * (_logit(proposed[mask]) - _logit(base[mask]))
                ),
                1.0,
            )
            if wicket
            else base[mask] + strength[mask] * (proposed[mask] - base[mask])
        )
        result[phase], _ = _tune_alpha(
            actual[mask], base[mask], adjusted, wicket=wicket
        )
    return result


def _apply_phase_policy(
    data: pd.DataFrame,
    base: np.ndarray,
    proposed: np.ndarray,
    alphas: dict[str, float],
    *,
    wicket: bool,
    smooth: bool,
) -> np.ndarray:
    output = base.copy()
    strength = _evidence_strength(data) if smooth else np.ones(len(data))
    for phase, alpha in alphas.items():
        mask = data["phase"].eq(phase).to_numpy()
        if wicket:
            delta = strength[mask] * (
                _logit(proposed[mask]) - _logit(base[mask])
            )
            softened = _logistic(_logit(base[mask]) + delta)
            output[mask] = _blend_wickets(
                base[mask], softened, alpha
            )
        else:
            softened = base[mask] + strength[mask] * (
                proposed[mask] - base[mask]
            )
            output[mask] = _blend_runs(base[mask], softened, alpha)
    return output


def train(root: Path = ROOT) -> dict:
    data = _features(
        pd.read_csv(
            root
            / "data/candidates/announced_bowler_current_spell_v2/training_overs.csv"
        )
    )
    data["match_date"] = pd.to_datetime(data["match_date"])
    data = data.sort_values(["match_date", "source_file", "innings", "over"])
    training = data[data["match_date"].dt.year <= 2023].copy()
    calibration = data[data["match_date"].dt.year == 2024].copy()
    holdout = data[data["match_date"].dt.year.isin([2025, 2026])].copy()
    split = len(calibration) // 2
    calibration_fit = calibration.iloc[:split]
    selection = calibration.iloc[split:]
    oof_data, oof_run_base, oof_wicket_base = _chronological_oof(training)

    run_base = _regressor(500, 6)
    run_base.fit(
        _frame(training, FEATURES, CATEGORICAL_V7),
        training["runs_in_over"],
        cat_features=CATEGORICAL_V7,
    )
    selection_run_base = run_base.predict(
        _frame(selection, FEATURES, CATEGORICAL_V7)
    )
    holdout_run_base = run_base.predict(_frame(holdout, FEATURES, CATEGORICAL_V7))
    run_adjuster = _regressor(400, 5)
    run_adjuster.fit(
        _adjustment_frame(oof_data, oof_run_base),
        oof_data["runs_in_over"],
        cat_features=ADJUSTMENT_CATEGORICAL,
    )
    selection_run_proposed = run_adjuster.predict(
        _adjustment_frame(selection, selection_run_base)
    )
    holdout_run_proposed = run_adjuster.predict(
        _adjustment_frame(holdout, holdout_run_base)
    )
    run_alpha, run_trials = _tune_alpha(
        selection["runs_in_over"].to_numpy(),
        selection_run_base,
        selection_run_proposed,
        wicket=False,
    )
    holdout_run_candidate = _blend_runs(
        holdout_run_base, holdout_run_proposed, run_alpha
    )

    wicket_base = _model(4, 450)
    wicket_base.fit(
        _frame(training, FEATURES, CATEGORICAL_V7),
        training["wicket_in_over"],
        cat_features=CATEGORICAL_V7,
    )
    base_calibration_raw = wicket_base.predict_proba(
        _frame(calibration_fit, FEATURES, CATEGORICAL_V7)
    )[:, 1]
    base_calibrator = _fit_platt(
        base_calibration_raw, calibration_fit["wicket_in_over"].to_numpy()
    )
    selection_wicket_base = _apply_platt(
        base_calibrator,
        wicket_base.predict_proba(
            _frame(selection, FEATURES, CATEGORICAL_V7)
        )[:, 1],
    )
    holdout_wicket_base = _apply_platt(
        base_calibrator,
        wicket_base.predict_proba(
            _frame(holdout, FEATURES, CATEGORICAL_V7)
        )[:, 1],
    )

    wicket_adjuster = CatBoostClassifier(
        loss_function="Logloss",
        iterations=450,
        depth=5,
        learning_rate=0.03,
        l2_leaf_reg=12,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    wicket_adjuster.fit(
        _adjustment_frame(oof_data, oof_wicket_base),
        oof_data["wicket_in_over"],
        cat_features=ADJUSTMENT_CATEGORICAL,
    )
    fit_wicket_proposed_raw = wicket_adjuster.predict_proba(
        _adjustment_frame(
            calibration_fit,
            _apply_platt(
                base_calibrator,
                wicket_base.predict_proba(
                    _frame(calibration_fit, FEATURES, CATEGORICAL_V7)
                )[:, 1],
            ),
        )
    )[:, 1]
    proposed_calibrator: LogisticRegression = _fit_platt(
        fit_wicket_proposed_raw,
        calibration_fit["wicket_in_over"].to_numpy(),
    )
    selection_wicket_proposed = _apply_platt(
        proposed_calibrator,
        wicket_adjuster.predict_proba(
            _adjustment_frame(selection, selection_wicket_base)
        )[:, 1],
    )
    holdout_wicket_proposed = _apply_platt(
        proposed_calibrator,
        wicket_adjuster.predict_proba(
            _adjustment_frame(holdout, holdout_wicket_base)
        )[:, 1],
    )
    wicket_alpha, wicket_trials = _tune_alpha(
        selection["wicket_in_over"].to_numpy(),
        selection_wicket_base,
        selection_wicket_proposed,
        wicket=True,
    )
    holdout_wicket_candidate = _blend_wickets(
        holdout_wicket_base, holdout_wicket_proposed, wicket_alpha
    )

    # Stage 2: retain the global adjustment only in phases where it improved
    # the pre-holdout selection score.
    run_safety_alphas = {}
    wicket_safety_alphas = {}
    for phase in ("powerplay", "middle", "death"):
        mask = selection["phase"].eq(phase).to_numpy()
        run_candidate = _blend_runs(
            selection_run_base[mask], selection_run_proposed[mask], run_alpha
        )
        wicket_candidate = _blend_wickets(
            selection_wicket_base[mask],
            selection_wicket_proposed[mask],
            wicket_alpha,
        )
        run_safety_alphas[phase] = (
            run_alpha
            if np.mean(
                np.abs(run_candidate - selection.loc[mask, "runs_in_over"])
            )
            < np.mean(
                np.abs(
                    selection_run_base[mask]
                    - selection.loc[mask, "runs_in_over"]
                )
            )
            else 0.0
        )
        wicket_safety_alphas[phase] = (
            wicket_alpha
            if brier_score_loss(
                selection.loc[mask, "wicket_in_over"], wicket_candidate
            )
            < brier_score_loss(
                selection.loc[mask, "wicket_in_over"],
                selection_wicket_base[mask],
            )
            else 0.0
        )
    holdout_run_safe = _apply_phase_policy(
        holdout,
        holdout_run_base,
        holdout_run_proposed,
        run_safety_alphas,
        wicket=False,
        smooth=False,
    )
    holdout_wicket_safe = _apply_phase_policy(
        holdout,
        holdout_wicket_base,
        holdout_wicket_proposed,
        wicket_safety_alphas,
        wicket=True,
        smooth=False,
    )

    # Stage 3: independently tune each phase with continuous evidence
    # shrinkage; unsupported players move gradually away from neutral.
    run_phase_alphas = _phase_alphas(
        selection,
        selection["runs_in_over"].to_numpy(),
        selection_run_base,
        selection_run_proposed,
        wicket=False,
        smooth=True,
    )
    wicket_phase_alphas = _phase_alphas(
        selection,
        selection["wicket_in_over"].to_numpy(),
        selection_wicket_base,
        selection_wicket_proposed,
        wicket=True,
        smooth=True,
    )
    holdout_run_final = _apply_phase_policy(
        holdout,
        holdout_run_base,
        holdout_run_proposed,
        run_phase_alphas,
        wicket=False,
        smooth=True,
    )
    holdout_wicket_final = _apply_phase_policy(
        holdout,
        holdout_wicket_base,
        holdout_wicket_proposed,
        wicket_phase_alphas,
        wicket=True,
        smooth=True,
    )

    existing_alerts = pd.read_csv(
        root / "models/candidates/ipl_wicket_v7_context/holdout_predictions.csv"
    )["candidate_alert"]
    alert_rate = float(existing_alerts.mean())
    actual_runs = holdout["runs_in_over"].to_numpy()
    actual_wicket = holdout["wicket_in_over"].to_numpy()
    overall_alert_count = round(alert_rate * len(holdout))
    stage_metrics = {
        "situation_baseline": _metrics(
            actual_runs,
            holdout_run_base,
            actual_wicket,
            holdout_wicket_base,
            overall_alert_count,
        ),
        "stage_1_oof_global": _metrics(
            actual_runs,
            holdout_run_candidate,
            actual_wicket,
            holdout_wicket_candidate,
            overall_alert_count,
        ),
        "stage_2_safety_routing": _metrics(
            actual_runs,
            holdout_run_safe,
            actual_wicket,
            holdout_wicket_safe,
            overall_alert_count,
        ),
        "stage_3_phase_smooth": _metrics(
            actual_runs,
            holdout_run_final,
            actual_wicket,
            holdout_wicket_final,
            overall_alert_count,
        ),
    }

    definitions = {
        "overall": np.ones(len(holdout), dtype=bool),
        "period_2025": holdout["match_date"].dt.year.eq(2025).to_numpy(),
        "period_2026": holdout["match_date"].dt.year.eq(2026).to_numpy(),
        "announced_offline_proxy": np.ones(len(holdout), dtype=bool),
        "unannounced_live_fallback": np.zeros(len(holdout), dtype=bool),
        "new_spells": holdout["spell_state"].eq("new_spell").to_numpy(),
        "returning_spells": holdout["spell_state"].eq("returning_spell").to_numpy(),
        "established_partnerships": holdout[
            "partnership_legal_ball_age"
        ].ge(24).to_numpy(),
        "powerplay": holdout["phase"].eq("powerplay").to_numpy(),
        "middle": holdout["phase"].eq("middle").to_numpy(),
        "death": holdout["phase"].eq("death").to_numpy(),
        "supported_bowler_histories": holdout[
            "bowler_history_supported"
        ].astype(bool).to_numpy(),
        "unsupported_bowler_histories": ~holdout[
            "bowler_history_supported"
        ].astype(bool).to_numpy(),
    }
    segments = {}
    for name, mask in definitions.items():
        if not mask.any():
            segments[name] = {
                "rows": 0,
                "policy": "situation prediction retained when live bowler is unannounced",
            }
            continue
        count = round(alert_rate * int(mask.sum()))
        segments[name] = {
            "baseline": _metrics(
                actual_runs[mask],
                holdout_run_base[mask],
                actual_wicket[mask],
                holdout_wicket_base[mask],
                count,
            ),
            "candidate": _metrics(
                actual_runs[mask],
                holdout_run_candidate[mask],
                actual_wicket[mask],
                holdout_wicket_candidate[mask],
                count,
            ),
        }

    gates = {
        "overall_wicket_brier_improves": (
            segments["overall"]["candidate"]["wicket_brier"]
            < segments["overall"]["baseline"]["wicket_brier"]
        ),
        "2025_wicket_brier_improves": (
            segments["period_2025"]["candidate"]["wicket_brier"]
            < segments["period_2025"]["baseline"]["wicket_brier"]
        ),
        "2026_wicket_brier_improves": (
            segments["period_2026"]["candidate"]["wicket_brier"]
            < segments["period_2026"]["baseline"]["wicket_brier"]
        ),
        "precision_improves_same_alert_volume": (
            segments["overall"]["candidate"][
                "precision_at_existing_alert_volume"
            ]
            > segments["overall"]["baseline"][
                "precision_at_existing_alert_volume"
            ]
        ),
        "run_mae_non_regression": (
            segments["overall"]["candidate"]["run_mae"]
            <= segments["overall"]["baseline"]["run_mae"]
        ),
        "fixed_width_3_coverage_non_regression": (
            segments["overall"]["candidate"]["fixed_width_3_run_coverage"]
            >= segments["overall"]["baseline"]["fixed_width_3_run_coverage"]
        ),
        "live_availability_validated": False,
    }
    report = {
        "candidate_version": "announced_bowler_current_spell_v3",
        "candidate_only": True,
        "production_changed": False,
        "offline_development_source": "Cricsheet first-delivery bowler",
        "live_source_pending": "TOI when an actual match starts",
        "identity_used_as_model_feature": False,
        "chronology": {
            "training": "through 2023",
            "calibration": "first half 2024",
            "blend_selection": "second half 2024",
            "holdout": [2025, 2026],
        },
        "bounds": {
            "maximum_run_delta": MAX_RUN_DELTA,
            "maximum_wicket_logit_delta": MAX_WICKET_LOGIT_DELTA,
        },
        "selected_run_alpha": run_alpha,
        "selected_wicket_alpha": wicket_alpha,
        "safety_run_alphas": run_safety_alphas,
        "safety_wicket_alphas": wicket_safety_alphas,
        "phase_smooth_run_alphas": run_phase_alphas,
        "phase_smooth_wicket_alphas": wicket_phase_alphas,
        "stage_metrics": stage_metrics,
        "selected_stage": "stage_1_oof_global",
        "stage_decisions": {
            "stage_1_oof_global": "accept",
            "stage_2_safety_routing": "reject_worse_than_stage_1",
            "stage_3_phase_smooth": "reject_worse_than_stage_1",
        },
        "run_alpha_trials": run_trials,
        "wicket_alpha_trials": wicket_trials,
        "segments": segments,
        "promotion_gates": gates,
        "decision": (
            "research_pass_await_live_availability"
            if all(value for key, value in gates.items() if key != "live_availability_validated")
            else "reject_keep_research"
        ),
    }
    destination = root / "models/candidates/announced_bowler_current_spell_v3"
    destination.mkdir(parents=True, exist_ok=True)
    run_base.save_model(destination / "situation_runs.cbm")
    run_adjuster.save_model(destination / "run_adjustment.cbm")
    wicket_base.save_model(destination / "situation_wicket.cbm")
    wicket_adjuster.save_model(destination / "wicket_adjustment.cbm")
    joblib.dump(base_calibrator, destination / "situation_wicket_platt.pkl")
    joblib.dump(proposed_calibrator, destination / "wicket_adjustment_platt.pkl")
    (destination / "runtime_manifest.json").write_text(
        json.dumps(
            {
                "candidate_version": "announced_bowler_current_spell_v3",
                "selected_stage": "stage_1_oof_global",
                "run_alpha": run_alpha,
                "wicket_alpha": wicket_alpha,
                "maximum_run_delta": MAX_RUN_DELTA,
                "maximum_wicket_logit_delta": MAX_WICKET_LOGIT_DELTA,
                "adjustment_context": ADJUSTMENT_CONTEXT,
                "adjustment_categorical": ADJUSTMENT_CATEGORICAL,
                "publishing_enabled": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    profile_columns = [
        "offline_next_bowler",
        "phase",
        "bowler_phase_history_balls",
        "bowler_phase_history_runs_conceded",
        "bowler_phase_history_wickets",
        "bowler_phase_history_dot_rate",
        "bowler_phase_history_boundary_concession_rate",
        "bowler_phase_history_strike_rate",
        "bowler_phase_history_economy",
    ]
    data.sort_values(
        ["match_date", "source_file", "innings", "over"]
    ).groupby(
        ["offline_next_bowler", "phase"], sort=False
    ).tail(1)[profile_columns].to_csv(
        destination / "bowler_phase_profiles.csv", index=False
    )
    pd.DataFrame(
        {
            "source_file": holdout["source_file"],
            "match_date": holdout["match_date"].astype(str),
            "innings": holdout["innings"],
            "over": holdout["over"],
            "actual_runs": actual_runs,
            "baseline_runs": holdout_run_base,
            "stage_1_runs": holdout_run_candidate,
            "stage_2_runs": holdout_run_safe,
            "candidate_runs": holdout_run_candidate,
            "actual_wicket": actual_wicket,
            "baseline_wicket": holdout_wicket_base,
            "stage_1_wicket": holdout_wicket_candidate,
            "stage_2_wicket": holdout_wicket_safe,
            "candidate_wicket": holdout_wicket_candidate,
        }
    ).to_csv(destination / "holdout_predictions.csv", index=False)
    (destination / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
