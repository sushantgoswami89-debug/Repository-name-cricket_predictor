"""Test leakage-safe shock probabilities as inputs to the IPL v5 run model."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from train_candidate_v33_response import _model as run_model
from train_ipl_phase_intelligence_v5 import CATEGORICAL, KEYS, _load, _x
from train_sharp_range_candidate import MAX_RUN_CLASS, _best_bands

SHOCK_COLUMNS = [
    "shock_boundary_0_probability",
    "shock_boundary_1_probability",
    "shock_boundary_2plus_probability",
    "shock_six_probability",
    "shock_wicket_probability",
    "shock_extras_spike_probability",
]


def _classifier(multiclass: bool = False) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass" if multiclass else "binary",
        num_class=3 if multiclass else None,
        n_estimators=55,
        learning_rate=0.06,
        num_leaves=19,
        max_depth=5,
        min_child_samples=120,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )


def _targets(data: pd.DataFrame) -> dict[str, np.ndarray]:
    boundaries = np.minimum((data["fours"] + data["sixes"]).to_numpy(), 2)
    return {
        "boundary": boundaries.astype(int),
        "six": (data["sixes"].to_numpy() > 0).astype(int),
        "wicket": (data["wickets"].to_numpy() > 0).astype(int),
        "extras": (data["extras_runs"].to_numpy() >= 2).astype(int),
    }


def _fit_shocks(
    frame: pd.DataFrame, columns: list[str]
) -> dict[str, lgb.LGBMClassifier]:
    targets = _targets(frame)
    categories = [column for column in CATEGORICAL if column in columns]
    models = {
        "boundary": _classifier(multiclass=True),
        "six": _classifier(),
        "wicket": _classifier(),
        "extras": _classifier(),
    }
    x = _x(frame, columns)
    for name, model in models.items():
        model.fit(x, targets[name], categorical_feature=categories)
    return models


def _predict_shocks(
    models: dict[str, lgb.LGBMClassifier],
    frame: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    x = _x(frame, columns)
    boundary = models["boundary"].predict_proba(x)
    return np.column_stack(
        [
            boundary,
            models["six"].predict_proba(x)[:, 1],
            models["wicket"].predict_proba(x)[:, 1],
            models["extras"].predict_proba(x)[:, 1],
        ]
    )


def _oof_shocks(training: pd.DataFrame, columns: list[str]) -> np.ndarray:
    years = training["match_date"].dt.year.to_numpy()
    output = np.zeros((len(training), len(SHOCK_COLUMNS)))
    default = np.array([0.45, 0.35, 0.20, 0.28, 0.26, 0.12])
    output[:] = default
    for year in sorted(np.unique(years)):
        fit_mask = years < year
        predict_mask = years == year
        if fit_mask.sum() < 3000:
            continue
        models = _fit_shocks(training.loc[fit_mask], columns)
        output[predict_mask] = _predict_shocks(
            models, training.loc[predict_mask], columns
        )
    return output


def _shock_metrics(
    training: pd.DataFrame,
    frame: pd.DataFrame,
    probability: np.ndarray,
) -> dict[str, Any]:
    train_targets = _targets(training)
    actual = _targets(frame)
    boundary_prior = np.bincount(train_targets["boundary"], minlength=3) / len(training)
    result: dict[str, Any] = {
        "rows": len(frame),
        "boundary": {
            "log_loss": float(log_loss(actual["boundary"], probability[:, :3])),
            "prior_log_loss": float(
                log_loss(
                    actual["boundary"],
                    np.tile(boundary_prior, (len(frame), 1)),
                )
            ),
            "accuracy": float(
                (np.argmax(probability[:, :3], axis=1) == actual["boundary"]).mean()
            ),
        },
    }
    for index, name in enumerate(("six", "wicket", "extras"), start=3):
        prior = float(train_targets[name].mean())
        result[name] = {
            "brier": float(brier_score_loss(actual[name], probability[:, index])),
            "prior_brier": float(
                brier_score_loss(actual[name], np.full(len(frame), prior))
            ),
            "auc": float(roc_auc_score(actual[name], probability[:, index])),
        }
    return result


def _run_models(
    training: pd.DataFrame,
    holdout: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, float, dict[str, Any]]:
    probability = np.zeros((len(holdout), MAX_RUN_CLASS + 1))
    latency_models: dict[str, Any] = {}
    categories = [column for column in CATEGORICAL if column in columns]
    for phase in ("powerplay", "middle", "death"):
        train_mask = training["phase"] == phase
        test_mask = holdout["phase"] == phase
        model = run_model()
        model.fit(
            _x(training.loc[train_mask], columns),
            np.minimum(training.loc[train_mask, "runs_in_over"], MAX_RUN_CLASS),
            categorical_feature=categories,
        )
        phase_probability = model.predict_proba(_x(holdout.loc[test_mask], columns))
        probability[
            np.ix_(
                np.flatnonzero(test_mask.to_numpy()),
                model.classes_.astype(int),
            )
        ] = phase_probability
        latency_models[phase] = model
    low, high, mass = _best_bands(probability, 2)
    actual = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
    hit = (actual >= low) & (actual <= high)
    sample = holdout.tail(1)
    phase = str(sample.iloc[0]["phase"])
    started = time.perf_counter()
    for _ in range(300):
        latency_models[phase].predict_proba(_x(sample, columns))
    latency = (time.perf_counter() - started) * 1000 / 300
    report = {
        "hit_rate": float(hit.mean()),
        "mean_band_probability": float(mass.mean()),
        "phase": {
            name: float(hit[holdout["phase"].to_numpy() == name].mean())
            for name in ("powerplay", "middle", "death")
        },
    }
    return hit, latency, report


def train(root: Path, output_dir: Path) -> dict[str, Any]:
    data, _ = _load(root)
    components = pd.read_csv(
        root / "data/candidates/v3.4/over_components.csv",
        usecols=KEYS + ["extras_runs"],
    )
    components["match_date"] = components["match_date"].astype(str)
    data["match_date"] = data["match_date"].astype(str)
    data = data.merge(components, on=KEYS, validate="one_to_one")
    data["match_date"] = pd.to_datetime(data["match_date"])
    training = data[data["match_date"].dt.year <= 2024].reset_index(drop=True)
    holdout = data[data["match_date"].dt.year >= 2025].reset_index(drop=True)
    base_columns = joblib.load(
        root / "models/candidates/ipl_engine_v5_phase_intelligence/feature_cols.pkl"
    )
    oof = _oof_shocks(training, base_columns)
    final_shocks = _fit_shocks(training, base_columns)
    holdout_shocks = _predict_shocks(final_shocks, holdout, base_columns)
    for index, name in enumerate(SHOCK_COLUMNS):
        training[name] = oof[:, index]
        holdout[name] = holdout_shocks[:, index]
    _, base_latency, base_report = _run_models(training, holdout, base_columns)
    _, shock_latency, shock_report = _run_models(
        training, holdout, base_columns + SHOCK_COLUMNS
    )
    year = holdout["match_date"].dt.year
    shock_diagnostics = {
        str(test_year): _shock_metrics(
            training,
            holdout[year == test_year],
            holdout_shocks[year.to_numpy() == test_year],
        )
        for test_year in (2025, 2026)
    }
    improvement = (shock_report["hit_rate"] - base_report["hit_rate"]) * 100
    gates = {
        "shock_probabilities_improve_range": bool(improvement > 0),
        "range_accuracy_at_least_32_38pct": bool(shock_report["hit_rate"] >= 0.3238),
        "all_shock_heads_beat_prior_on_2026": bool(
            shock_diagnostics["2026"]["boundary"]["log_loss"]
            < shock_diagnostics["2026"]["boundary"]["prior_log_loss"]
            and all(
                shock_diagnostics["2026"][name]["brier"]
                < shock_diagnostics["2026"][name]["prior_brier"]
                for name in ("six", "wicket", "extras")
            )
        ),
        "latency_below_50ms": bool(shock_latency < 50),
    }
    report = {
        "candidate_version": "ipl_v10_shock_probability_gate",
        "production_models_changed": False,
        "training_rows_through_2024": len(training),
        "holdout_rows_2025_plus": len(holdout),
        "shock_diagnostics": shock_diagnostics,
        "base_retrained": base_report,
        "shock_augmented": shock_report,
        "improvement_percentage_points": improvement,
        "base_prediction_ms": base_latency,
        "shock_augmented_prediction_ms": shock_latency,
        "promotion_gates": gates,
        "integration_recommended": all(gates.values()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in final_shocks.items():
        joblib.dump(model, output_dir / f"{name}_shock_model.pkl")
    joblib.dump(base_columns + SHOCK_COLUMNS, output_dir / "feature_cols.pkl")
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    destination = project_root / "models/candidates/ipl_v10_shock_gate"
    print(json.dumps(train(project_root, destination), indent=2))
