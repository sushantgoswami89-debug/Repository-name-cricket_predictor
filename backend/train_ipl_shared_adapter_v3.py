"""Train shared T20 run/wicket knowledge with a 2023+ IPL adapter."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from train_candidate_v33_response import _model as run_model
from train_sharp_range_candidate import MAX_RUN_CLASS, _best_bands, _phase_report
from train_validate_candidate_v3 import CATEGORICAL_FEATURES, V3_FEATURES, _features

KEYS = ["source_file", "match_date", "innings", "over"]


def _binary_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=55,
        learning_rate=0.06,
        num_leaves=19,
        max_depth=5,
        min_child_samples=100,
        reg_lambda=1.5,
        random_state=42,
        verbose=-1,
    )


def _features_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return _features(frame, columns)


def train(
    project_root: Path,
    output_dir: Path,
    *,
    ipl_only_backbone: bool = False,
) -> dict[str, Any]:
    base = pd.read_csv(project_root / "data/candidates/v3/verified_training_overs.csv")
    adapter = pd.read_csv(project_root / "data/candidates/ipl_v3/adapter_features.csv")
    for frame in (base, adapter):
        frame["match_date"] = frame["match_date"].astype(str)
    base["match_date"] = pd.to_datetime(base["match_date"])
    adapter["match_date"] = pd.to_datetime(adapter["match_date"])
    shared_training = base[base["match_date"].dt.year <= 2022]
    if ipl_only_backbone:
        ipl_source_files = {
            path.name
            for path in (project_root / "data/raw/cricsheet/ipl").glob("*.json")
        }
        shared_training = shared_training[
            shared_training["source_file"].isin(ipl_source_files)
        ]
    shared_training = shared_training.reset_index(drop=True)
    ipl = base.merge(adapter, on=KEYS, how="inner", validate="one_to_one")
    adapter_training = ipl[ipl["match_date"].dt.year.between(2023, 2024)].reset_index(
        drop=True
    )
    holdout = ipl[ipl["match_date"].dt.year >= 2025].reset_index(drop=True)

    shared_runs = run_model()
    shared_runs.fit(
        _features_frame(shared_training, V3_FEATURES),
        np.minimum(shared_training["runs_in_over"], MAX_RUN_CLASS),
        categorical_feature=CATEGORICAL_FEATURES,
    )
    shared_wicket = _binary_model()
    shared_wicket.fit(
        _features_frame(shared_training, V3_FEATURES),
        shared_training["wicket_in_over"],
        categorical_feature=CATEGORICAL_FEATURES,
    )

    def add_shared_outputs(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        probability = shared_runs.predict_proba(
            _features_frame(frame, V3_FEATURES)
        )
        low, high, mass = _best_bands(probability, 2)
        classes = np.arange(probability.shape[1])
        frame["shared_t20_expected_runs"] = probability @ classes
        frame["shared_t20_band_low"] = low
        frame["shared_t20_band_high"] = high
        frame["shared_t20_band_probability"] = mass
        frame["shared_t20_run_entropy"] = -np.sum(
            probability * np.log(np.maximum(probability, 1e-9)), axis=1
        )
        wicket_probability = shared_wicket.predict_proba(
            _features_frame(frame, V3_FEATURES)
        )[:, 1]
        frame["shared_t20_wicket_probability"] = wicket_probability
        return probability, wicket_probability

    _, _ = add_shared_outputs(adapter_training)
    shared_holdout_runs, shared_holdout_wicket = add_shared_outputs(holdout)
    shared_columns = [column for column in holdout if column.startswith("shared_t20_")]
    adapter_columns = [
        column
        for column in adapter
        if column.startswith("ipl_adapter_")
        and column
        not in {"ipl_adapter_batting_team_win", "ipl_adapter_result_known"}
    ]
    model_columns = V3_FEATURES + shared_columns + adapter_columns

    adjusted_runs = run_model()
    adjusted_runs.fit(
        _features_frame(adapter_training, model_columns),
        np.minimum(adapter_training["runs_in_over"], MAX_RUN_CLASS),
        categorical_feature=CATEGORICAL_FEATURES,
    )
    adjusted_probability = adjusted_runs.predict_proba(
        _features_frame(holdout, model_columns)
    )
    adjusted_low, adjusted_high, adjusted_mass = _best_bands(
        adjusted_probability, 2
    )
    shared_low, shared_high, shared_mass = _best_bands(shared_holdout_runs, 2)
    actual_runs = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
    shared_hit = (actual_runs >= shared_low) & (actual_runs <= shared_high)
    adjusted_hit = (actual_runs >= adjusted_low) & (actual_runs <= adjusted_high)

    adjusted_wicket = _binary_model()
    adjusted_wicket.fit(
        _features_frame(adapter_training, model_columns),
        adapter_training["wicket_in_over"],
        categorical_feature=CATEGORICAL_FEATURES,
    )
    adjusted_wicket_probability = adjusted_wicket.predict_proba(
        _features_frame(holdout, model_columns)
    )[:, 1]
    wicket_actual = holdout["wicket_in_over"].astype(int).to_numpy()

    known_training = adapter_training["ipl_adapter_result_known"] == 1
    known_holdout = holdout["ipl_adapter_result_known"] == 1
    win_model = _binary_model()
    win_model.fit(
        _features_frame(adapter_training.loc[known_training], model_columns),
        adapter_training.loc[known_training, "ipl_adapter_batting_team_win"],
        categorical_feature=CATEGORICAL_FEATURES,
    )
    win_probability = win_model.predict_proba(
        _features_frame(holdout.loc[known_holdout], model_columns)
    )[:, 1]
    win_actual = holdout.loc[
        known_holdout, "ipl_adapter_batting_team_win"
    ].astype(int).to_numpy()

    sample = _features_frame(holdout.tail(1), model_columns)
    adjusted_runs.predict_proba(sample)
    adjusted_wicket.predict_proba(sample)
    win_model.predict_proba(sample)
    started = time.perf_counter()
    for _ in range(300):
        adjusted_runs.predict_proba(sample)
        adjusted_wicket.predict_proba(sample)
        win_model.predict_proba(sample)
    latency = (time.perf_counter() - started) * 1000 / 300
    run_improvement = float((adjusted_hit.mean() - shared_hit.mean()) * 100)
    shared_wicket_brier = float(
        brier_score_loss(wicket_actual, shared_holdout_wicket)
    )
    adjusted_wicket_brier = float(
        brier_score_loss(wicket_actual, adjusted_wicket_probability)
    )
    win_brier = float(brier_score_loss(win_actual, win_probability))
    win_log_loss = float(log_loss(win_actual, win_probability))
    gates = {
        "run_accuracy_at_least_32_38pct": bool(
            adjusted_hit.mean() >= 0.3238
        ),
        "run_adapter_improves_shared_backbone": bool(run_improvement > 0),
        "wicket_brier_improves_shared_backbone": bool(
            adjusted_wicket_brier < shared_wicket_brier
        ),
        "win_brier_below_naive_0_25": bool(win_brier < 0.25),
        "all_heads_latency_below_50ms": bool(latency < 50.0),
    }
    report = {
        "candidate_version": (
            "ipl_engine_v4_ipl_only_adapter"
            if ipl_only_backbone
            else "ipl_engine_v3_shared_t20_adapter"
        ),
        "backbone_scope": "ipl_only" if ipl_only_backbone else "ipl_and_t20i",
        "model_scope": "candidate_only_ipl",
        "production_models_changed": False,
        "shared_backbone_training_rows_through_2022": len(shared_training),
        "ipl_adapter_training_rows_2023_2024": len(adapter_training),
        "ipl_holdout_rows_2025_plus": len(holdout),
        "ipl_holdout_matches": int(holdout["source_file"].nunique()),
        "run_head": {
            "shared_backbone_hit_rate": float(shared_hit.mean()),
            "adjusted_hit_rate": float(adjusted_hit.mean()),
            "improvement_percentage_points": run_improvement,
            "mean_adjusted_band_probability": float(adjusted_mass.mean()),
            "mean_shared_band_probability": float(shared_mass.mean()),
            "phase": _phase_report(
                holdout, actual_runs, adjusted_low, adjusted_high
            ),
        },
        "wicket_head": {
            "shared_backbone_brier": shared_wicket_brier,
            "adjusted_brier": adjusted_wicket_brier,
            "brier_improvement": shared_wicket_brier - adjusted_wicket_brier,
        },
        "win_head": {
            "rows": int(known_holdout.sum()),
            "brier_score": win_brier,
            "log_loss": win_log_loss,
            "accuracy_at_0_5": float(
                ((win_probability >= 0.5) == win_actual).mean()
            ),
        },
        "mean_three_head_prediction_ms": latency,
        "promotion_gates": gates,
        "integration_recommended": all(gates.values()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in (
        ("shared_runs", shared_runs),
        ("shared_wicket", shared_wicket),
        ("ipl_adjusted_runs", adjusted_runs),
        ("ipl_adjusted_wicket", adjusted_wicket),
        ("ipl_win", win_model),
    ):
        joblib.dump(model, output_dir / f"{name}_model.pkl")
    joblib.dump(model_columns, output_dir / "adapter_feature_cols.pkl")
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            train(
                root,
                root / "models/candidates/ipl_engine_v4_ipl_only_adapter",
                ipl_only_backbone=True,
            ),
            indent=2,
        )
    )
