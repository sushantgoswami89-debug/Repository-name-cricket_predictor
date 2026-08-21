"""Train detailed four, six, cluster, and wicket-shock boundary gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from train_candidate_v33_response import _model
from train_candidate_v37_boundary_venue import _boundary_probabilities
from train_sharp_range_candidate import (
    MAX_RUN_CLASS,
    _best_bands,
    _phase_report,
    _sequence_failures,
)
from train_validate_candidate_v3 import CATEGORICAL_FEATURES, V3_FEATURES, _features

KEYS = ["source_file", "match_date", "innings", "over"]
V32_HIT_RATE = 0.31376664261821996
V39_HIT_RATE = 0.32048927657899917
DETAIL_TARGETS = (
    "has_four",
    "has_six",
    "multiple_boundaries",
    "boundary_wicket_shock",
)


def _feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = _features(frame, columns)
    if "venue_track_type" in result:
        result["venue_track_type"] = result["venue_track_type"].astype("category")
    return result


def _binary_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=45,
        learning_rate=0.08,
        num_leaves=19,
        max_depth=5,
        min_child_samples=140,
        reg_lambda=1.5,
        random_state=42,
        verbose=-1,
    )


def _detail_probabilities(
    training: pd.DataFrame, holdout: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, lgb.LGBMClassifier]]:
    train_output = pd.DataFrame(index=training.index)
    holdout_output = pd.DataFrame(index=holdout.index)
    models: dict[str, lgb.LGBMClassifier] = {}
    years = training["match_date"].dt.year.to_numpy()
    blocks = ((2014, 2017), (2018, 2020), (2021, 2023))
    for target_name in DETAIL_TARGETS:
        target = training[target_name].astype(int).to_numpy()
        oof = np.full(len(training), target.mean(), dtype=float)
        for start, end in blocks:
            fit = years < start
            predict = (years >= start) & (years <= end)
            if fit.sum() < 5000 or not predict.any():
                continue
            fold_model = _binary_model()
            fold_model.fit(
                _feature_frame(training.loc[fit], columns),
                target[fit],
                categorical_feature=CATEGORICAL_FEATURES + ["venue_track_type"],
            )
            oof[predict] = fold_model.predict_proba(
                _feature_frame(training.loc[predict], columns)
            )[:, 1]
        model = _binary_model()
        model.fit(
            _feature_frame(training, columns),
            target,
            categorical_feature=CATEGORICAL_FEATURES + ["venue_track_type"],
        )
        feature_name = f"boundary_detail_{target_name}_probability"
        train_output[feature_name] = oof
        holdout_output[feature_name] = model.predict_proba(
            _feature_frame(holdout, columns)
        )[:, 1]
        models[target_name] = model
    return train_output, holdout_output, models


def train(project_root: Path, output_dir: Path) -> dict[str, Any]:
    base = pd.read_csv(project_root / "data/candidates/v3/verified_training_overs.csv")
    venue = pd.read_csv(project_root / "data/candidates/v3.7/venue_track_features.csv")
    components = pd.read_csv(project_root / "data/candidates/v3.4/over_components.csv")
    for frame in (base, venue, components):
        frame["match_date"] = frame["match_date"].astype(str)
    components["boundary_count"] = components["fours"] + components["sixes"]
    components["has_four"] = (components["fours"] > 0).astype(int)
    components["has_six"] = (components["sixes"] > 0).astype(int)
    components["multiple_boundaries"] = (components["boundary_count"] >= 2).astype(int)
    components["boundary_wicket_shock"] = (
        (components["boundary_count"] > 0) & (components["wickets"] > 0)
    ).astype(int)
    data = base.merge(venue, on=KEYS, validate="one_to_one").merge(
        components[KEYS + ["boundary_count", *DETAIL_TARGETS]],
        on=KEYS,
        validate="one_to_one",
    )
    data["match_date"] = pd.to_datetime(data["match_date"])
    training = data[data["match_date"].dt.year <= 2023].reset_index(drop=True)
    holdout = data[data["match_date"].dt.year >= 2025].reset_index(drop=True)
    gate_columns = V3_FEATURES + [
        column for column in venue if column.startswith("venue_track_")
    ]
    detail_train, detail_holdout, detail_models = _detail_probabilities(
        training, holdout, gate_columns
    )
    for column in detail_train:
        training[column] = detail_train[column]
        holdout[column] = detail_holdout[column]
    detail_columns = list(detail_train.columns)
    count_train, count_holdout, count_model = _boundary_probabilities(training, holdout)
    count_columns = [
        "boundary_gate_none_probability",
        "boundary_gate_one_probability",
        "boundary_gate_multiple_probability",
        "boundary_gate_entropy",
    ]
    for index, name in enumerate(count_columns[:3]):
        training[name] = count_train[:, index]
        holdout[name] = count_holdout[:, index]
    for frame, probability in ((training, count_train), (holdout, count_holdout)):
        frame[count_columns[3]] = -np.sum(
            probability * np.log(np.maximum(probability, 1e-9)), axis=1
        )
    groups = {
        "four_six_detail_gate": detail_columns,
        "count_plus_boundary_detail": count_columns + detail_columns,
    }
    actual = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
    baseline = (
        pd.read_csv(
            project_root / "models/candidates/v3.2_sharp_range/holdout_sharp_ranges.csv"
        )["sharp_2_hit"]
        .astype(bool)
        .to_numpy()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for name, additions in groups.items():
        columns = V3_FEATURES + additions
        model = _model()
        model.fit(
            _feature_frame(training, columns),
            np.minimum(training["runs_in_over"], MAX_RUN_CLASS),
            categorical_feature=CATEGORICAL_FEATURES,
        )
        probability = model.predict_proba(_feature_frame(holdout, columns))
        low, high, mass = _best_bands(probability, 2)
        hit = (actual >= low) & (actual <= high)
        candidate_wins = int((hit & ~baseline).sum())
        baseline_wins = int((~hit & baseline).sum())
        reports[name] = {
            "features_added": additions,
            "two_run_hit_rate": float(hit.mean()),
            "improvement_over_v32_percentage_points": float(
                (hit.mean() - V32_HIT_RATE) * 100
            ),
            "improvement_over_v39_percentage_points": float(
                (hit.mean() - V39_HIT_RATE) * 100
            ),
            "paired_vs_v32_p_value": float(
                binomtest(
                    min(candidate_wins, baseline_wins),
                    candidate_wins + baseline_wins,
                    0.5,
                ).pvalue
            ),
            "mean_band_probability": float(mass.mean()),
            "phase": _phase_report(holdout, actual, low, high),
        }
        joblib.dump(model, output_dir / f"{name}_model.pkl")
        joblib.dump(columns, output_dir / f"{name}_feature_cols.pkl")
    winner_name = max(reports, key=lambda key: reports[key]["two_run_hit_rate"])
    winner = reports[winner_name]
    gates = {
        "accuracy_at_least_32_38pct": winner["two_run_hit_rate"] >= 0.3238,
        "beats_v39": winner["two_run_hit_rate"] > V39_HIT_RATE,
        "paired_vs_v32_p_below_0_01": winner["paired_vs_v32_p_value"] < 0.01,
        "complete_match_sequence_failures_zero": _sequence_failures(holdout) == 0,
    }
    detail_accuracy = {
        name: float(
            (
                (detail_holdout[f"boundary_detail_{name}_probability"] >= 0.5)
                == holdout[name].astype(bool)
            ).mean()
        )
        for name in DETAIL_TARGETS
    }
    report = {
        "candidate_version": "v3.11_boundary_detail",
        "model_scope": "candidate_only",
        "production_models_changed": False,
        "verified_rows": len(data),
        "holdout_rows": len(holdout),
        "holdout_matches": int(holdout["source_file"].nunique()),
        "detail_gate_accuracy": detail_accuracy,
        "ablations": reports,
        "winner": winner_name,
        "winner_metrics": winner,
        "promotion_gates": gates,
        "integration_recommended": all(gates.values()),
    }
    joblib.dump(count_model, output_dir / "boundary_count_submodel.pkl")
    for name, model in detail_models.items():
        joblib.dump(model, output_dir / f"{name}_submodel.pkl")
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            train(root, root / "models/candidates/v3.11_boundary_detail"), indent=2
        )
    )
