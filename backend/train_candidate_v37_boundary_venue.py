"""Train boundary-gated and venue-track Candidate v3.7 ablations."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from train_candidate_v33_response import _model as run_model
from train_sharp_range_candidate import (
    MAX_RUN_CLASS,
    _best_bands,
    _phase_report,
    _sequence_failures,
)
from train_validate_candidate_v3 import CATEGORICAL_FEATURES, V3_FEATURES, _features

KEYS = ["source_file", "match_date", "innings", "over"]
V32_HIT_RATE = 0.31376664261821996


def _feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = _features(frame, columns)
    if "venue_track_type" in result:
        result["venue_track_type"] = result["venue_track_type"].astype("category")
    return result


def _boundary_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=55,
        learning_rate=0.08,
        num_leaves=21,
        max_depth=6,
        min_child_samples=120,
        reg_lambda=1.5,
        random_state=42,
        verbose=-1,
    )


def _boundary_probabilities(
    training: pd.DataFrame, holdout: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, lgb.LGBMClassifier]:
    """Generate expanding-time training probabilities and frozen-test probabilities."""

    columns = V3_FEATURES + [
        column for column in training if column.startswith("venue_track_")
    ]
    categories = CATEGORICAL_FEATURES + ["venue_track_type"]
    target = np.minimum(training["boundary_count"].to_numpy(), 2)
    prior = np.bincount(target, minlength=3) / len(target)
    oof = np.tile(prior, (len(training), 1))
    years = training["match_date"].dt.year.to_numpy()
    for year in sorted(np.unique(years)):
        fit_mask = years < year
        predict_mask = years == year
        if fit_mask.sum() < 5000 or len(np.unique(target[fit_mask])) < 3:
            continue
        model = _boundary_model()
        model.fit(
            _feature_frame(training.loc[fit_mask], columns),
            target[fit_mask],
            categorical_feature=categories,
        )
        oof[predict_mask] = model.predict_proba(
            _feature_frame(training.loc[predict_mask], columns)
        )
    model = _boundary_model()
    model.fit(
        _feature_frame(training, columns), target, categorical_feature=categories
    )
    return oof, model.predict_proba(_feature_frame(holdout, columns)), model


def train(project_root: Path, output_dir: Path) -> dict[str, Any]:
    base = pd.read_csv(project_root / "data/candidates/v3/verified_training_overs.csv")
    venue = pd.read_csv(project_root / "data/candidates/v3.7/venue_track_features.csv")
    components = pd.read_csv(project_root / "data/candidates/v3.4/over_components.csv")
    for frame in (base, venue, components):
        frame["match_date"] = frame["match_date"].astype(str)
    components["boundary_count"] = components["fours"] + components["sixes"]
    data = base.merge(venue, on=KEYS, validate="one_to_one").merge(
        components[KEYS + ["boundary_count"]], on=KEYS, validate="one_to_one"
    )
    if len(data) != len(base):
        raise ValueError("v3.7 features do not reconcile with verified data.")
    data["match_date"] = pd.to_datetime(data["match_date"])
    training = data[data["match_date"].dt.year <= 2023].reset_index(drop=True)
    holdout = data[data["match_date"].dt.year >= 2025].reset_index(drop=True)
    boundary_train, boundary_holdout, boundary_model = _boundary_probabilities(
        training, holdout
    )
    boundary_feature_columns = V3_FEATURES + [
        column for column in training if column.startswith("venue_track_")
    ]
    boundary_sample = _feature_frame(holdout.tail(1), boundary_feature_columns)
    boundary_model.predict_proba(boundary_sample)
    boundary_started = time.perf_counter()
    for _ in range(300):
        boundary_model.predict_proba(boundary_sample)
    boundary_latency_ms = (time.perf_counter() - boundary_started) * 1000 / 300
    boundary_columns = [
        "boundary_gate_none_probability",
        "boundary_gate_one_probability",
        "boundary_gate_multiple_probability",
        "boundary_gate_entropy",
    ]
    for index, name in enumerate(boundary_columns[:3]):
        training[name] = boundary_train[:, index]
        holdout[name] = boundary_holdout[:, index]
    training[boundary_columns[3]] = -np.sum(
        boundary_train * np.log(np.maximum(boundary_train, 1e-9)), axis=1
    )
    holdout[boundary_columns[3]] = -np.sum(
        boundary_holdout * np.log(np.maximum(boundary_holdout, 1e-9)), axis=1
    )

    venue_columns = [column for column in venue if column.startswith("venue_track_")]
    groups = {
        "boundary_gate": boundary_columns,
        "venue_track": venue_columns,
        "boundary_venue_combined": boundary_columns + venue_columns,
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
    band_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    latency: dict[str, float] = {}
    for name, additions in groups.items():
        columns = V3_FEATURES + additions
        categories = CATEGORICAL_FEATURES + (
            ["venue_track_type"] if "venue_track_type" in additions else []
        )
        model = run_model()
        model.fit(
            _feature_frame(training, columns),
            np.minimum(training["runs_in_over"], MAX_RUN_CLASS),
            categorical_feature=categories,
        )
        probability = model.predict_proba(_feature_frame(holdout, columns))
        low, high, mass = _best_bands(probability, 2)
        hit = (actual >= low) & (actual <= high)
        candidate_wins = int((~baseline & hit).sum())
        baseline_wins = int((baseline & ~hit).sum())
        paired_p = float(
            binomtest(
                min(candidate_wins, baseline_wins),
                candidate_wins + baseline_wins,
                0.5,
            ).pvalue
        )
        sample = _feature_frame(holdout.tail(1), columns)
        model.predict_proba(sample)
        started = time.perf_counter()
        for _ in range(300):
            model.predict_proba(sample)
        latency[name] = (time.perf_counter() - started) * 1000 / 300
        reports[name] = {
            "features_added": additions,
            "two_run_hit_rate": float(hit.mean()),
            "improvement_over_v32_percentage_points": float(
                (hit.mean() - V32_HIT_RATE) * 100
            ),
            "paired_exact_p_value": paired_p,
            "candidate_only_wins": candidate_wins,
            "v32_only_wins": baseline_wins,
            "mean_band_probability": float(mass.mean()),
            "phase": _phase_report(holdout, actual, low, high),
            "mean_main_prediction_ms": latency[name],
        }
        band_outputs[name] = (low, high)
        joblib.dump(model, output_dir / f"{name}_model.pkl")
        joblib.dump(columns, output_dir / f"{name}_feature_cols.pkl")
        joblib.dump(categories, output_dir / f"{name}_cat_cols.pkl")

    powerplay = holdout["phase"].astype(str).to_numpy() == "powerplay"
    router_low = np.where(
        powerplay,
        band_outputs["boundary_venue_combined"][0],
        band_outputs["boundary_gate"][0],
    )
    router_high = np.where(
        powerplay,
        band_outputs["boundary_venue_combined"][1],
        band_outputs["boundary_gate"][1],
    )
    router_hit = (actual >= router_low) & (actual <= router_high)
    router_wins = int((~baseline & router_hit).sum())
    baseline_wins = int((baseline & ~router_hit).sum())
    reports["phase_selective_router"] = {
        "routing": {
            "powerplay": "boundary_venue_combined",
            "middle": "boundary_gate",
            "death": "boundary_gate",
        },
        "two_run_hit_rate": float(router_hit.mean()),
        "improvement_over_v32_percentage_points": float(
            (router_hit.mean() - V32_HIT_RATE) * 100
        ),
        "paired_exact_p_value": float(
            binomtest(
                min(router_wins, baseline_wins), router_wins + baseline_wins, 0.5
            ).pvalue
        ),
        "candidate_only_wins": router_wins,
        "v32_only_wins": baseline_wins,
        "phase": _phase_report(holdout, actual, router_low, router_high),
        "mean_main_prediction_ms": max(latency.values()),
    }

    winner_name = max(reports, key=lambda key: reports[key]["two_run_hit_rate"])
    winner = reports[winner_name]
    baseline_phase = {
        "powerplay": 0.2774744366127271,
        "middle": 0.34074231628203766,
        "death": 0.31140663342040686,
    }
    gates = {
        "accuracy_at_least_32_38pct": winner["two_run_hit_rate"] >= 0.3238,
        "improvement_at_least_0_5pp": winner["improvement_over_v32_percentage_points"]
        >= 0.5,
        "paired_p_below_0_01": winner["paired_exact_p_value"] < 0.01,
        "no_phase_regresses_over_2pct": all(
            winner["phase"][phase]["hit_rate"] >= rate - 0.02
            for phase, rate in baseline_phase.items()
        ),
        "latency_below_50ms": boundary_latency_ms
        + winner["mean_main_prediction_ms"]
        < 50.0,
        "complete_match_sequence_failures_zero": _sequence_failures(holdout) == 0,
    }
    boundary_actual = np.minimum(holdout["boundary_count"].to_numpy(), 2)
    report = {
        "candidate_version": "v3.7_boundary_venue_track",
        "model_scope": "candidate_only",
        "production_models_changed": False,
        "verified_rows": len(data),
        "holdout_rows": len(holdout),
        "holdout_matches": int(holdout["source_file"].nunique()),
        "boundary_gate_accuracy": float(
            (np.argmax(boundary_holdout, axis=1) == boundary_actual).mean()
        ),
        "boundary_gate_single_prediction_ms": boundary_latency_ms,
        "winner_end_to_end_prediction_ms": boundary_latency_ms
        + winner["mean_main_prediction_ms"],
        "complete_match_sequence_failures": _sequence_failures(holdout),
        "ablations": reports,
        "winner": winner_name,
        "winner_metrics": winner,
        "promotion_gates": gates,
        "integration_recommended": all(gates.values()),
    }
    joblib.dump(boundary_model, output_dir / "boundary_count_submodel.pkl")
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            train(root, root / "models/candidates/v3.7_boundary_venue_track"),
            indent=2,
        )
    )
