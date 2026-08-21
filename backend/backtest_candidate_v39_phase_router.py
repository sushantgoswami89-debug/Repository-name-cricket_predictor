"""Exact paired backtest for the strongest phase-specific candidate models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from train_candidate_v33_response import _model as run_model
from train_candidate_v37_boundary_venue import _boundary_probabilities
from train_sharp_range_candidate import (
    MAX_RUN_CLASS,
    _best_bands,
    _phase_report,
    _sequence_failures,
)
from train_validate_candidate_v3 import V3_FEATURES, _features

KEYS = ["source_file", "match_date", "innings", "over"]
V32_HIT_RATE = 0.31376664261821996


def _feature_frame(
    frame: pd.DataFrame, columns: list[str], categories: list[str]
) -> pd.DataFrame:
    result = _features(frame, columns)
    for column in categories:
        if column in result:
            result[column] = result[column].astype("category")
    return result


def _load_probability(frame: pd.DataFrame, directory: Path, name: str) -> np.ndarray:
    model = joblib.load(directory / f"{name}_model.pkl")
    columns = joblib.load(directory / f"{name}_feature_cols.pkl")
    categories = joblib.load(directory / f"{name}_cat_cols.pkl")
    features = _feature_frame(frame, columns, categories)
    for column, levels in zip(
        categories, model.booster_.pandas_categorical, strict=True
    ):
        features[column] = pd.Categorical(features[column], categories=levels)
    return model.predict_proba(features)


def _match_bootstrap(
    frame: pd.DataFrame,
    candidate_hit: np.ndarray,
    baseline_hit: np.ndarray,
    iterations: int = 10_000,
) -> dict[str, float]:
    values = pd.DataFrame(
        {
            "source_file": frame["source_file"].to_numpy(),
            "delta": candidate_hit.astype(float) - baseline_hit.astype(float),
        }
    )
    match_delta = values.groupby("source_file")["delta"].agg(["sum", "count"])
    sums = match_delta["sum"].to_numpy()
    counts = match_delta["count"].to_numpy()
    random = np.random.default_rng(42)
    bootstrap = np.empty(iterations)
    for index in range(iterations):
        selected = random.integers(0, len(sums), len(sums))
        bootstrap[index] = sums[selected].sum() / counts[selected].sum()
    return {
        "delta_percentage_points": float(
            (candidate_hit.mean() - baseline_hit.mean()) * 100
        ),
        "ci_95_low_percentage_points": float(np.quantile(bootstrap, 0.025) * 100),
        "ci_95_high_percentage_points": float(np.quantile(bootstrap, 0.975) * 100),
        "probability_delta_not_positive": float(np.mean(bootstrap <= 0)),
    }


def _calibration_report(probability: np.ndarray, hit: np.ndarray) -> dict[str, Any]:
    order = np.argsort(probability)
    bins: list[dict[str, float | int]] = []
    weighted_error = 0.0
    for positions in np.array_split(order, 10):
        predicted = float(probability[positions].mean())
        actual = float(hit[positions].mean())
        weighted_error += len(positions) * abs(predicted - actual)
        bins.append(
            {
                "rows": len(positions),
                "mean_predicted_probability": predicted,
                "actual_hit_rate": actual,
            }
        )
    coverage = {}
    for threshold in (0.30, 0.325, 0.35, 0.375, 0.40):
        selected = probability >= threshold
        coverage[str(threshold)] = {
            "coverage": float(selected.mean()),
            "accuracy": float(hit[selected].mean()) if selected.any() else None,
        }
    return {
        "brier_score": float(np.mean((probability - hit.astype(float)) ** 2)),
        "expected_calibration_error": weighted_error / len(hit),
        "decile_bins": bins,
        "diagnostic_thresholds_not_approved_for_live_use": coverage,
    }


def backtest(project_root: Path, output_dir: Path) -> dict[str, Any]:
    base = pd.read_csv(project_root / "data/candidates/v3/verified_training_overs.csv")
    venue = pd.read_csv(project_root / "data/candidates/v3.7/venue_track_features.csv")
    player = pd.read_csv(
        project_root / "data/candidates/v3.8/innings_phase_player_features.csv"
    )
    components = pd.read_csv(project_root / "data/candidates/v3.4/over_components.csv")
    for frame in (base, venue, player, components):
        frame["match_date"] = frame["match_date"].astype(str)
    components["boundary_count"] = components["fours"] + components["sixes"]
    data = (
        base.merge(venue, on=KEYS, validate="one_to_one")
        .merge(player, on=KEYS, validate="one_to_one")
        .merge(
            components[KEYS + ["boundary_count"]],
            on=KEYS,
            validate="one_to_one",
        )
    )
    data["match_date"] = pd.to_datetime(data["match_date"])
    training = data[data["match_date"].dt.year <= 2023].reset_index(drop=True)
    holdout = data[data["match_date"].dt.year >= 2025].reset_index(drop=True)

    boundary_directory = project_root / "models/candidates/v3.7_boundary_venue_track"
    boundary_train, boundary_probability, _ = _boundary_probabilities(training, holdout)
    for index, name in enumerate(
        (
            "boundary_gate_none_probability",
            "boundary_gate_one_probability",
            "boundary_gate_multiple_probability",
        )
    ):
        training[name] = boundary_train[:, index]
        holdout[name] = boundary_probability[:, index]
    training["boundary_gate_entropy"] = -np.sum(
        boundary_train * np.log(np.maximum(boundary_train, 1e-9)), axis=1
    )
    holdout["boundary_gate_entropy"] = -np.sum(
        boundary_probability * np.log(np.maximum(boundary_probability, 1e-9)), axis=1
    )

    boundary_run_columns = [
        *V3_FEATURES,
        "boundary_gate_none_probability",
        "boundary_gate_one_probability",
        "boundary_gate_multiple_probability",
        "boundary_gate_entropy",
    ]
    death_model = run_model()
    death_model.fit(
        _feature_frame(training, boundary_run_columns, ["phase"]),
        np.minimum(training["runs_in_over"], MAX_RUN_CLASS),
        categorical_feature=["phase"],
    )
    v37_boundary = death_model.predict_proba(
        _feature_frame(holdout, boundary_run_columns, ["phase"])
    )
    v37_venue = _load_probability(
        holdout, boundary_directory, "boundary_venue_combined"
    )
    v38_directory = project_root / "models/candidates/v3.8_innings_phase_bowler_type"
    v38_middle = _load_probability(holdout, v38_directory, "boundary_player_profiles")
    phase = holdout["phase"].astype(str).to_numpy()
    probability = np.where(
        (phase == "powerplay")[:, None],
        v37_venue,
        np.where((phase == "middle")[:, None], v38_middle, v37_boundary),
    )
    low, high, mass = _best_bands(probability, 2)
    actual = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
    hit = (actual >= low) & (actual <= high)

    baseline_frame = pd.read_csv(
        project_root / "models/candidates/v3.2_sharp_range/holdout_sharp_ranges.csv"
    )
    baseline_frame["match_date"] = baseline_frame["match_date"].astype(str)
    holdout_keys = holdout[KEYS].copy()
    holdout_keys["match_date"] = holdout_keys["match_date"].astype(str)
    baseline = (
        holdout_keys.merge(
            baseline_frame[KEYS + ["sharp_2_hit"]],
            on=KEYS,
            validate="one_to_one",
        )["sharp_2_hit"]
        .astype(bool)
        .to_numpy()
    )
    candidate_wins = int((hit & ~baseline).sum())
    baseline_wins = int((~hit & baseline).sum())
    hit_rate = float(hit.mean())
    phase_metrics = _phase_report(holdout, actual, low, high)
    paired_p = float(
        binomtest(
            min(candidate_wins, baseline_wins),
            candidate_wins + baseline_wins,
            0.5,
        ).pvalue
    )
    gates = {
        "accuracy_at_least_32_38pct": hit_rate >= 0.3238,
        "improvement_over_v32_at_least_0_5pp": (hit_rate - V32_HIT_RATE) >= 0.005,
        "paired_p_below_0_01": paired_p < 0.01,
        "match_bootstrap_ci_above_zero": _match_bootstrap(holdout, hit, baseline)[
            "ci_95_low_percentage_points"
        ]
        > 0,
        "complete_match_sequence_failures_zero": _sequence_failures(holdout) == 0,
    }
    report = {
        "candidate_version": "v3.9_exact_phase_router",
        "model_scope": "candidate_only",
        "production_models_changed": False,
        "routing": {
            "powerplay": "v3.7 boundary + venue/track",
            "middle": "v3.8 boundary + innings/phase/bowler-type profiles",
            "death": "v3.7 boundary gate",
        },
        "holdout_rows": len(holdout),
        "holdout_matches": int(holdout["source_file"].nunique()),
        "two_run_hit_rate": hit_rate,
        "improvement_over_v32_percentage_points": (hit_rate - V32_HIT_RATE) * 100,
        "mean_band_probability": float(mass.mean()),
        "phase": phase_metrics,
        "paired_test": {
            "candidate_only_wins": candidate_wins,
            "v32_only_wins": baseline_wins,
            "exact_p_value": paired_p,
        },
        "match_level_bootstrap": _match_bootstrap(holdout, hit, baseline),
        "range_probability_calibration": _calibration_report(mass, hit),
        "complete_match_sequence_failures": _sequence_failures(holdout),
        "promotion_gates": gates,
        "integration_recommended": all(gates.values()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            backtest(root, root / "models/candidates/v3.9_exact_phase_router"),
            indent=2,
        )
    )
