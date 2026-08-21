"""Train current-match pitch adaptation ablations."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
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


def _feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = _features(frame, columns)
    if "venue_track_type" in result:
        result["venue_track_type"] = result["venue_track_type"].astype("category")
    return result


def train(project_root: Path, output_dir: Path) -> dict[str, Any]:
    base = pd.read_csv(project_root / "data/candidates/v3/verified_training_overs.csv")
    venue = pd.read_csv(project_root / "data/candidates/v3.7/venue_track_features.csv")
    pitch = pd.read_csv(
        project_root / "data/candidates/v3.10/current_match_pitch_features.csv"
    )
    components = pd.read_csv(project_root / "data/candidates/v3.4/over_components.csv")
    for frame in (base, venue, pitch, components):
        frame["match_date"] = frame["match_date"].astype(str)
    components["boundary_count"] = components["fours"] + components["sixes"]
    data = (
        base.merge(venue, on=KEYS, validate="one_to_one")
        .merge(pitch, on=KEYS, validate="one_to_one")
        .merge(
            components[KEYS + ["boundary_count"]],
            on=KEYS,
            validate="one_to_one",
        )
    )
    for name in ("runs", "boundaries", "dots", "wickets"):
        venue_rate = data[f"venue_track_venue_{name}_rate"]
        data[f"match_pitch_delta_match_{name}"] = (
            data[f"match_pitch_match_{name}_rate"] - venue_rate
        )
        data[f"match_pitch_delta_first_innings_{name}"] = (
            data[f"match_pitch_first_innings_{name}_rate"] - venue_rate
        )
    data["match_date"] = pd.to_datetime(data["match_date"])
    training = data[data["match_date"].dt.year <= 2023].reset_index(drop=True)
    holdout = data[data["match_date"].dt.year >= 2025].reset_index(drop=True)
    boundary_train, boundary_holdout, boundary_model = _boundary_probabilities(
        training, holdout
    )
    boundary_columns = [
        "boundary_gate_none_probability",
        "boundary_gate_one_probability",
        "boundary_gate_multiple_probability",
        "boundary_gate_entropy",
    ]
    for index, name in enumerate(boundary_columns[:3]):
        training[name] = boundary_train[:, index]
        holdout[name] = boundary_holdout[:, index]
    for frame, probability in ((training, boundary_train), (holdout, boundary_holdout)):
        frame[boundary_columns[3]] = -np.sum(
            probability * np.log(np.maximum(probability, 1e-9)), axis=1
        )
    pitch_columns = [column for column in data if column.startswith("match_pitch_")]
    venue_columns = [column for column in venue if column.startswith("venue_track_")]
    groups = {
        "current_match_pitch": pitch_columns,
        "boundary_match_pitch": boundary_columns + pitch_columns,
        "boundary_venue_match_pitch": boundary_columns + venue_columns + pitch_columns,
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
        categories = CATEGORICAL_FEATURES + (
            ["venue_track_type"] if "venue_track_type" in additions else []
        )
        model = _model()
        model.fit(
            _feature_frame(training, columns),
            np.minimum(training["runs_in_over"], MAX_RUN_CLASS),
            categorical_feature=categories,
        )
        probability = model.predict_proba(_feature_frame(holdout, columns))
        low, high, mass = _best_bands(probability, 2)
        hit = (actual >= low) & (actual <= high)
        candidate_wins = int((hit & ~baseline).sum())
        baseline_wins = int((~hit & baseline).sum())
        sample = _feature_frame(holdout.tail(1), columns)
        model.predict_proba(sample)
        started = time.perf_counter()
        for _ in range(300):
            model.predict_proba(sample)
        latency = (time.perf_counter() - started) * 1000 / 300
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
            "mean_main_prediction_ms": latency,
        }
        joblib.dump(model, output_dir / f"{name}_model.pkl")
        joblib.dump(columns, output_dir / f"{name}_feature_cols.pkl")
        joblib.dump(categories, output_dir / f"{name}_cat_cols.pkl")
    winner_name = max(reports, key=lambda key: reports[key]["two_run_hit_rate"])
    winner = reports[winner_name]
    gates = {
        "accuracy_at_least_32_38pct": winner["two_run_hit_rate"] >= 0.3238,
        "beats_v39": winner["two_run_hit_rate"] > V39_HIT_RATE,
        "paired_vs_v32_p_below_0_01": winner["paired_vs_v32_p_value"] < 0.01,
        "latency_below_50ms": winner["mean_main_prediction_ms"] < 50.0,
        "complete_match_sequence_failures_zero": _sequence_failures(holdout) == 0,
    }
    report = {
        "candidate_version": "v3.10_current_match_pitch",
        "model_scope": "candidate_only",
        "production_models_changed": False,
        "verified_rows": len(data),
        "holdout_rows": len(holdout),
        "holdout_matches": int(holdout["source_file"].nunique()),
        "ablations": reports,
        "winner": winner_name,
        "winner_metrics": winner,
        "promotion_gates": gates,
        "integration_recommended": all(gates.values()),
    }
    joblib.dump(boundary_model, output_dir / "boundary_submodel.pkl")
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            train(root, root / "models/candidates/v3.10_current_match_pitch"),
            indent=2,
        )
    )
