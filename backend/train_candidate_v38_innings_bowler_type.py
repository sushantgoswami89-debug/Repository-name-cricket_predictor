"""Train innings-role, phase, track, and bowler-type player ablations."""

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
V37_HIT_RATE = 0.3194890798189808


def _feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = _features(frame, columns)
    for column in (
        "venue_track_type",
        "player_innings_bowler_type",
        "player_innings_bowler_arm",
    ):
        if column in result:
            result[column] = result[column].astype("category")
    return result


def _paired(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, float | int]:
    candidate_wins = int((candidate & ~baseline).sum())
    baseline_wins = int((~candidate & baseline).sum())
    return {
        "candidate_only_wins": candidate_wins,
        "baseline_only_wins": baseline_wins,
        "paired_exact_p_value": float(
            binomtest(
                min(candidate_wins, baseline_wins),
                candidate_wins + baseline_wins,
                0.5,
            ).pvalue
        ),
    }


def train(project_root: Path, output_dir: Path) -> dict[str, Any]:
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
    if len(data) != len(base):
        raise ValueError("v3.8 features do not reconcile with verified data.")
    data["match_date"] = pd.to_datetime(data["match_date"])
    training = data[data["match_date"].dt.year <= 2023].reset_index(drop=True)
    holdout = data[data["match_date"].dt.year >= 2025].reset_index(drop=True)
    boundary_train, boundary_holdout, _ = _boundary_probabilities(training, holdout)
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

    common = [
        "player_innings_is_chase",
        "player_innings_bowler_type",
        "player_innings_bowler_arm",
    ]
    batter = common + [
        column for column in player if column.startswith("player_innings_batter_")
    ]
    bowler = common + [
        column
        for column in player
        if column.startswith("player_innings_bowler_") and column not in common
    ]
    both = list(dict.fromkeys(batter + bowler))
    venue_columns = [column for column in venue if column.startswith("venue_track_")]
    groups = {
        "batter_innings_phase_type": batter,
        "bowler_innings_phase_type": bowler,
        "player_profiles_combined": both,
        "boundary_player_profiles": boundary_columns + both,
        "boundary_venue_player_profiles": boundary_columns + venue_columns + both,
    }
    actual = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
    v32_hit = (
        pd.read_csv(
            project_root / "models/candidates/v3.2_sharp_range/holdout_sharp_ranges.csv"
        )["sharp_2_hit"]
        .astype(bool)
        .to_numpy()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    bands: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    latencies: dict[str, float] = {}
    for name, additions in groups.items():
        columns = V3_FEATURES + additions
        categories = CATEGORICAL_FEATURES + [
            column
            for column in (
                "venue_track_type",
                "player_innings_bowler_type",
                "player_innings_bowler_arm",
            )
            if column in additions
        ]
        model = _model()
        model.fit(
            _feature_frame(training, columns),
            np.minimum(training["runs_in_over"], MAX_RUN_CLASS),
            categorical_feature=categories,
        )
        probability = model.predict_proba(_feature_frame(holdout, columns))
        low, high, mass = _best_bands(probability, 2)
        hit = (actual >= low) & (actual <= high)
        sample = _feature_frame(holdout.tail(1), columns)
        model.predict_proba(sample)
        started = time.perf_counter()
        for _ in range(300):
            model.predict_proba(sample)
        latencies[name] = (time.perf_counter() - started) * 1000 / 300
        reports[name] = {
            "features_added": additions,
            "two_run_hit_rate": float(hit.mean()),
            "improvement_over_v32_percentage_points": float(
                (hit.mean() - V32_HIT_RATE) * 100
            ),
            "improvement_over_v37_percentage_points": float(
                (hit.mean() - V37_HIT_RATE) * 100
            ),
            "paired_vs_v32": _paired(hit, v32_hit),
            "mean_band_probability": float(mass.mean()),
            "phase": _phase_report(holdout, actual, low, high),
            "mean_main_prediction_ms": latencies[name],
        }
        bands[name] = (low, high)
        joblib.dump(model, output_dir / f"{name}_model.pkl")
        joblib.dump(columns, output_dir / f"{name}_feature_cols.pkl")
        joblib.dump(categories, output_dir / f"{name}_cat_cols.pkl")

    powerplay = holdout["phase"].astype(str).to_numpy() == "powerplay"
    full = bands["boundary_venue_player_profiles"]
    lighter = bands["boundary_player_profiles"]
    router_low = np.where(powerplay, full[0], lighter[0])
    router_high = np.where(powerplay, full[1], lighter[1])
    router_hit = (actual >= router_low) & (actual <= router_high)
    reports["phase_selective_player_router"] = {
        "routing": {
            "powerplay": "boundary_venue_player_profiles",
            "middle": "boundary_player_profiles",
            "death": "boundary_player_profiles",
        },
        "two_run_hit_rate": float(router_hit.mean()),
        "improvement_over_v32_percentage_points": float(
            (router_hit.mean() - V32_HIT_RATE) * 100
        ),
        "improvement_over_v37_percentage_points": float(
            (router_hit.mean() - V37_HIT_RATE) * 100
        ),
        "paired_vs_v32": _paired(router_hit, v32_hit),
        "phase": _phase_report(holdout, actual, router_low, router_high),
        "mean_main_prediction_ms": max(latencies.values()),
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
        "beats_v37": winner["two_run_hit_rate"] > V37_HIT_RATE,
        "paired_vs_v32_p_below_0_01": winner["paired_vs_v32"]["paired_exact_p_value"]
        < 0.01,
        "no_phase_regresses_over_2pct": all(
            winner["phase"][phase]["hit_rate"] >= rate - 0.02
            for phase, rate in baseline_phase.items()
        ),
        "latency_below_50ms": winner["mean_main_prediction_ms"] < 50.0,
        "complete_match_sequence_failures_zero": _sequence_failures(holdout) == 0,
    }
    report = {
        "candidate_version": "v3.8_innings_phase_bowler_type",
        "model_scope": "candidate_only",
        "production_models_changed": False,
        "verified_rows": len(data),
        "holdout_rows": len(holdout),
        "holdout_matches": int(holdout["source_file"].nunique()),
        "bowler_type_coverage": holdout["player_innings_bowler_type"]
        .value_counts()
        .to_dict(),
        "bowler_arm_coverage": holdout["player_innings_bowler_arm"]
        .value_counts()
        .to_dict(),
        "ablations": reports,
        "winner": winner_name,
        "winner_metrics": winner,
        "promotion_gates": gates,
        "integration_recommended": all(gates.values()),
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            train(root, root / "models/candidates/v3.8_innings_phase_bowler_type"),
            indent=2,
        )
    )
