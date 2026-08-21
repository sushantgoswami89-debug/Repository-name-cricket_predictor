"""Train an IPL engine using only the 2023+ Impact Player era."""

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
from train_sharp_range_candidate import MAX_RUN_CLASS, _best_bands, _phase_report
from train_validate_candidate_v3 import CATEGORICAL_FEATURES, V3_FEATURES, _features

KEYS = ["source_file", "match_date", "innings", "over"]


def _feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = _features(frame, columns)
    for column in (
        "venue_track_type",
        "ipl_impact_batting_role",
        "ipl_impact_bowling_role",
    ):
        if column in result:
            result[column] = result[column].astype("category")
    return result


def train(project_root: Path, output_dir: Path) -> dict[str, Any]:
    base = pd.read_csv(project_root / "data/candidates/v3/verified_training_overs.csv")
    impact = pd.read_csv(
        project_root / "data/candidates/ipl_v1/impact_player_features.csv"
    )
    venue = pd.read_csv(project_root / "data/candidates/v3.7/venue_track_features.csv")
    for frame in (base, impact, venue):
        frame["match_date"] = frame["match_date"].astype(str)
    data = base.merge(impact, on=KEYS, how="inner", validate="one_to_one").merge(
        venue,
        on=KEYS,
        validate="one_to_one",
    )
    if len(data) != len(impact):
        raise ValueError("IPL impact features do not reconcile with IPL overs.")
    data["match_date"] = pd.to_datetime(data["match_date"])
    training = data[data["match_date"].dt.year.between(2023, 2024)].reset_index(
        drop=True
    )
    holdout = data[data["match_date"].dt.year >= 2025].reset_index(drop=True)
    impact_columns = [column for column in impact if column.startswith("ipl_impact_")]
    venue_columns = [column for column in venue if column.startswith("venue_track_")]
    groups = {
        "ipl_baseline": [],
        "impact_player": impact_columns,
        "venue_impact": venue_columns + impact_columns,
    }
    actual = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    hits: dict[str, np.ndarray] = {}
    for name, additions in groups.items():
        columns = V3_FEATURES + additions
        categories = CATEGORICAL_FEATURES + [
            column
            for column in (
                "venue_track_type",
                "ipl_impact_batting_role",
                "ipl_impact_bowling_role",
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
        reports[name] = {
            "features_added": additions,
            "two_run_hit_rate": float(hit.mean()),
            "mean_band_probability": float(mass.mean()),
            "phase": _phase_report(holdout, actual, low, high),
            "mean_main_prediction_ms": (time.perf_counter() - started) * 1000 / 300,
        }
        hits[name] = hit
        joblib.dump(model, output_dir / f"{name}_model.pkl")
        joblib.dump(columns, output_dir / f"{name}_feature_cols.pkl")
        joblib.dump(categories, output_dir / f"{name}_cat_cols.pkl")

    baseline = hits["ipl_baseline"]
    for name, values in reports.items():
        hit = hits[name]
        candidate_wins = int((hit & ~baseline).sum())
        baseline_wins = int((~hit & baseline).sum())
        values["improvement_over_ipl_baseline_percentage_points"] = float(
            (hit.mean() - baseline.mean()) * 100
        )
        values["paired_vs_ipl_baseline_p_value"] = (
            float(
                binomtest(
                    min(candidate_wins, baseline_wins),
                    candidate_wins + baseline_wins,
                    0.5,
                ).pvalue
            )
            if candidate_wins + baseline_wins
            else 1.0
        )
    winner_name = max(reports, key=lambda key: reports[key]["two_run_hit_rate"])
    winner = reports[winner_name]
    gates = {
        "accuracy_at_least_32_38pct": winner["two_run_hit_rate"] >= 0.3238,
        "improvement_at_least_0_5pp": winner[
            "improvement_over_ipl_baseline_percentage_points"
        ]
        >= 0.5,
        "paired_p_below_0_01": winner["paired_vs_ipl_baseline_p_value"] < 0.01,
        "latency_below_50ms": winner["mean_main_prediction_ms"] < 50.0,
    }
    report = {
        "candidate_version": "ipl_engine_v2_impact_era_only",
        "model_scope": "candidate_only_ipl",
        "production_models_changed": False,
        "verified_ipl_rows": len(data),
        "pre_impact_rows_excluded": int((data["match_date"].dt.year < 2023).sum()),
        "training_rows_2023_2024": len(training),
        "holdout_rows_2025_plus": len(holdout),
        "holdout_matches": int(holdout["source_file"].nunique()),
        "impact_era_holdout_rows": int(holdout["ipl_impact_rule_era"].sum()),
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
            train(root, root / "models/candidates/ipl_engine_v2_impact_era"),
            indent=2,
        )
    )
