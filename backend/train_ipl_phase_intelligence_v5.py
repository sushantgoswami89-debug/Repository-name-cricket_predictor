"""Train an IPL-only phase router with player, venue, and live-state context."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from train_candidate_v33_response import _model as run_model
from train_sharp_range_candidate import MAX_RUN_CLASS, _best_bands, _phase_report
from train_validate_candidate_v3 import V3_FEATURES

KEYS = ["source_file", "match_date", "innings", "over"]
CATEGORICAL = [
    "phase",
    "venue_track_type",
    "player_innings_bowler_type",
    "player_innings_bowler_arm",
    "batter_entry_phase",
    "home_context",
    "previous_over_event",
    "home_phase_transition",
]

HOME_MARKERS = {
    "Chennai Super Kings": ("chennai",),
    "Mumbai Indians": ("wankhede", "mumbai"),
    "Royal Challengers Bangalore": ("chinnaswamy", "bangalore", "bengaluru"),
    "Royal Challengers Bengaluru": ("chinnaswamy", "bangalore", "bengaluru"),
    "Kolkata Knight Riders": ("eden gardens", "kolkata"),
    "Delhi Capitals": ("arun jaitley", "delhi"),
    "Sunrisers Hyderabad": ("hyderabad",),
    "Rajasthan Royals": ("jaipur",),
    "Punjab Kings": ("mohali", "mullanpur", "dharamsala"),
    "Gujarat Titans": ("ahmedabad",),
    "Lucknow Super Giants": ("lucknow",),
}


def _x(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame[columns].copy()
    for column in CATEGORICAL:
        if column in result:
            result[column] = result[column].fillna("unknown").astype("category")
    return result


def _wicket_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=75,
        learning_rate=0.05,
        num_leaves=19,
        max_depth=5,
        min_child_samples=80,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )


def _load(project_root: Path) -> tuple[pd.DataFrame, list[str]]:
    paths = [
        "data/candidates/v3/verified_training_overs.csv",
        "data/candidates/v3.7/venue_track_features.csv",
        "data/candidates/v3.8/innings_phase_player_features.csv",
        "data/candidates/v3.5/batter_entry_phase_features.csv",
        "data/candidates/v3.6/state_strike_share_features.csv",
        "data/candidates/ipl_v3/adapter_features.csv",
    ]
    frames = [pd.read_csv(project_root / path) for path in paths]
    components = pd.read_csv(
        project_root / "data/candidates/v3.4/over_components.csv"
    )
    frames.append(
        components[
            KEYS + ["fours", "sixes", "wickets", "total_runs", "dot_balls"]
        ]
    )
    for frame in frames:
        frame["match_date"] = frame["match_date"].astype(str)
    data = frames[0]
    for extra in frames[1:]:
        data = data.merge(extra, on=KEYS, how="inner", validate="one_to_one")
    ipl_files = {
        path.name for path in (project_root / "data/raw/cricsheet/ipl").glob("*.json")
    }
    data = data[data["source_file"].isin(ipl_files)].copy()
    contexts: list[dict[str, Any]] = []
    for source_file in sorted(ipl_files):
        raw = json.loads(
            (project_root / "data/raw/cricsheet/ipl" / source_file).read_text(
                encoding="utf-8"
            )
        )
        info = raw["info"]
        venue = f"{info.get('venue', '')} {info.get('city', '')}".lower()
        event = info.get("event", {})
        stage = f"{event.get('stage', '')} {event.get('match_number', '')}".lower()
        is_neutral = any(
            marker in stage
            for marker in ("final", "qualifier", "eliminator", "playoff")
        )
        for innings_number, innings_data in enumerate(
            raw.get("innings", []), start=1
        ):
            team = str(innings_data.get("team", ""))
            markers = HOME_MARKERS.get(team, ())
            context = (
                "neutral"
                if is_neutral
                else "home"
                if any(marker in venue for marker in markers)
                else "away"
            )
            contexts.append(
                {
                    "source_file": source_file,
                    "innings": innings_number,
                    "home_context": context,
                }
            )
    data = data.merge(
        pd.DataFrame(contexts),
        on=["source_file", "innings"],
        how="left",
        validate="many_to_one",
    )
    data["match_date"] = pd.to_datetime(data["match_date"])
    data = data.sort_values(["source_file", "innings", "over"])
    innings = data.groupby(["source_file", "innings"], sort=False)
    for source in ("fours", "sixes", "wickets", "total_runs", "dot_balls"):
        data[f"previous_over_{source}"] = innings[source].shift(1).fillna(-1)
    data["previous_over_boundaries"] = (
        data["previous_over_fours"] + data["previous_over_sixes"]
    )
    data["previous_over_transition_score"] = np.where(
        data["previous_over_total_runs"] < 0,
        -1,
        data["previous_over_total_runs"]
        + 1.5 * data["previous_over_boundaries"]
        - 1.0 * data["previous_over_wickets"],
    )
    boundaries = data["previous_over_boundaries"]
    data["previous_over_event"] = np.select(
        [
            data["previous_over_total_runs"] < 0,
            (boundaries == 0) & (data["previous_over_wickets"] == 0),
            (boundaries == 0) & (data["previous_over_wickets"] > 0),
            (boundaries == 1) & (data["previous_over_wickets"] == 0),
            (boundaries == 1) & (data["previous_over_wickets"] > 0),
            boundaries >= 2,
        ],
        [
            "innings_start",
            "quiet",
            "wicket",
            "one_boundary",
            "boundary_wicket",
            "multiple_boundaries",
        ],
        default="other",
    )
    data["home_phase_transition"] = (
        data["home_context"].astype(str)
        + "|"
        + data["phase"].astype(str)
        + "|"
        + data["previous_over_event"].astype(str)
    )
    data["home_boundary_signal"] = (
        (data["home_context"] == "home").astype(int) * boundaries
    )
    data["home_wicket_signal"] = (
        (data["home_context"] == "home").astype(int)
        * data["previous_over_wickets"]
    )
    prefixes = (
        "venue_track_",
        "player_innings_",
        "entry_phase_",
        "settlement_",
        "strike_share_",
        "bowler_match_",
        "volatility_",
        "ipl_adapter_",
        "previous_over_",
        "home_",
    )
    excluded = {"ipl_adapter_batting_team_win", "ipl_adapter_result_known"}
    columns = V3_FEATURES + [
        column
        for column in data.columns
        if column.startswith(prefixes) and column not in excluded
    ]
    columns = list(dict.fromkeys(columns))
    return data, columns


def train(project_root: Path, output_dir: Path) -> dict[str, Any]:
    data, columns = _load(project_root)
    training = data[data["match_date"].dt.year <= 2024].reset_index(drop=True)
    holdout = data[data["match_date"].dt.year >= 2025].reset_index(drop=True)
    actual_runs = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
    run_low = np.zeros(len(holdout), dtype=int)
    run_high = np.zeros(len(holdout), dtype=int)
    run_mass = np.zeros(len(holdout))
    wicket_probability = np.zeros(len(holdout))
    models: dict[str, Any] = {}
    phase_rows: dict[str, dict[str, int]] = {}
    for phase in ("powerplay", "middle", "death"):
        train_mask = training["phase"] == phase
        test_mask = holdout["phase"] == phase
        run = run_model()
        wicket = _wicket_model()
        categories = [column for column in CATEGORICAL if column in columns]
        run.fit(
            _x(training.loc[train_mask], columns),
            np.minimum(training.loc[train_mask, "runs_in_over"], MAX_RUN_CLASS),
            categorical_feature=categories,
        )
        wicket.fit(
            _x(training.loc[train_mask], columns),
            training.loc[train_mask, "wicket_in_over"],
            categorical_feature=categories,
        )
        probability = run.predict_proba(_x(holdout.loc[test_mask], columns))
        low, high, mass = _best_bands(probability, 2)
        run_low[test_mask] = low
        run_high[test_mask] = high
        run_mass[test_mask] = mass
        wicket_probability[test_mask] = wicket.predict_proba(
            _x(holdout.loc[test_mask], columns)
        )[:, 1]
        models[f"{phase}_runs"] = run
        models[f"{phase}_wicket"] = wicket
        phase_rows[phase] = {
            "training": int(train_mask.sum()), "holdout": int(test_mask.sum())
        }
    hit = (actual_runs >= run_low) & (actual_runs <= run_high)
    sample = holdout.tail(1)
    phase = str(sample.iloc[0]["phase"])
    started = time.perf_counter()
    for _ in range(300):
        models[f"{phase}_runs"].predict_proba(_x(sample, columns))
        models[f"{phase}_wicket"].predict_proba(_x(sample, columns))
    latency = (time.perf_counter() - started) * 1000 / 300
    accuracy = float(hit.mean())
    wicket_brier = float(
        brier_score_loss(holdout["wicket_in_over"], wicket_probability)
    )
    gates = {
        "run_accuracy_at_least_32_38pct": bool(accuracy >= 0.3238),
        "run_improves_ipl_v5": bool(accuracy > 0.24946236559139784),
        "wicket_improves_ipl_v5": bool(wicket_brier < 0.19519474572613377),
        "latency_below_50ms": bool(latency < 50),
    }
    report = {
        "candidate_version": "ipl_engine_v7_home_phase_transition",
        "production_models_changed": False,
        "training_rows_through_2024": len(training),
        "holdout_rows_2025_plus": len(holdout),
        "holdout_matches": int(holdout["source_file"].nunique()),
        "phase_rows": phase_rows,
        "run_head": {
            "two_run_band_hit_rate": accuracy,
            "mean_band_probability": float(run_mass.mean()),
            "phase": _phase_report(holdout, actual_runs, run_low, run_high),
        },
        "wicket_head": {"brier_score": wicket_brier},
        "mean_run_and_wicket_prediction_ms": latency,
        "promotion_gates": gates,
        "integration_recommended": all(gates.values()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, output_dir / f"{name}_model.pkl")
    joblib.dump(columns, output_dir / "feature_cols.pkl")
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    destination = root / "models/candidates/ipl_engine_v7_home_phase_transition"
    print(json.dumps(train(root, destination), indent=2))
