"""Train a coherent joint-event IPL delivery model with beam propagation."""

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
from sklearn.preprocessing import LabelEncoder

from train_ipl_ball_simulator_v8 import FEATURES, build_dataset

BEAM_WIDTH = 24
EVENT_BRANCHES = 5
PHASES = ["powerplay", "middle", "death"]


def _frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[FEATURES].copy()
    result["phase"] = pd.Categorical(result["phase"], categories=PHASES)
    return result


def _model(classes: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=classes,
        n_estimators=110,
        learning_rate=0.055,
        num_leaves=27,
        max_depth=7,
        min_child_samples=100,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )


def _swap(state: dict[str, Any]) -> None:
    pairs = [
        ("batter_match_balls", "partner_match_balls"),
        ("batter_match_runs_per_ball", "partner_match_runs_per_ball"),
        ("batter_match_dot_rate", "partner_match_dot_rate"),
        ("batter_match_boundary_rate", "partner_match_boundary_rate"),
        ("batter_career_balls", "partner_career_balls"),
        ("batter_career_runs_per_ball", "partner_career_runs_per_ball"),
        ("batter_career_dot_rate", "partner_career_dot_rate"),
        ("batter_career_boundary_rate", "partner_career_boundary_rate"),
    ]
    for left, right in pairs:
        state[left], state[right] = state[right], state[left]


def _update_rate(
    state: dict[str, Any], prefix: str, runs: int, wicket: int, legal: int
) -> None:
    if not legal:
        return
    balls_name = f"{prefix}_balls"
    old_balls = float(state[balls_name])
    new_balls = old_balls + 1
    values = {
        "runs_per_ball": runs,
        "dot_rate": int(runs == 0),
        "boundary_rate": int(runs in {4, 6}),
    }
    if prefix == "bowler_match":
        values["wicket_rate"] = wicket
    for suffix, event_value in values.items():
        name = f"{prefix}_{suffix}"
        state[name] = (float(state[name]) * old_balls + event_value) / new_balls
    state[balls_name] = new_balls


def _new_batter(state: dict[str, Any]) -> None:
    for name in (
        "batter_match_balls",
        "batter_match_runs_per_ball",
        "batter_match_dot_rate",
        "batter_match_boundary_rate",
        "batter_career_balls",
        "batter_career_runs_per_ball",
        "batter_career_dot_rate",
        "batter_career_boundary_rate",
    ):
        state[name] = 0.0


def _advance(
    original: dict[str, Any], event: str, event_probability: float
) -> tuple[dict[str, Any], bool]:
    state = original.copy()
    kind, runs_text, wicket_text = event.split("_")
    runs = int(runs_text)
    wicket = int(wicket_text)
    legal = int(kind == "legal")
    state["_probability"] *= event_probability
    state["_total"] += runs
    state["_any_wicket"] = int(bool(state["_any_wicket"] or wicket))
    state["_legal"] += legal
    state["_attempts"] += 1
    state["score"] += runs
    state["wickets_down"] += wicket
    state["over_runs"] += runs
    state["over_dots"] += int(runs == 0)
    state["over_boundaries"] += int(runs in {4, 6})
    state["over_wickets"] += wicket
    state["ball_in_over"] = state["_legal"]
    innings_balls = (int(state["over"]) - 1) * 6 + state["_legal"]
    state["current_run_rate"] = state["score"] * 6 / max(innings_balls, 1)
    if state["is_chase"]:
        state["runs_required"] = max(state["runs_required"] - runs, 0)
        state["required_run_rate"] = (
            state["runs_required"] * 6 / max(120 - innings_balls, 1)
        )
    _update_rate(state, "batter_match", runs, wicket, legal)
    _update_rate(state, "bowler_match", runs, wicket, legal)
    old_recent = min(float(state["recent_balls"]), 12.0)
    if legal:
        new_recent = min(old_recent + 1, 12.0)
        denominator = max(new_recent, 1.0)
        for name, value in (
            ("recent_runs_per_ball", runs),
            ("recent_dot_rate", int(runs == 0)),
            ("recent_boundary_rate", int(runs in {4, 6})),
            ("recent_wicket_rate", wicket),
        ):
            weight = min(old_recent, 11.0)
            state[name] = (float(state[name]) * weight + value) / denominator
        state["recent_balls"] = new_recent
    if runs % 2:
        _swap(state)
    if wicket:
        _new_batter(state)
    complete = bool(
        state["_legal"] >= 6
        or state["_attempts"] >= 10
        or state["wickets_down"] >= 10
        or (state["is_chase"] and state["runs_required"] <= 0)
    )
    return state, complete


def _propagate(
    row: pd.Series,
    model: lgb.LGBMClassifier,
    labels: LabelEncoder,
) -> tuple[np.ndarray, float]:
    initial = row[FEATURES].to_dict()
    initial.update(
        {
            "_probability": 1.0,
            "_total": 0,
            "_any_wicket": 0,
            "_legal": 0,
            "_attempts": 0,
        }
    )
    active = [initial]
    completed: list[dict[str, Any]] = []
    while active:
        probability = model.predict_proba(_frame(pd.DataFrame(active)))
        expanded: list[dict[str, Any]] = []
        for state, distribution in zip(active, probability, strict=True):
            event_indexes = np.argsort(distribution)[-EVENT_BRANCHES:]
            for event_index in event_indexes:
                event_probability = float(distribution[event_index])
                if event_probability < 0.001:
                    continue
                event = str(labels.inverse_transform([event_index])[0])
                next_state, complete = _advance(state, event, event_probability)
                (completed if complete else expanded).append(next_state)
        if not expanded:
            break
        expanded.sort(key=lambda item: item["_probability"], reverse=True)
        active = expanded[:BEAM_WIDTH]
    completed.extend(active)
    total_probability = sum(state["_probability"] for state in completed)
    distribution = np.zeros(41)
    wicket_probability = 0.0
    for state in completed:
        normalized = state["_probability"] / max(total_probability, 1e-12)
        distribution[min(int(state["_total"]), 40)] += normalized
        wicket_probability += normalized * state["_any_wicket"]
    return distribution, wicket_probability


def train(root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(root)
    dataset["match_date"] = pd.to_datetime(dataset["match_date"])
    training = dataset[dataset["match_date"].dt.year <= 2024].reset_index(drop=True)
    holdout = dataset[dataset["match_date"].dt.year >= 2025].reset_index(drop=True)
    labels = LabelEncoder().fit(training["target_event"])
    model = _model(len(labels.classes_))
    model.fit(
        _frame(training),
        labels.transform(training["target_event"]),
        categorical_feature=["phase"],
    )
    group_keys = ["source_file", "innings", "over"]
    starts = holdout.groupby(group_keys, sort=False).head(1).reset_index(drop=True)
    actual = (
        holdout.groupby(group_keys, sort=False)
        .agg(actual_runs=("target_runs", "sum"), actual_wicket=("target_wicket", "max"))
        .reset_index()
    )
    checkpoint_path = output_dir / "propagation_checkpoint.json"
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.exists()
        else {"lows": [], "masses": [], "wicket_probabilities": []}
    )
    lows = [int(value) for value in checkpoint["lows"]]
    masses = [float(value) for value in checkpoint["masses"]]
    wicket_probabilities = [
        float(value) for value in checkpoint["wicket_probabilities"]
    ]
    started = time.perf_counter()
    for position, (_, row) in enumerate(
        starts.iloc[len(lows) :].iterrows(), start=len(lows) + 1
    ):
        distribution, wicket_probability = _propagate(row, model, labels)
        band_mass = np.convolve(distribution, np.ones(3), mode="valid")
        low = int(np.argmax(band_mass))
        lows.append(low)
        masses.append(float(band_mass[low]))
        wicket_probabilities.append(float(np.clip(wicket_probability, 0.0, 1.0)))
        if position % 50 == 0:
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "lows": lows,
                        "masses": masses,
                        "wicket_probabilities": wicket_probabilities,
                    }
                ),
                encoding="utf-8",
            )
    latency = (time.perf_counter() - started) * 1000 / len(starts)
    low_array = np.asarray(lows)
    actual_runs = actual["actual_runs"].to_numpy()
    hit = (actual_runs >= low_array) & (actual_runs <= low_array + 2)
    phases = starts["phase"].to_numpy()
    phase_report = {
        phase: {
            "rows": int((phases == phase).sum()),
            "hit_rate": float(hit[phases == phase].mean()),
        }
        for phase in PHASES
    }
    accuracy = float(hit.mean())
    wicket_brier = float(
        brier_score_loss(actual["actual_wicket"], wicket_probabilities)
    )
    gates = {
        "run_accuracy_at_least_32_38pct": bool(accuracy >= 0.3238),
        "run_improves_ipl_v5": bool(accuracy > 0.24946236559139784),
        "wicket_improves_ipl_v5": bool(wicket_brier < 0.19519474572613377),
        "propagation_below_50ms_per_over": bool(latency < 50),
    }
    report = {
        "candidate_version": "ipl_engine_v9_joint_event_propagation",
        "production_models_changed": False,
        "joint_event_classes": labels.classes_.tolist(),
        "training_deliveries_2023_2024": len(training),
        "holdout_deliveries_2025_plus": len(holdout),
        "holdout_overs": len(starts),
        "beam_width": BEAM_WIDTH,
        "two_run_band_hit_rate": accuracy,
        "mean_band_probability": float(np.mean(masses)),
        "phase": phase_report,
        "wicket_brier": wicket_brier,
        "mean_propagation_ms_per_over": latency,
        "promotion_gates": gates,
        "integration_recommended": all(gates.values()),
    }
    joblib.dump(model, output_dir / "joint_delivery_event_model.pkl")
    joblib.dump(labels, output_dir / "event_label_encoder.pkl")
    joblib.dump(FEATURES, output_dir / "feature_cols.pkl")
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    destination = project_root / "models/candidates/ipl_engine_v9_joint_event"
    print(json.dumps(train(project_root, destination), indent=2))
