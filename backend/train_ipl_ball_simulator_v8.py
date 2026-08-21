"""Train and backtest a sequential IPL ball-by-ball over simulator."""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from app.ml.candidate_v3_dataset import (
    KNOWN_RULE_ANOMALY_EXCLUSIONS,
    _is_legal,
    _is_wicket,
)

MAX_BALL_RUNS = 7
SIMULATIONS = 160
FEATURES = [
    "over",
    "ball_in_over",
    "score",
    "wickets_down",
    "current_run_rate",
    "is_chase",
    "runs_required",
    "required_run_rate",
    "batter_match_balls",
    "batter_match_runs_per_ball",
    "batter_match_dot_rate",
    "batter_match_boundary_rate",
    "partner_match_balls",
    "partner_match_runs_per_ball",
    "partner_match_dot_rate",
    "partner_match_boundary_rate",
    "bowler_match_balls",
    "bowler_match_runs_per_ball",
    "bowler_match_dot_rate",
    "bowler_match_boundary_rate",
    "bowler_match_wicket_rate",
    "batter_career_balls",
    "batter_career_runs_per_ball",
    "batter_career_dot_rate",
    "batter_career_boundary_rate",
    "partner_career_balls",
    "partner_career_runs_per_ball",
    "partner_career_dot_rate",
    "partner_career_boundary_rate",
    "bowler_career_balls",
    "bowler_career_runs_per_ball",
    "bowler_career_dot_rate",
    "bowler_career_boundary_rate",
    "bowler_career_wicket_rate",
    "recent_runs_per_ball",
    "recent_balls",
    "recent_dot_rate",
    "recent_boundary_rate",
    "recent_wicket_rate",
    "over_runs",
    "over_dots",
    "over_boundaries",
    "over_wickets",
    "previous_over_runs",
    "previous_over_boundaries",
    "previous_over_wickets",
    "phase",
]


def _empty() -> dict[str, int]:
    return {"balls": 0, "runs": 0, "dots": 0, "boundaries": 0, "wickets": 0}


def _rate(profile: dict[str, int], name: str) -> float:
    return profile[name] / profile["balls"] if profile["balls"] else 0.0


def _phase(over: int) -> str:
    return "powerplay" if over <= 6 else "middle" if over <= 15 else "death"


def _event(delivery: dict[str, Any], wicket: int) -> str:
    total = min(int(delivery["runs"]["total"]), MAX_BALL_RUNS)
    extras = delivery.get("extras", {})
    kind = "wide" if "wides" in extras else "noball" if "noballs" in extras else "legal"
    return f"{kind}_{total}_{wicket}"


def _frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[FEATURES].copy()
    result["phase"] = pd.Categorical(
        result["phase"], categories=["powerplay", "middle", "death"]
    )
    return result


def build_dataset(root: Path) -> pd.DataFrame:
    paths: list[tuple[str, str, Path]] = []
    excluded = set(KNOWN_RULE_ANOMALY_EXCLUSIONS)
    for scope in ("t20i", "ipl"):
        for path in (root / "data/raw/cricsheet" / scope).glob("*.json"):
            if path.stem in excluded:
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            paths.append((str(raw["info"]["dates"][0]), scope, path))
    paths.sort(key=lambda item: (item[0], item[2].name))
    batter_career: dict[str, dict[str, int]] = defaultdict(_empty)
    bowler_career: dict[str, dict[str, int]] = defaultdict(_empty)
    rows: list[dict[str, Any]] = []
    for match_date, scope, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if str(raw["info"].get("gender", "")).lower() != "male":
            continue
        if str(raw["info"].get("match_type", "")).upper() != "T20":
            continue
        first_total: int | None = None
        staged: list[tuple[str, str, int, int, int]] = []
        for innings_number, innings in enumerate(raw.get("innings", []), start=1):
            score = wickets = legal_balls = 0
            target = (
                first_total + 1
                if innings_number == 2 and first_total is not None
                else 0
            )
            recent: deque[tuple[int, int, int, int]] = deque(maxlen=12)
            bat_match: dict[str, dict[str, int]] = defaultdict(_empty)
            bowl_match: dict[str, dict[str, int]] = defaultdict(_empty)
            previous = {"runs": -1, "boundaries": -1, "wickets": -1}
            for source_over in innings.get("overs", []):
                over_number = int(source_over["over"]) + 1
                over_runs = over_dots = over_boundaries = over_wickets = 0
                for delivery_index, delivery in enumerate(
                    source_over.get("deliveries", []), start=1
                ):
                    batter = str(delivery["batter"])
                    partner = str(delivery["non_striker"])
                    bowler = str(delivery["bowler"])
                    batter_now = bat_match[batter]
                    partner_now = bat_match[partner]
                    bowler_now = bowl_match[bowler]
                    batter_prior = batter_career[batter]
                    partner_prior = batter_career[partner]
                    bowler_prior = bowler_career[bowler]
                    recent_balls = len(recent)
                    runs_required = max(target - score, 0) if target else 0
                    balls_remaining = max(120 - legal_balls, 1)
                    target_runs = min(int(delivery["runs"]["total"]), MAX_BALL_RUNS)
                    wicket = int(_is_wicket(delivery))
                    legal = int(_is_legal(delivery))
                    if scope == "ipl" and int(match_date[:4]) >= 2023:
                        rows.append(
                            {
                                "source_file": path.name,
                                "match_date": match_date,
                                "innings": innings_number,
                                "over": over_number,
                                "delivery_index": delivery_index,
                                "ball_in_over": legal_balls % 6,
                                "score": score,
                                "wickets_down": wickets,
                                "current_run_rate": score * 6 / legal_balls
                                if legal_balls
                                else 0.0,
                                "is_chase": int(bool(target)),
                                "runs_required": runs_required,
                                "required_run_rate": runs_required * 6 / balls_remaining
                                if target
                                else 0.0,
                                "batter_match_balls": batter_now["balls"],
                                "batter_match_runs_per_ball": _rate(batter_now, "runs"),
                                "batter_match_dot_rate": _rate(batter_now, "dots"),
                                "batter_match_boundary_rate": _rate(
                                    batter_now, "boundaries"
                                ),
                                "partner_match_balls": partner_now["balls"],
                                "partner_match_runs_per_ball": _rate(
                                    partner_now, "runs"
                                ),
                                "partner_match_dot_rate": _rate(partner_now, "dots"),
                                "partner_match_boundary_rate": _rate(
                                    partner_now, "boundaries"
                                ),
                                "bowler_match_balls": bowler_now["balls"],
                                "bowler_match_runs_per_ball": _rate(bowler_now, "runs"),
                                "bowler_match_dot_rate": _rate(bowler_now, "dots"),
                                "bowler_match_boundary_rate": _rate(
                                    bowler_now, "boundaries"
                                ),
                                "bowler_match_wicket_rate": _rate(
                                    bowler_now, "wickets"
                                ),
                                "batter_career_balls": batter_prior["balls"],
                                "batter_career_runs_per_ball": _rate(
                                    batter_prior, "runs"
                                ),
                                "batter_career_dot_rate": _rate(batter_prior, "dots"),
                                "batter_career_boundary_rate": _rate(
                                    batter_prior, "boundaries"
                                ),
                                "partner_career_balls": partner_prior["balls"],
                                "partner_career_runs_per_ball": _rate(
                                    partner_prior, "runs"
                                ),
                                "partner_career_dot_rate": _rate(partner_prior, "dots"),
                                "partner_career_boundary_rate": _rate(
                                    partner_prior, "boundaries"
                                ),
                                "bowler_career_balls": bowler_prior["balls"],
                                "bowler_career_runs_per_ball": _rate(
                                    bowler_prior, "runs"
                                ),
                                "bowler_career_dot_rate": _rate(bowler_prior, "dots"),
                                "bowler_career_boundary_rate": _rate(
                                    bowler_prior, "boundaries"
                                ),
                                "bowler_career_wicket_rate": _rate(
                                    bowler_prior, "wickets"
                                ),
                                "recent_runs_per_ball": sum(x[0] for x in recent)
                                / recent_balls
                                if recent_balls
                                else 0.0,
                                "recent_balls": recent_balls,
                                "recent_dot_rate": sum(x[1] for x in recent)
                                / recent_balls
                                if recent_balls
                                else 0.0,
                                "recent_boundary_rate": sum(x[2] for x in recent)
                                / recent_balls
                                if recent_balls
                                else 0.0,
                                "recent_wicket_rate": sum(x[3] for x in recent)
                                / recent_balls
                                if recent_balls
                                else 0.0,
                                "over_runs": over_runs,
                                "over_dots": over_dots,
                                "over_boundaries": over_boundaries,
                                "over_wickets": over_wickets,
                                "previous_over_runs": previous["runs"],
                                "previous_over_boundaries": previous["boundaries"],
                                "previous_over_wickets": previous["wickets"],
                                "phase": _phase(over_number),
                                "target_runs": target_runs,
                                "target_wicket": wicket,
                                "target_legal": legal,
                                "target_event": _event(delivery, wicket),
                            }
                        )
                    total = int(delivery["runs"]["total"])
                    batter_runs = int(delivery["runs"]["batter"])
                    boundary = int(batter_runs in {4, 6})
                    dot = int(total == 0)
                    score += total
                    wickets += wicket
                    over_runs += total
                    over_dots += dot
                    over_boundaries += boundary
                    over_wickets += wicket
                    if legal:
                        legal_balls += 1
                        recent.append((total, dot, boundary, wicket))
                        for profile in (batter_now,):
                            profile["balls"] += 1
                            profile["runs"] += batter_runs
                            profile["dots"] += dot
                            profile["boundaries"] += boundary
                            profile["wickets"] += wicket
                        bowler_now["balls"] += 1
                        bowler_now["runs"] += total
                        bowler_now["dots"] += dot
                        bowler_now["boundaries"] += boundary
                        bowler_now["wickets"] += wicket
                        staged.append((batter, bowler, batter_runs, total, wicket))
                previous = {
                    "runs": over_runs,
                    "boundaries": over_boundaries,
                    "wickets": over_wickets,
                }
            if innings_number == 1:
                first_total = score
        for batter, bowler, batter_runs, total, wicket in staged:
            bp = batter_career[batter]
            bp["balls"] += 1
            bp["runs"] += batter_runs
            bp["dots"] += int(total == 0)
            bp["boundaries"] += int(batter_runs in {4, 6})
            bp["wickets"] += wicket
            wp = bowler_career[bowler]
            wp["balls"] += 1
            wp["runs"] += total
            wp["dots"] += int(total == 0)
            wp["boundaries"] += int(batter_runs in {4, 6})
            wp["wickets"] += wicket
    return pd.DataFrame(rows)


def _classifier(objective: str, classes: int | None = None) -> lgb.LGBMClassifier:
    kwargs: dict[str, Any] = {"objective": objective}
    if classes:
        kwargs["num_class"] = classes
    return lgb.LGBMClassifier(
        **kwargs,
        n_estimators=90,
        learning_rate=0.06,
        num_leaves=25,
        max_depth=6,
        min_child_samples=120,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )


def _simulate_over(
    row: pd.Series,
    run_model: lgb.LGBMClassifier,
    wicket_model: lgb.LGBMClassifier,
    legal_model: lgb.LGBMClassifier,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    state = pd.DataFrame([row[FEATURES].to_dict()] * SIMULATIONS)
    totals = np.zeros(SIMULATIONS, dtype=int)
    any_wicket = np.zeros(SIMULATIONS, dtype=bool)
    legal_balls = np.zeros(SIMULATIONS, dtype=int)
    active = np.ones(SIMULATIONS, dtype=bool)
    attempts = 0
    while active.any() and attempts < 10:
        indexes = np.flatnonzero(active)
        current = state.iloc[indexes].copy()
        run_probability = run_model.predict_proba(_frame(current))
        wicket_probability = wicket_model.predict_proba(_frame(current))[:, 1]
        legal_probability = legal_model.predict_proba(_frame(current))[:, 1]
        draws = rng.random(len(indexes))
        cumulative = np.cumsum(run_probability, axis=1)
        runs = (draws[:, None] > cumulative).sum(axis=1)
        wickets = rng.random(len(indexes)) < wicket_probability
        legal = rng.random(len(indexes)) < legal_probability
        totals[indexes] += runs
        any_wicket[indexes] |= wickets
        legal_balls[indexes] += legal
        state.loc[indexes, "score"] += runs
        state.loc[indexes, "over_runs"] += runs
        state.loc[indexes, "over_dots"] += runs == 0
        state.loc[indexes, "over_boundaries"] += np.isin(runs, [4, 6])
        state.loc[indexes, "over_wickets"] += wickets
        state.loc[indexes, "wickets_down"] += wickets
        state.loc[indexes, "ball_in_over"] = legal_balls[indexes]
        innings_balls = (state.loc[indexes, "over"].to_numpy() - 1) * 6 + legal_balls[
            indexes
        ]
        state.loc[indexes, "current_run_rate"] = (
            state.loc[indexes, "score"].to_numpy() * 6 / np.maximum(innings_balls, 1)
        )
        if int(row["is_chase"]):
            remaining = np.maximum(120 - innings_balls, 1)
            state.loc[indexes, "runs_required"] = np.maximum(
                state.loc[indexes, "runs_required"].to_numpy() - runs, 0
            )
            state.loc[indexes, "required_run_rate"] = (
                state.loc[indexes, "runs_required"].to_numpy() * 6 / remaining
            )
        active = legal_balls < 6
        attempts += 1
    return totals, any_wicket


def train(root: Path, output_dir: Path) -> dict[str, Any]:
    dataset = build_dataset(root)
    dataset["match_date"] = pd.to_datetime(dataset["match_date"])
    training = dataset[dataset["match_date"].dt.year <= 2024].reset_index(drop=True)
    holdout = dataset[dataset["match_date"].dt.year >= 2025].reset_index(drop=True)
    run_model = _classifier("multiclass", MAX_BALL_RUNS + 1)
    wicket_model = _classifier("binary")
    legal_model = _classifier("binary")
    x_train = _frame(training)
    for model, target in (
        (run_model, "target_runs"),
        (wicket_model, "target_wicket"),
        (legal_model, "target_legal"),
    ):
        model.fit(x_train, training[target], categorical_feature=["phase"])
    starts = holdout.groupby(["source_file", "innings", "over"], sort=False).head(1)
    actual = (
        holdout.groupby(["source_file", "innings", "over"], sort=False)
        .agg(actual_runs=("target_runs", "sum"), actual_wicket=("target_wicket", "max"))
        .reset_index()
    )
    rng = np.random.default_rng(42)
    lows: list[int] = []
    highs: list[int] = []
    masses: list[float] = []
    wicket_probabilities: list[float] = []
    started = time.perf_counter()
    for _, row in starts.iterrows():
        totals, wickets = _simulate_over(row, run_model, wicket_model, legal_model, rng)
        counts = np.bincount(np.minimum(totals, 40), minlength=41)
        sums = np.convolve(counts, np.ones(3, dtype=int), mode="valid")
        low = int(np.argmax(sums))
        lows.append(low)
        highs.append(low + 2)
        masses.append(float(sums[low] / SIMULATIONS))
        wicket_probabilities.append(float(wickets.mean()))
    elapsed_ms = (time.perf_counter() - started) * 1000 / len(starts)
    actual_runs = actual["actual_runs"].to_numpy()
    low_array = np.asarray(lows)
    high_array = np.asarray(highs)
    hit = (actual_runs >= low_array) & (actual_runs <= high_array)
    phase_values = starts["phase"].to_numpy()
    phase_report = {
        phase: {
            "rows": int((phase_values == phase).sum()),
            "hit_rate": float(hit[phase_values == phase].mean()),
        }
        for phase in ("powerplay", "middle", "death")
    }
    wicket_brier = float(
        brier_score_loss(actual["actual_wicket"], wicket_probabilities)
    )
    accuracy = float(hit.mean())
    gates = {
        "run_accuracy_at_least_32_38pct": bool(accuracy >= 0.3238),
        "run_improves_ipl_v5": bool(accuracy > 0.24946236559139784),
        "wicket_improves_ipl_v5": bool(wicket_brier < 0.19519474572613377),
        "simulation_below_50ms_per_over": bool(elapsed_ms < 50),
    }
    report = {
        "candidate_version": "ipl_engine_v8_ball_simulator",
        "production_models_changed": False,
        "training_deliveries_2023_2024": len(training),
        "holdout_deliveries_2025_plus": len(holdout),
        "holdout_overs": len(starts),
        "simulations_per_over": SIMULATIONS,
        "two_run_band_hit_rate": accuracy,
        "mean_simulation_band_probability": float(np.mean(masses)),
        "phase": phase_report,
        "wicket_brier": wicket_brier,
        "mean_simulation_ms_per_over": elapsed_ms,
        "promotion_gates": gates,
        "integration_recommended": all(gates.values()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(run_model, output_dir / "ball_runs_model.pkl")
    joblib.dump(wicket_model, output_dir / "ball_wicket_model.pkl")
    joblib.dump(legal_model, output_dir / "ball_legality_model.pkl")
    joblib.dump(FEATURES, output_dir / "feature_cols.pkl")
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    destination = project_root / "models/candidates/ipl_engine_v8_ball_simulator"
    print(json.dumps(train(project_root, destination), indent=2))
