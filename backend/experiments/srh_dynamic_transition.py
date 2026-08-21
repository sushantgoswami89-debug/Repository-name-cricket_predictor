"""SRH-only pilot for dynamic over state transitions (not production code)."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score

TEAM = "Sunrisers Hyderabad"
BASE_FEATURES = ["over", "innings", "is_home"]
DYNAMIC_FEATURES = BASE_FEATURES + [
    "previous_over_runs",
    "previous_over_wicket",
    "wicketless_over_streak",
]
MATCHUP_FEATURES = DYNAMIC_FEATURES + ["striker", "non_striker", "bowler"]
STRIKE_BOWLER_FEATURES = MATCHUP_FEATURES + [
    "bowler_prior_wicket_rate_vs_team",
    "bowler_prior_evidence_vs_team",
]


def _dismissals(delivery: dict[str, Any]) -> int:
    return sum(
        wicket.get("kind") != "retired hurt" for wicket in delivery.get("wickets", [])
    )


def _bowler_wickets(delivery: dict[str, Any]) -> int:
    non_bowler_kinds = {
        "retired hurt",
        "retired out",
        "run out",
        "obstructing the field",
    }
    return sum(
        wicket.get("kind") not in non_bowler_kinds
        for wicket in delivery.get("wickets", [])
    )


def _is_legal(delivery: dict[str, Any]) -> bool:
    extras = delivery.get("extras", {})
    return "wides" not in extras and "noballs" not in extras


def _is_home(team: str, venue: str) -> int:
    lower = venue.lower()
    if team == "Sunrisers Hyderabad":
        return int("hyderabad" in lower or "rajiv gandhi" in lower)
    if team == "Mumbai Indians":
        return int("mumbai" in lower or "wankhede" in lower)
    return 0


def build_dataset(
    project_root: Path,
    team: str = TEAM,
    max_over: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pattern = project_root / "data/raw/cricsheet/ipl/*.json"
    matches = []
    for filename in glob.glob(str(pattern)):
        raw = json.loads(Path(filename).read_text(encoding="utf-8"))
        info = raw.get("info", {})
        dates = info.get("dates", [])
        if not dates:
            continue
        date = pd.Timestamp(dates[0])
        matches.append((date, filename, raw))
    bowler_history: dict[str, list[int]] = {}
    for date, filename, raw in sorted(matches, key=lambda value: (value[0], value[1])):
        info = raw.get("info", {})
        venue = str(info.get("venue", ""))
        opponent = next(
            (name for name in info.get("teams", []) if name != team), "unknown"
        )
        innings_rows = raw.get("innings", [])
        for innings_index, innings in enumerate(innings_rows, start=1):
            if innings.get("team") != team:
                continue
            previous_runs = -1
            previous_wicket = 0
            wicketless_streak = 0
            for over in sorted(innings.get("overs", []), key=lambda row: row["over"]):
                over_number = int(over["over"]) + 1
                deliveries = over.get("deliveries", [])
                if not deliveries:
                    continue
                runs = sum(
                    int(ball.get("runs", {}).get("total", 0)) for ball in deliveries
                )
                wickets = sum(_dismissals(ball) for ball in deliveries)
                first = deliveries[0]
                bowler = str(first.get("bowler", "unknown"))
                prior_balls, prior_wickets = bowler_history.get(bowler, [0, 0])
                if over_number <= max_over:
                    rows.append(
                        {
                            "match_id": Path(filename).stem,
                            "date": date,
                            "over": over_number,
                            "innings": innings_index,
                            "opponent": opponent,
                            "is_home": _is_home(team, venue),
                            "previous_over_runs": previous_runs,
                            "previous_over_wicket": previous_wicket,
                            "wicketless_over_streak": wicketless_streak,
                            "striker": str(first.get("batter", "unknown")),
                            "non_striker": str(first.get("non_striker", "unknown")),
                            "bowler": bowler,
                            # Empirical-Bayes rate: two prior wickets per 120
                            # balls prevents tiny samples becoming strike labels.
                            "bowler_prior_wicket_rate_vs_team": (prior_wickets + 2)
                            / (prior_balls + 120),
                            "bowler_prior_evidence_vs_team": min(
                                1.0, prior_balls / 120
                            ),
                            "runs": runs,
                            "wicket": int(wickets > 0),
                        }
                    )
                history = bowler_history.setdefault(bowler, [0, 0])
                history[0] += sum(_is_legal(ball) for ball in deliveries)
                history[1] += sum(_bowler_wickets(ball) for ball in deliveries)
                previous_runs = runs
                previous_wicket = int(wickets > 0)
                wicketless_streak = 0 if wickets else wicketless_streak + 1
    return pd.DataFrame(rows).sort_values(["date", "match_id", "over"])


def _model_metrics(
    data: pd.DataFrame, features: list[str], train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, float]:
    categoricals = [
        feature
        for feature in features
        if feature in {"team", "opponent", "striker", "non_striker", "bowler"}
    ]
    prepared = data[features].copy()
    for column in categoricals:
        prepared[column] = prepared[column].astype("category")
    x_train = prepared.loc[train.index]
    x_test = prepared.loc[test.index]
    runs_model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=160,
        learning_rate=0.035,
        num_leaves=11,
        min_child_samples=25,
        reg_lambda=2.0,
        verbosity=-1,
        random_state=42,
    ).fit(x_train, train["runs"], categorical_feature=categoricals)
    wicket_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=140,
        learning_rate=0.035,
        num_leaves=9,
        min_child_samples=30,
        reg_lambda=2.0,
        verbosity=-1,
        random_state=42,
    ).fit(x_train, train["wicket"], categorical_feature=categoricals)
    runs_prediction = np.clip(runs_model.predict(x_test), 0, None)
    wicket_probability = wicket_model.predict_proba(x_test)[:, 1]
    return {
        "runs_mae": float(mean_absolute_error(test["runs"], runs_prediction)),
        "runs_bias": float(np.mean(runs_prediction - test["runs"])),
        "wicket_brier": float(brier_score_loss(test["wicket"], wicket_probability)),
        "wicket_auc": float(roc_auc_score(test["wicket"], wicket_probability)),
    }


def _next_wicket_timing(project_root: Path, team: str = TEAM) -> dict[str, Any]:
    offsets: list[int] = []
    for filename in glob.glob(str(project_root / "data/raw/cricsheet/ipl/*.json")):
        raw = json.loads(Path(filename).read_text(encoding="utf-8"))
        for innings in raw.get("innings", []):
            if innings.get("team") != team:
                continue
            wicket_overs = [
                int(over["over"]) + 1
                for over in innings.get("overs", [])
                if sum(_dismissals(ball) for ball in over.get("deliveries", [])) > 0
            ]
            if wicket_overs and wicket_overs[0] == 1 and len(wicket_overs) > 1:
                offsets.append(wicket_overs[1] - 1)
    counts = pd.Series(offsets).value_counts().sort_index()
    total = max(1, len(offsets))
    return {
        "innings_with_first_over_wicket_and_later_wicket": len(offsets),
        "next_wicket_offset_distribution": {
            str(int(offset)): {
                "count": int(count),
                "probability": float(count / total),
            }
            for offset, count in counts.items()
        },
    }


def run(project_root: Path, team: str = TEAM) -> dict[str, Any]:
    data = build_dataset(project_root, team)
    train = data[data["date"].dt.year <= 2023]
    test = data[data["date"].dt.year >= 2024]
    if train.empty or test.empty:
        raise ValueError("SRH requires both pre-2024 training and 2024+ holdout rows.")
    naive_runs = float(train["runs"].mean())
    naive_wicket = float(train["wicket"].mean())
    result = {
        "scope": f"{team} IPL batting overs 1-3 only",
        "split": "train through 2023; unseen holdout 2024+",
        "rows": {"total": len(data), "train": len(train), "holdout": len(test)},
        "matches": {
            "total": int(data["match_id"].nunique()),
            "holdout": int(test["match_id"].nunique()),
        },
        "naive_baseline": {
            "runs_mae": float(
                mean_absolute_error(test["runs"], np.full(len(test), naive_runs))
            ),
            "wicket_brier": float(
                brier_score_loss(test["wicket"], np.full(len(test), naive_wicket))
            ),
        },
        "context_baseline": _model_metrics(data, BASE_FEATURES, train, test),
        "dynamic_transition": _model_metrics(data, DYNAMIC_FEATURES, train, test),
        "dynamic_with_matchup": _model_metrics(data, MATCHUP_FEATURES, train, test),
        "next_wicket_timing": _next_wicket_timing(project_root, team),
    }
    baseline = result["context_baseline"]
    dynamic = result["dynamic_transition"]
    result["dynamic_vs_context"] = {
        "runs_mae_improvement": (baseline["runs_mae"] - dynamic["runs_mae"])
        / baseline["runs_mae"],
        "wicket_brier_improvement": (baseline["wicket_brier"] - dynamic["wicket_brier"])
        / baseline["wicket_brier"],
    }
    era_train = data[(data["date"].dt.year >= 2019) & (data["date"].dt.year <= 2024)]
    era_test = data[data["date"].dt.year >= 2025]
    era_baseline = _model_metrics(data, BASE_FEATURES, era_train, era_test)
    era_dynamic = _model_metrics(data, DYNAMIC_FEATURES, era_train, era_test)
    era_matchup = _model_metrics(data, MATCHUP_FEATURES, era_train, era_test)
    result["current_team_era_check"] = {
        "split": "2019-2024 train; unseen 2025+ holdout",
        "train_rows": len(era_train),
        "holdout_rows": len(era_test),
        "context_baseline": era_baseline,
        "dynamic_transition": era_dynamic,
        "dynamic_with_matchup": era_matchup,
        "matchup_vs_context": {
            "runs_mae_improvement": (era_baseline["runs_mae"] - era_matchup["runs_mae"])
            / era_baseline["runs_mae"],
            "wicket_brier_improvement": (
                era_baseline["wicket_brier"] - era_matchup["wicket_brier"]
            )
            / era_baseline["wicket_brier"],
        },
    }
    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    result = run(project_root)
    output = project_root / "data/reports/srh_dynamic_transition.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
