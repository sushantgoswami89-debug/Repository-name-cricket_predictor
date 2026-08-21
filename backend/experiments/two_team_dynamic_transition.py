"""Second-layer pilot: one shared SRH and Mumbai Indians data pool."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import srh_dynamic_transition as pilot

TEAMS = ("Sunrisers Hyderabad", "Mumbai Indians")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    frames = []
    for team in TEAMS:
        frame = pilot.build_dataset(project_root, team)
        frame["team"] = team
        frames.append(frame)
    data = (
        pd.concat(frames)
        .sort_values(["date", "match_id", "innings", "over"])
        .reset_index(drop=True)
    )

    context_features = ["team", *pilot.BASE_FEATURES]
    dynamic_features = ["team", *pilot.DYNAMIC_FEATURES]
    matchup_features = ["team", *pilot.MATCHUP_FEATURES]
    strike_features = ["team", *pilot.STRIKE_BOWLER_FEATURES]
    train = data[data["date"].dt.year <= 2023]
    holdout = data[data["date"].dt.year >= 2024]
    era_train = data[(data["date"].dt.year >= 2019) & (data["date"].dt.year <= 2024)]
    era_holdout = data[data["date"].dt.year >= 2025]

    def evaluate(train_rows, test_rows):
        baseline = pilot._model_metrics(data, context_features, train_rows, test_rows)
        dynamic = pilot._model_metrics(data, dynamic_features, train_rows, test_rows)
        matchup = pilot._model_metrics(data, matchup_features, train_rows, test_rows)
        strike = pilot._model_metrics(data, strike_features, train_rows, test_rows)
        return {
            "rows": {"train": len(train_rows), "holdout": len(test_rows)},
            "context_baseline": baseline,
            "dynamic_transition": dynamic,
            "dynamic_with_matchup": matchup,
            "dynamic_with_strike_bowler": strike,
            "matchup_vs_context": {
                "runs_mae_improvement": (baseline["runs_mae"] - matchup["runs_mae"])
                / baseline["runs_mae"],
                "wicket_brier_improvement": (
                    baseline["wicket_brier"] - matchup["wicket_brier"]
                )
                / baseline["wicket_brier"],
            },
            "strike_bowler_vs_matchup": {
                "wicket_brier_improvement": (
                    matchup["wicket_brier"] - strike["wicket_brier"]
                )
                / matchup["wicket_brier"],
                "wicket_auc_change": strike["wicket_auc"] - matchup["wicket_auc"],
            },
        }

    result = {
        "scope": "One SRH+MI pool; team label retained; IPL batting overs 1-3",
        "teams": list(TEAMS),
        "rows": len(data),
        "all_history": evaluate(train, holdout),
        "current_team_era": evaluate(era_train, era_holdout),
        "wicket_timing": {
            team: pilot._next_wicket_timing(project_root, team) for team in TEAMS
        },
    }
    output = project_root / "data/reports/srh_mi_pooled_transition.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
