"""Full 20-over dynamic transition pilot for SRH and MI, both directions."""

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
        frame = pilot.build_dataset(project_root, team, max_over=20)
        frame["team"] = team
        frames.append(frame)
    data = (
        pd.concat(frames)
        .sort_values(["date", "match_id", "innings", "over"])
        .reset_index(drop=True)
    )
    data["phase"] = pd.cut(
        data["over"],
        bins=[0, 6, 15, 20],
        labels=["powerplay", "middle", "death"],
    ).astype("str")
    exact_matchup = data[
        data.apply(lambda row: {row["team"], row["opponent"]} == set(TEAMS), axis=1)
    ]
    train = data[(data["date"].dt.year >= 2019) & (data["date"].dt.year <= 2024)]
    holdout = exact_matchup[exact_matchup["date"].dt.year >= 2025]

    context = ["team", "opponent", *pilot.BASE_FEATURES]
    dynamic = ["team", "opponent", *pilot.DYNAMIC_FEATURES]
    matchup = ["team", "opponent", *pilot.MATCHUP_FEATURES]
    strike = ["team", "opponent", *pilot.STRIKE_BOWLER_FEATURES]
    metrics = {
        "context_baseline": pilot._model_metrics(data, context, train, holdout),
        "dynamic_transition": pilot._model_metrics(data, dynamic, train, holdout),
        "dynamic_with_matchup": pilot._model_metrics(data, matchup, train, holdout),
        "dynamic_with_strike_bowler": pilot._model_metrics(
            data, strike, train, holdout
        ),
    }
    phase_metrics = {}
    for phase in ("powerplay", "middle", "death"):
        phase_train = train[train["phase"] == phase]
        phase_holdout = holdout[holdout["phase"] == phase]
        phase_dynamic = pilot._model_metrics(data, dynamic, phase_train, phase_holdout)
        phase_strike = pilot._model_metrics(data, strike, phase_train, phase_holdout)
        phase_metrics[phase] = {
            "rows": {
                "train": len(phase_train),
                "holdout": len(phase_holdout),
            },
            "dynamic": phase_dynamic,
            "strike_bowler": phase_strike,
            "strike_vs_dynamic": {
                "runs_mae_improvement": (
                    phase_dynamic["runs_mae"] - phase_strike["runs_mae"]
                )
                / phase_dynamic["runs_mae"],
                "wicket_brier_improvement": (
                    phase_dynamic["wicket_brier"] - phase_strike["wicket_brier"]
                )
                / phase_dynamic["wicket_brier"],
                "wicket_auc_change": (
                    phase_strike["wicket_auc"] - phase_dynamic["wicket_auc"]
                ),
            },
        }
    baseline = metrics["context_baseline"]
    strike_result = metrics["dynamic_with_strike_bowler"]
    result = {
        "scope": "SRH vs MI and MI vs SRH, overs 1-20",
        "support_pool": "All SRH and MI batting innings, 2019-2024",
        "holdout": "Exact head-to-head matches, 2025+",
        "rows": {
            "support_train": len(train),
            "exact_head_to_head_all": len(exact_matchup),
            "exact_holdout": len(holdout),
            "exact_holdout_matches": int(holdout["match_id"].nunique()),
        },
        "metrics": metrics,
        "phase_metrics": phase_metrics,
        "strike_vs_context": {
            "runs_mae_improvement": (baseline["runs_mae"] - strike_result["runs_mae"])
            / baseline["runs_mae"],
            "wicket_brier_improvement": (
                baseline["wicket_brier"] - strike_result["wicket_brier"]
            )
            / baseline["wicket_brier"],
            "wicket_auc_change": (strike_result["wicket_auc"] - baseline["wicket_auc"]),
        },
    }
    output = project_root / "data/reports/srh_mi_full_innings_transition.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
