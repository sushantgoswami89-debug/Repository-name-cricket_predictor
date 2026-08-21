"""Test supported venue × bowling-type wicket interactions by match state."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.innings_phase_player_dataset import bowling_category


VENUES = ["chennai_chepauk", "bengaluru_chinnaswamy"]
MIN_TOTAL_OVERS = 100
MIN_TYPE_OVERS = 30
MIN_COMPARABLE_YEARS = 3
MIN_YEAR_TYPE_OVERS = 10


def _bowling_group(category: str) -> str:
    if category in {"leg_spin", "off_spin"}:
        return "spin"
    if category in {"fast", "medium"}:
        return "pace"
    return "unknown"


def _metrics(data: pd.DataFrame) -> dict[str, float | int]:
    balls = float(data["legal_balls"].sum())
    return {
        "overs": len(data),
        "bowler_wickets": int(data["bowler_wickets_in_over"].sum()),
        "wickets_per_over": float(data["bowler_wickets_in_over"].mean()),
        "balls_per_wicket": (
            balls / data["bowler_wickets_in_over"].sum()
            if data["bowler_wickets_in_over"].sum()
            else None
        ),
        "runs_per_over": float(data["total_runs"].mean()),
        "dot_rate": float(data["dot_balls"].sum() / balls) if balls else 0.0,
        "boundary_ball_rate": (
            float((data["fours"].sum() + data["sixes"].sum()) / balls)
            if balls
            else 0.0
        ),
    }


def analyze(root: Path) -> dict:
    data = pd.read_csv(
        root
        / "data/candidates/ipl_wicket_v8_announced_bowler/training_overs.csv"
    )
    components = pd.read_csv(
        root / "data/candidates/v3.4/over_components.csv"
    )
    keys = ["source_file", "match_date", "innings", "over"]
    data["match_date"] = data["match_date"].astype(str)
    components["match_date"] = components["match_date"].astype(str)
    data = data.merge(
        components[
            keys
            + [
                "legal_balls",
                "dot_balls",
                "fours",
                "sixes",
                "total_runs",
            ]
        ],
        on=keys,
        validate="one_to_one",
    )
    styles = pd.read_csv(
        root / "data/candidates/v3.8/bowling_styles.csv"
    ).fillna("")
    style_by_id = {
        str(row.cricsheet_id): _bowling_group(
            bowling_category(str(row.bowling_style))[0]
        )
        for row in styles.itertuples()
    }
    data["bowling_group"] = (
        data["announced_bowler"]
        .astype(str)
        .str.replace("player:", "", regex=False)
        .map(style_by_id)
        .fillna("unknown")
    )
    data["season"] = pd.to_datetime(data["match_date"]).dt.year
    data["innings_state"] = np.where(
        data["innings"] == 1, "first_innings", "chase"
    )
    data["batter_age"] = pd.cut(
        data["striker_match_balls"],
        [-1, 5, 11, 23, np.inf],
        labels=["0_5", "6_11", "12_23", "24_plus"],
    ).astype(str)
    data["recent_wicket_state"] = np.where(
        data["recent_wicket_rate"] > 0, "recent_wicket", "stable"
    )
    cohort = data[
        data["venue_name"].isin(VENUES)
        & data["bowling_group"].isin(["spin", "pace"])
    ].copy()
    dimensions = {
        "phase": "phase",
        "innings": "innings_state",
        "batter_age": "batter_age",
        "recent_wicket": "recent_wicket_state",
        "chase_pressure": "chase_pressure",
    }
    comparisons = []
    for dimension, column in dimensions.items():
        for (venue, value), segment in cohort.groupby(
            ["venue_name", column], observed=True
        ):
            spin = segment[segment["bowling_group"] == "spin"]
            pace = segment[segment["bowling_group"] == "pace"]
            if spin.empty or pace.empty:
                continue
            yearly = {}
            directions = []
            for year, year_segment in segment.groupby("season"):
                year_spin = year_segment[
                    year_segment["bowling_group"] == "spin"
                ]
                year_pace = year_segment[
                    year_segment["bowling_group"] == "pace"
                ]
                if (
                    len(year_spin) < MIN_YEAR_TYPE_OVERS
                    or len(year_pace) < MIN_YEAR_TYPE_OVERS
                ):
                    continue
                delta = float(
                    year_spin["bowler_wickets_in_over"].mean()
                    - year_pace["bowler_wickets_in_over"].mean()
                )
                yearly[str(year)] = {
                    "spin_overs": len(year_spin),
                    "pace_overs": len(year_pace),
                    "spin_minus_pace_wickets_per_over": delta,
                }
                directions.append(delta > 0)
            supported = bool(
                len(segment) >= MIN_TOTAL_OVERS
                and len(spin) >= MIN_TYPE_OVERS
                and len(pace) >= MIN_TYPE_OVERS
                and len(directions) >= MIN_COMPARABLE_YEARS
                and (
                    sum(directions) / len(directions) >= 0.75
                    or sum(directions) / len(directions) <= 0.25
                )
            )
            comparisons.append(
                {
                    "venue": venue,
                    "dimension": dimension,
                    "value": str(value),
                    "spin": _metrics(spin),
                    "pace": _metrics(pace),
                    "spin_minus_pace_wickets_per_over": float(
                        spin["bowler_wickets_in_over"].mean()
                        - pace["bowler_wickets_in_over"].mean()
                    ),
                    "comparable_years": len(directions),
                    "spin_higher_years": int(sum(directions)),
                    "supported": supported,
                    "yearly": yearly,
                }
            )
    authorized = [
        {
            "venue": row["venue"],
            "dimension": row["dimension"],
            "value": row["value"],
            "direction": (
                "SPIN_HIGHER"
                if row["spin_minus_pace_wickets_per_over"] > 0
                else "SPIN_LOWER"
            ),
        }
        for row in comparisons
        if row["supported"]
    ]
    report = {
        "analysis_version": "venue_bowling_phase_interactions_v1",
        "oracle_bowler_type": True,
        "promotion_eligible": False,
        "venues": VENUES,
        "support_policy": {
            "minimum_total_overs": MIN_TOTAL_OVERS,
            "minimum_overs_each_type": MIN_TYPE_OVERS,
            "minimum_comparable_years": MIN_COMPARABLE_YEARS,
            "minimum_year_overs_each_type": MIN_YEAR_TYPE_OVERS,
            "direction_agreement": 0.75,
        },
        "classified_over_coverage": float(
            len(cohort) / len(data[data["venue_name"].isin(VENUES)])
        ),
        "comparisons": comparisons,
        "authorized_interactions": authorized,
    }
    output = root / "data/reports/venue_bowling_phase_interactions_v1"
    output.mkdir(parents=True, exist_ok=True)
    flat = [
        {
            "venue": row["venue"],
            "dimension": row["dimension"],
            "value": row["value"],
            "spin_overs": row["spin"]["overs"],
            "pace_overs": row["pace"]["overs"],
            "spin_wickets_per_over": row["spin"]["wickets_per_over"],
            "pace_wickets_per_over": row["pace"]["wickets_per_over"],
            "spin_runs_per_over": row["spin"]["runs_per_over"],
            "pace_runs_per_over": row["pace"]["runs_per_over"],
            "spin_dot_rate": row["spin"]["dot_rate"],
            "pace_dot_rate": row["pace"]["dot_rate"],
            "spin_boundary_rate": row["spin"]["boundary_ball_rate"],
            "pace_boundary_rate": row["pace"]["boundary_ball_rate"],
            "comparable_years": row["comparable_years"],
            "spin_higher_years": row["spin_higher_years"],
            "supported": row["supported"],
        }
        for row in comparisons
    ]
    pd.DataFrame(flat).to_csv(output / "comparisons.csv", index=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    analyze(Path(__file__).resolve().parents[1])
