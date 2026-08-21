"""Measure situational outcomes across every identity-safe IPL debut window."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


MIN_ROWS = 100
MIN_YEAR_ROWS = 20
MIN_STABLE_YEARS = 3


def _ball_bucket(balls: pd.Series) -> pd.Series:
    return pd.cut(
        balls,
        bins=[-1, 5, 11, 23, np.inf],
        labels=["0_5", "6_11", "12_23", "24_plus"],
    ).astype(str)


def _pressure_trigger(data: pd.DataFrame) -> np.ndarray:
    return np.select(
        [
            data["recent_wicket_rate"].to_numpy() > 0,
            data["chase_pressure"].to_numpy() == "high",
            data["wickets_in_hand"].to_numpy() <= 3,
        ],
        ["recent_wicket", "high_required_rate", "three_wickets_or_less"],
        default="neutral",
    )


def _summary(data: pd.DataFrame) -> dict[str, float | int]:
    return {
        "rows": len(data),
        "mean_runs": float(data["runs_in_over"].mean()),
        "wicket_rate": float(data["wicket_in_over"].mean()),
        "boundary_pressure_rate": float(data["recent_boundary_rate"].mean()),
    }


def analyze(root: Path) -> dict:
    data = pd.read_csv(
        root / "data/candidates/ipl_cold_start_v5_t20_priors/training_overs.csv"
    )
    data["match_date"] = pd.to_datetime(data["match_date"])
    cohort = data[
        (data["striker_prior_balls"] < 120)
        & (data["cold_start_is_bowler"] == 0)
    ].copy()
    cohort["season"] = cohort["match_date"].dt.year
    cohort["ball_bucket"] = _ball_bucket(cohort["striker_match_balls"])
    cohort["pressure_trigger"] = _pressure_trigger(cohort)

    rows = []
    for keys, segment in cohort.groupby(
        ["ball_bucket", "pressure_trigger", "phase"], observed=True
    ):
        ball_bucket, trigger, phase = keys
        neutral = cohort[
            (cohort["ball_bucket"] == ball_bucket)
            & (cohort["pressure_trigger"] == "neutral")
            & (cohort["phase"] == phase)
        ]
        row = {
            "ball_bucket": ball_bucket,
            "pressure_trigger": trigger,
            "phase": phase,
            **_summary(segment),
            "neutral_rows": len(neutral),
            "runs_delta_vs_neutral": (
                float(
                    segment["runs_in_over"].mean()
                    - neutral["runs_in_over"].mean()
                )
                if len(neutral)
                else None
            ),
            "wicket_delta_vs_neutral": (
                float(
                    segment["wicket_in_over"].mean()
                    - neutral["wicket_in_over"].mean()
                )
                if len(neutral)
                else None
            ),
        }
        stable = []
        yearly = {}
        for year, year_segment in segment.groupby("season"):
            year_neutral = neutral[neutral["season"] == year]
            if len(year_segment) < MIN_YEAR_ROWS or len(year_neutral) < MIN_YEAR_ROWS:
                continue
            wicket_delta = float(
                year_segment["wicket_in_over"].mean()
                - year_neutral["wicket_in_over"].mean()
            )
            runs_delta = float(
                year_segment["runs_in_over"].mean()
                - year_neutral["runs_in_over"].mean()
            )
            yearly[str(year)] = {
                "rows": len(year_segment),
                "runs_delta_vs_neutral": runs_delta,
                "wicket_delta_vs_neutral": wicket_delta,
            }
            stable.append(wicket_delta > 0)
        row["comparable_years"] = len(stable)
        row["positive_wicket_delta_years"] = int(sum(stable))
        row["yearly"] = yearly
        row["supported"] = bool(
            trigger != "neutral"
            and
            len(segment) >= MIN_ROWS
            and len(stable) >= MIN_STABLE_YEARS
            and (
                sum(stable) / len(stable) >= 0.75
                or sum(stable) / len(stable) <= 0.25
            )
        )
        rows.append(row)

    survived = cohort[cohort["ball_bucket"].isin(["12_23", "24_plus"])]
    entry = cohort[cohort["ball_bucket"].isin(["0_5", "6_11"])]
    yearly_transition = {}
    signs = []
    for year in sorted(cohort["season"].unique()):
        before = entry[entry["season"] == year]
        after = survived[survived["season"] == year]
        if len(before) < MIN_YEAR_ROWS or len(after) < MIN_YEAR_ROWS:
            continue
        run_delta = float(
            after["runs_in_over"].mean() - before["runs_in_over"].mean()
        )
        wicket_delta = float(
            after["wicket_in_over"].mean() - before["wicket_in_over"].mean()
        )
        yearly_transition[str(year)] = {
            "entry_rows": len(before),
            "survived_rows": len(after),
            "runs_delta": run_delta,
            "wicket_delta": wicket_delta,
        }
        signs.append((run_delta > 0, wicket_delta > 0))
    transition = {
        "entry": _summary(entry),
        "survived_12_plus": _summary(survived),
        "runs_delta": float(
            survived["runs_in_over"].mean() - entry["runs_in_over"].mean()
        ),
        "wicket_delta": float(
            survived["wicket_in_over"].mean() - entry["wicket_in_over"].mean()
        ),
        "comparable_years": len(signs),
        "years_runs_and_wickets_both_higher": int(
            sum(run_up and wicket_up for run_up, wicket_up in signs)
        ),
        "yearly": yearly_transition,
    }
    transition["supported_high_variance"] = bool(
        len(signs) >= MIN_STABLE_YEARS
        and sum(run_up and wicket_up for run_up, wicket_up in signs)
        / len(signs)
        >= 0.75
    )
    report = {
        "analysis_version": "historical_ipl_debut_states_v1",
        "identity_safe": True,
        "future_information_used": False,
        "cohort_definition": (
            "non-bowler striker with fewer than 120 prior IPL balls"
        ),
        "cohort_rows": len(cohort),
        "players": int(cohort["striker"].nunique()),
        "seasons": [
            int(year) for year in sorted(cohort["season"].unique())
        ],
        "support_policy": {
            "minimum_rows": MIN_ROWS,
            "minimum_rows_per_comparable_year": MIN_YEAR_ROWS,
            "minimum_comparable_years": MIN_STABLE_YEARS,
            "direction_agreement": 0.75,
        },
        "survival_transition": transition,
        "states": rows,
        "authorized_states": [
            {
                "ball_bucket": row["ball_bucket"],
                "pressure_trigger": row["pressure_trigger"],
                "phase": row["phase"],
            }
            for row in rows
            if row["supported"]
        ],
    }
    output = root / "data/reports/historical_ipl_debut_states_v1"
    output.mkdir(parents=True, exist_ok=True)
    flat = [{k: v for k, v in row.items() if k != "yearly"} for row in rows]
    pd.DataFrame(flat).to_csv(output / "state_metrics.csv", index=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    analyze(Path(__file__).resolve().parents[1])
