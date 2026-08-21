"""Wicket-only comparison of SRH and Mumbai Indians bowling."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd
from srh_dynamic_transition import _dismissals, _is_home

TEAMS = ("Sunrisers Hyderabad", "Mumbai Indians")


def build(project_root: Path) -> pd.DataFrame:
    rows = []
    for filename in glob.glob(str(project_root / "data/raw/cricsheet/ipl/*.json")):
        raw = json.loads(Path(filename).read_text(encoding="utf-8"))
        info = raw.get("info", {})
        match_teams = info.get("teams", [])
        venue = str(info.get("venue", ""))
        date = pd.Timestamp(info.get("dates", [None])[0])
        for innings_number, innings in enumerate(raw.get("innings", []), start=1):
            batting_team = innings.get("team", "")
            bowling_team = next(
                (team for team in match_teams if team != batting_team), ""
            )
            if bowling_team not in TEAMS:
                continue
            for over in innings.get("overs", []):
                deliveries = over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(over["over"]) + 1
                wickets = sum(_dismissals(ball) for ball in deliveries)
                rows.append(
                    {
                        "match_id": Path(filename).stem,
                        "date": date,
                        "bowling_team": bowling_team,
                        "batting_team": batting_team,
                        "bowling_innings": innings_number,
                        "bowling_order": (
                            "bowling_first" if innings_number == 1 else "bowling_second"
                        ),
                        "over": over_number,
                        "phase": (
                            "powerplay"
                            if over_number <= 6
                            else "middle" if over_number <= 15 else "death"
                        ),
                        "venue": venue,
                        "home_away": (
                            "home" if _is_home(bowling_team, venue) else "away"
                        ),
                        "bowler": str(deliveries[0].get("bowler", "unknown")),
                        "wicket_over": int(wickets > 0),
                        "wickets": wickets,
                        "head_to_head": batting_team in TEAMS,
                    }
                )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, groups: list[str]) -> list[dict[str, object]]:
    summary = (
        frame.groupby(groups, observed=True)
        .agg(
            overs=("wicket_over", "size"),
            wicket_overs=("wicket_over", "sum"),
            wickets=("wickets", "sum"),
        )
        .reset_index()
    )
    summary["wicket_over_rate"] = summary["wicket_overs"] / summary["overs"]
    return summary.to_dict(orient="records")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data = build(project_root)
    result = {
        "scope": "Wicket outcomes only; SRH and MI bowling; all IPL history",
        "overall": summarize(data, ["bowling_team"]),
        "by_bowling_order": summarize(data, ["bowling_team", "bowling_order"]),
        "by_phase": summarize(data, ["bowling_team", "phase"]),
        "by_order_and_phase": summarize(
            data, ["bowling_team", "bowling_order", "phase"]
        ),
        "by_home_away": summarize(data, ["bowling_team", "home_away"]),
        "head_to_head": summarize(
            data[data["head_to_head"]],
            ["bowling_team", "bowling_order", "phase"],
        ),
    }
    output = project_root / "data/reports/srh_mi_bowling_comparison.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
