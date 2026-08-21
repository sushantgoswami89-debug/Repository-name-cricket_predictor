"""Shared-history impact quality and match-result targets for the IPL adapter."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.candidate_v3_dataset import _is_legal, _is_wicket
from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS


def _empty_batter() -> dict[str, int]:
    return {"balls": 0, "runs": 0, "boundaries": 0, "dismissals": 0}


def _empty_bowler() -> dict[str, int]:
    return {"balls": 0, "runs": 0, "dots": 0, "wickets": 0}


def _batting_strength(profile: dict[str, int], prior: dict[str, int]) -> float:
    balls = profile["balls"]
    prior_balls = max(1, prior["balls"])
    runs = (profile["runs"] + 60 * prior["runs"] / prior_balls) / (balls + 60)
    boundaries = (profile["boundaries"] + 60 * prior["boundaries"] / prior_balls) / (
        balls + 60
    )
    dismissals = (profile["dismissals"] + 60 * prior["dismissals"] / prior_balls) / (
        balls + 60
    )
    return runs + 4.0 * boundaries - 3.0 * dismissals


def _bowling_strength(profile: dict[str, int], prior: dict[str, int]) -> float:
    balls = profile["balls"]
    prior_balls = max(1, prior["balls"])
    runs = (profile["runs"] + 90 * prior["runs"] / prior_balls) / (balls + 90)
    dots = (profile["dots"] + 90 * prior["dots"] / prior_balls) / (balls + 90)
    wickets = (profile["wickets"] + 90 * prior["wickets"] / prior_balls) / (balls + 90)
    return -runs + 2.0 * dots + 6.0 * wickets


def build_ipl_adapter_dataset(project_root: Path) -> pd.DataFrame:
    paths: list[tuple[str, str, Path]] = []
    excluded = set(KNOWN_RULE_ANOMALY_EXCLUSIONS)
    for scope in ("t20i", "ipl"):
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json"):
            if path.stem in excluded:
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            paths.append((str(raw["info"]["dates"][0]), scope, path))
    paths.sort(key=lambda item: (item[0], item[2].name))

    batters: dict[str, dict[str, int]] = defaultdict(_empty_batter)
    bowlers: dict[str, dict[str, int]] = defaultdict(_empty_bowler)
    global_batter = _empty_batter()
    global_bowler = _empty_bowler()
    rows: list[dict[str, Any]] = []
    for match_date, scope, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        teams = [str(team) for team in raw["info"].get("teams", [])]
        winner = str(raw["info"].get("outcome", {}).get("winner", ""))
        impact: dict[str, dict[str, float | int]] = {}
        staged: list[tuple[str, str, dict[str, int], dict[str, int]]] = []

        def apply_replacement(
            delivery: dict[str, Any],
            over_number: int,
            impact_state: dict[str, dict[str, float | int]] = impact,
        ) -> None:
            for replacement in delivery.get("replacements", {}).get("match", []):
                if replacement.get("reason") != "impact_player":
                    continue
                incoming = str(replacement.get("in", ""))
                outgoing = str(replacement.get("out", ""))
                team = str(replacement.get("team", ""))
                impact_state[team] = {
                    "over": over_number,
                    "batting_delta": _batting_strength(batters[incoming], global_batter)
                    - _batting_strength(batters[outgoing], global_batter),
                    "bowling_delta": _bowling_strength(bowlers[incoming], global_bowler)
                    - _bowling_strength(bowlers[outgoing], global_bowler),
                    "incoming_batter_samples": batters[incoming]["balls"],
                    "incoming_bowler_samples": bowlers[incoming]["balls"],
                }

        for innings_number, innings in enumerate(raw.get("innings", []), start=1):
            batting_team = str(innings.get("team", ""))
            bowling_team = next((team for team in teams if team != batting_team), "")
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                apply_replacement(deliveries[0], over_number)
                if scope == "ipl":
                    batting_impact = impact.get(batting_team, {})
                    bowling_impact = impact.get(bowling_team, {})
                    rows.append(
                        {
                            "source_file": path.name,
                            "match_date": match_date,
                            "innings": innings_number,
                            "over": over_number,
                            "ipl_adapter_batting_quality_delta": float(
                                batting_impact.get("batting_delta", 0.0)
                            ),
                            "ipl_adapter_bowling_quality_delta": float(
                                bowling_impact.get("bowling_delta", 0.0)
                            ),
                            "ipl_adapter_batting_sample": int(
                                batting_impact.get("incoming_batter_samples", 0)
                            ),
                            "ipl_adapter_bowling_sample": int(
                                bowling_impact.get("incoming_bowler_samples", 0)
                            ),
                            "ipl_adapter_batting_used": int(bool(batting_impact)),
                            "ipl_adapter_bowling_used": int(bool(bowling_impact)),
                            "ipl_adapter_batting_overs_since": (
                                over_number - int(batting_impact["over"])
                                if batting_impact
                                else -1
                            ),
                            "ipl_adapter_bowling_overs_since": (
                                over_number - int(bowling_impact["over"])
                                if bowling_impact
                                else -1
                            ),
                            "ipl_adapter_batting_team_win": int(
                                bool(winner) and batting_team == winner
                            ),
                            "ipl_adapter_result_known": int(bool(winner)),
                        }
                    )
                for delivery in deliveries:
                    if not _is_legal(delivery):
                        continue
                    batter = str(delivery["batter"])
                    bowler = str(delivery["bowler"])
                    batter_runs = int(delivery["runs"]["batter"])
                    total_runs = int(delivery["runs"]["total"])
                    wicket = int(_is_wicket(delivery))
                    staged.append(
                        (
                            batter,
                            bowler,
                            {
                                "balls": 1,
                                "runs": batter_runs,
                                "boundaries": int(batter_runs in {4, 6}),
                                "dismissals": wicket,
                            },
                            {
                                "balls": 1,
                                "runs": total_runs,
                                "dots": int(total_runs == 0),
                                "wickets": wicket,
                            },
                        )
                    )
                for delivery in deliveries[1:]:
                    apply_replacement(delivery, over_number)
        for batter, bowler, batter_event, bowler_event in staged:
            for profile in (batters[batter], global_batter):
                for name, value in batter_event.items():
                    profile[name] += value
            for profile in (bowlers[bowler], global_bowler):
                for name, value in bowler_event.items():
                    profile[name] += value
    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("IPL adapter dataset contains duplicate over keys.")
    return result.sort_values(
        ["match_date", "source_file", "innings", "over"]
    ).reset_index(drop=True)
