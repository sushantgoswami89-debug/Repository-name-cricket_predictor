"""Leakage-safe IPL impact-player state at the start of each over."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS


def role_category(role: str) -> str:
    normalized = role.lower()
    if "allrounder" in normalized:
        return "allrounder"
    if "bowler" in normalized:
        return "bowler"
    if "batter" in normalized or "wicketkeeper" in normalized:
        return "batter"
    return "unknown"


def build_ipl_impact_dataset(project_root: Path) -> pd.DataFrame:
    roles = pd.read_csv(project_root / "data/candidates/v3.8/player_roles.csv").fillna(
        ""
    )
    role_by_id = {
        str(row.cricsheet_id): role_category(str(row.playing_role))
        for row in roles.itertuples()
    }
    paths: list[tuple[str, Path]] = []
    excluded = set(KNOWN_RULE_ANOMALY_EXCLUSIONS)
    for path in (project_root / "data/raw/cricsheet/ipl").glob("*.json"):
        if path.stem in excluded:
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        paths.append((str(raw["info"]["dates"][0]), path))
    paths.sort(key=lambda item: (item[0], item[1].name))

    rows: list[dict[str, Any]] = []
    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        year = int(match_date[:4])
        registry = raw["info"].get("registry", {}).get("people", {})
        teams = [str(team) for team in raw["info"].get("teams", [])]
        used: dict[str, dict[str, Any]] = {}

        def apply_replacements(
            delivery: dict[str, Any],
            over_number: int,
            used_state: dict[str, dict[str, Any]] = used,
            player_registry: dict[str, str] = registry,
        ) -> None:
            replacements = delivery.get("replacements", {}).get("match", [])
            for replacement in replacements:
                if replacement.get("reason") != "impact_player":
                    continue
                team = str(replacement.get("team", ""))
                incoming = str(replacement.get("in", ""))
                used_state[team] = {
                    "over": over_number,
                    "role": role_by_id.get(
                        str(player_registry.get(incoming, "")), "unknown"
                    ),
                }

        for innings_number, innings in enumerate(raw.get("innings", []), start=1):
            batting_team = str(innings.get("team", ""))
            bowling_team = next((team for team in teams if team != batting_team), "")
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                apply_replacements(deliveries[0], over_number)
                batting_state = used.get(batting_team)
                bowling_state = used.get(bowling_team)
                rows.append(
                    {
                        "source_file": path.name,
                        "match_date": match_date,
                        "innings": innings_number,
                        "over": over_number,
                        "ipl_impact_rule_era": int(year >= 2023),
                        "ipl_impact_batting_used": int(batting_state is not None),
                        "ipl_impact_bowling_used": int(bowling_state is not None),
                        "ipl_impact_batting_available": int(
                            year >= 2023 and batting_state is None
                        ),
                        "ipl_impact_bowling_available": int(
                            year >= 2023 and bowling_state is None
                        ),
                        "ipl_impact_batting_role": (
                            batting_state["role"] if batting_state else "none"
                        ),
                        "ipl_impact_bowling_role": (
                            bowling_state["role"] if bowling_state else "none"
                        ),
                        "ipl_impact_batting_overs_since": (
                            over_number - int(batting_state["over"])
                            if batting_state
                            else -1
                        ),
                        "ipl_impact_bowling_overs_since": (
                            over_number - int(bowling_state["over"])
                            if bowling_state
                            else -1
                        ),
                    }
                )
                for delivery in deliveries[1:]:
                    apply_replacements(delivery, over_number)
    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("IPL impact-player dataset contains duplicate over keys.")
    return result.sort_values(
        ["match_date", "source_file", "innings", "over"]
    ).reset_index(drop=True)
