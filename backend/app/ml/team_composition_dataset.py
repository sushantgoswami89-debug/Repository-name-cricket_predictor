"""Leakage-safe team role composition: batting-side all-rounder depth,
bowling-side spin/pace balance.

User-requested direction (2026-08-23): "team combinations opener, middle,
hitter, allrounder, spinners and pacers combination." The confirmed
playing XI (`info.players` in Cricsheet, `ToiSnapshot.team_players` live)
is known before a ball is bowled -- genuinely match-level, constant for
both innings, same timing as toss. `data/external/cricsheet_player_styles.csv`
already carries `playing_role` (Bowler/Allrounder/Batter/Wicketkeeper
variants) and `bowling_style` (pace/spin) per player, already loaded by
`WicketContract22FeatureComputer` for individual bowler-type
classification -- this aggregates the SAME source to the team level for
the first time: how many all-rounders does this batting side have, and
does the bowling side lean pace-heavy or spin-heavy.

~80% role/bowling-style coverage among players who actually appeared in
IPL 2024-2025 squads (verified before building this) -- missing players
degrade to "unknown"/excluded from the count rather than crashing or
skewing the ratio artificially.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.ipl_impact_dataset import role_category

UNKNOWN_CATEGORY = "__UNKNOWN__"


def _classify_bowling_style(style: Any) -> str:
    if not isinstance(style, str):
        return UNKNOWN_CATEGORY
    lowered = style.lower()
    if "spin" in lowered or "break" in lowered or "orthodox" in lowered or "chinaman" in lowered:
        return "spin"
    if "fast" in lowered or "medium" in lowered or "pace" in lowered:
        return "pace"
    return UNKNOWN_CATEGORY


def _team_composition(
    players: list[str],
    registry: dict[str, str],
    role_by_id: dict[str, str],
    bowling_style_by_id: dict[str, str],
) -> dict[str, int]:
    pace = spin = allrounders = specialist_batters = specialist_bowlers = 0
    for name in players:
        cricsheet_id = registry.get(name, "")
        raw_role = role_by_id.get(cricsheet_id, "")
        role = role_category(raw_role if isinstance(raw_role, str) else "")
        if role == "allrounder":
            allrounders += 1
        elif role == "batter":
            specialist_batters += 1
        elif role == "bowler":
            specialist_bowlers += 1
        # bowling_style is a nominal database attribute recorded for every
        # player (batting_style's sibling column) -- a specialist batter or
        # wicketkeeper who never actually bowls (e.g. MS Dhoni, "Right arm
        # Medium") still has one on file. Only count it for players whose
        # ROLE says they're actually part of the bowling attack, verified
        # against a real CSK squad before fixing this (Rahane/Dhoni/Gaikwad
        # all had nominal styles despite never bowling in IPL).
        if role in ("bowler", "allrounder"):
            style = _classify_bowling_style(bowling_style_by_id.get(cricsheet_id))
            if style == "pace":
                pace += 1
            elif style == "spin":
                spin += 1
    return {
        "pace": pace, "spin": spin, "allrounders": allrounders,
        "specialist_batters": specialist_batters, "specialist_bowlers": specialist_bowlers,
    }


def build_team_composition_dataset(
    project_root: Path, scopes: tuple[str, ...] = ("ipl", "t20i")
) -> pd.DataFrame:
    styles_path = project_root / "data/external/cricsheet_player_styles.csv"
    styles = pd.read_csv(styles_path)
    role_by_id = dict(zip(styles["cricsheet_id"], styles["playing_role"]))
    bowling_style_by_id = dict(zip(styles["cricsheet_id"], styles["bowling_style"]))

    paths: list[tuple[str, Path]] = []
    for scope in scopes:
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json"):
            raw = json.loads(path.read_text(encoding="utf-8"))
            paths.append((str(raw["info"]["dates"][0]), path))
    paths.sort(key=lambda item: (item[0], item[1].name))

    rows: list[dict[str, Any]] = []
    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        info = raw["info"]
        registry = info.get("registry", {}).get("people", {})
        squads = info.get("players", {})
        if len(squads) != 2:
            continue
        team_composition = {
            team: _team_composition(players, registry, role_by_id, bowling_style_by_id)
            for team, players in squads.items()
        }

        regular_innings = [
            item for item in raw.get("innings", []) if not bool(item.get("super_over", False))
        ]
        for innings_number, innings in enumerate(regular_innings, start=1):
            batting_team = str(innings.get("team", ""))
            bowling_team = next((team for team in squads if team != batting_team), "")
            batting_comp = team_composition.get(batting_team, {})
            bowling_comp = team_composition.get(bowling_team, {})
            for source_over in innings.get("overs", []):
                if not source_over.get("deliveries"):
                    continue
                rows.append(
                    {
                        "source_file": path.name,
                        "innings": innings_number,
                        "over": int(source_over["over"]) + 1,
                        "batting_team_allrounder_count": batting_comp.get("allrounders", 0),
                        "batting_team_specialist_batter_count": batting_comp.get("specialist_batters", 0),
                        "bowling_team_pace_count": bowling_comp.get("pace", 0),
                        "bowling_team_spin_count": bowling_comp.get("spin", 0),
                        "bowling_team_allrounder_count": bowling_comp.get("allrounders", 0),
                    }
                )

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Team composition dataset contains duplicate over keys.")
    return result
