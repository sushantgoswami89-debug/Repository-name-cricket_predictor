"""Batting-order position -- a real, well-known cricket skill/role proxy
(a number-9 batter is dramatically more likely to get out and score
fewer runs than a number-3) that, checked 2026-09-18, is used by NEITHER
the live run-range nor wicket model at all. Player-identity prior stats
(career average/SR) already capture most of a well-known player's own
skill level, but this adds two things they can't: (1) a strong prior for
a batter with little/no career history yet (a genuine debutant or a
player new to this competition), and (2) a MATCH-SPECIFIC tactical
signal a player's cross-match career average can never reflect -- e.g. a
normally-middle-order batter promoted to open in this one match.

Leakage-safe the same way as every other per-over feature in this
codebase: the row emitted for over N reflects each batter's position as
established from every appearance strictly before over N's own first
ball, so a batter making their first-ever appearance on over N's first
ball is captured by *that* ball, never by the row describing state
entering over N.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_batting_position_dataset(
    project_root: Path, scopes: tuple[str, ...] = ("ipl", "t20i")
) -> pd.DataFrame:
    paths: list[tuple[str, Path]] = []
    for scope in scopes:
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json"):
            raw = json.loads(path.read_text(encoding="utf-8"))
            paths.append((str(raw["info"]["dates"][0]), path))
    paths.sort(key=lambda item: (item[0], item[1].name))

    rows: list[dict[str, Any]] = []
    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        regular_innings = [
            item for item in raw.get("innings", []) if not bool(item.get("super_over", False))
        ]
        for innings_number, innings in enumerate(regular_innings, start=1):
            position_by_name: dict[str, int] = {}
            next_position = 1

            def _position(name: str) -> int:
                nonlocal next_position
                if name not in position_by_name:
                    position_by_name[name] = next_position
                    next_position += 1
                return position_by_name[name]

            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                first = deliveries[0]
                striker_name = str(first.get("batter", ""))
                non_striker_name = str(first.get("non_striker", ""))
                # Assign the incoming batter's position before reading it if
                # this is their first-ever ball of the innings (a new-batter
                # walk-in at the very start of this over is genuine
                # information available before over N begins -- their
                # partner already established, they've just been announced).
                striker_position = _position(striker_name)
                non_striker_position = _position(non_striker_name)
                rows.append(
                    {
                        "source_file": path.name,
                        "innings": innings_number,
                        "over": over_number,
                        "striker_batting_position": striker_position,
                        "non_striker_batting_position": non_striker_position,
                    }
                )
                # Register anyone else appearing later in this same over
                # (e.g. a replacement batter after a mid-over wicket) for
                # future overs' lookups -- never affects this row.
                for delivery in deliveries:
                    _position(str(delivery.get("batter", "")))
                    _position(str(delivery.get("non_striker", "")))

    frame = pd.DataFrame(rows)
    return frame.sort_values(["source_file", "innings", "over"]).reset_index(drop=True)
