"""Leakage-safe match-winner label: did the team currently batting in
this innings go on to win the match?

User-requested (2026-08-23): a live win-probability model, as a third
prediction alongside run-range and wicket (not a replacement). The label
is a genuinely different target from anything else in this codebase --
runs-in-over and wicket-in-over are both about the next six balls; this
is about the eventual match result, known only at full-time, joined back
onto every over of the match the same way a final score is.

Not leakage in the usual sense (nothing here lets a training row see
information from later in ITS OWN match state before the row's own
over -- the label is exactly what we want to predict, same relationship
runs_in_over/wicket_in_over already have to their own targets). Matches
with no resolvable winner (rain-affected "no result", unresolved "tie")
are excluded -- 186 of 6,767 matches (2.7%) across IPL+T20I, verified
directly against Cricsheet's own outcome field before building this.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_match_winner_dataset(
    project_root: Path, scopes: tuple[str, ...] = ("ipl", "t20i")
) -> pd.DataFrame:
    paths: list[tuple[str, Path]] = []
    for scope in scopes:
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json"):
            raw = json.loads(path.read_text(encoding="utf-8"))
            paths.append((str(raw["info"]["dates"][0]), path))
    paths.sort(key=lambda item: (item[0], item[1].name))

    rows: list[dict[str, Any]] = []
    excluded_no_winner = 0
    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        info = raw["info"]
        winner = info.get("outcome", {}).get("winner")
        if not winner:
            excluded_no_winner += 1
            continue

        regular_innings = [
            item for item in raw.get("innings", []) if not bool(item.get("super_over", False))
        ]
        for innings_number, innings in enumerate(regular_innings, start=1):
            batting_team = str(innings.get("team", ""))
            batting_team_won = int(batting_team == winner)
            for source_over in innings.get("overs", []):
                if not source_over.get("deliveries"):
                    continue
                rows.append(
                    {
                        "source_file": path.name,
                        "innings": innings_number,
                        "over": int(source_over["over"]) + 1,
                        "batting_team_won": batting_team_won,
                    }
                )

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Match winner dataset contains duplicate over keys.")
    return result
