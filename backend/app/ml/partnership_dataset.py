"""Leakage-safe current-partnership (stand) scoring rate.

User-flagged direction (2026-08-23): "partnership builds." The codebase
already tracks `partnership_legal_ball_age` (balls faced collectively by
the current striker+non_striker, from `ipl_phase_moe_dataset.py`) but
never the actual RATE that pair is scoring at -- only how long they've
been together, not how well. This adds the real cricket "partnership"
stat: team runs and legal balls since the fall of the last wicket (a
genuine stand, resetting exactly at each dismissal), which is a cleaner
definition than `partnership_legal_ball_age` -- that feature sums each
individual batter's own total balls faced this innings, which overstates
stand length whenever the longer-set batter has already been through an
earlier partner this innings. This tracks the stand itself, not the two
individuals' separate innings-long ball counts.

Leakage-safe the same way as every other per-over feature in this
codebase: the row emitted for over N reflects state as of the start of
over N (before its first ball), so a wicket that fell during over N-1 is
already reflected, and nothing from over N itself leaks in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_partnership_dataset(
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
            stand_runs = 0
            stand_balls = 0
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                rows.append(
                    {
                        "source_file": path.name,
                        "innings": innings_number,
                        "over": over_number,
                        "partnership_runs": stand_runs,
                        "partnership_balls": stand_balls,
                        "partnership_run_rate": (
                            6.0 * stand_runs / stand_balls if stand_balls else 0.0
                        ),
                    }
                )
                for delivery in deliveries:
                    extras = delivery.get("extras", {})
                    is_legal = "wides" not in extras and "noballs" not in extras
                    total_runs = int(delivery.get("runs", {}).get("total", 0))
                    stand_runs += total_runs
                    if is_legal:
                        stand_balls += 1
                    if delivery.get("wickets"):
                        stand_runs = 0
                        stand_balls = 0

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Partnership dataset contains duplicate over keys.")
    return result


def build_partnership_trend_dataset(
    project_root: Path, scopes: tuple[str, ...] = ("ipl", "t20i"), window: int = 12
) -> pd.DataFrame:
    """Is the CURRENT stand accelerating or decelerating relative to its
    own history, scoped strictly to this partnership (not the whole
    team's last-N-ball window, which can span across a wicket and mix in
    a different, now-dismissed pair's contribution)?

    Motivated by a direct question (2026-08-23): match_winner_v1 was
    found to be a confirming indicator, not a leading one -- it never
    gets ahead of the scoreboard (see docs/finding_win_probability_chase_bug.md's
    follow-up reversal-lead-time analysis). The existing partnership_run_rate
    is itself only a flat average over the whole stand so far, with no
    sense of whether it's currently speeding up or slowing down -- this
    tests whether that trend, scoped to just the current pair, gives the
    model something to react to before the cumulative state fully shows
    it.

    `partnership_recent_run_rate`: run rate of the last `window` legal
    balls faced by the CURRENT partnership only (bounded at the last
    wicket -- resets to 0 immediately after one, same as partnership_runs).
    `partnership_trend`: recent minus overall stand rate. Positive = this
    pair is currently scoring faster than their own stand's average so
    far; negative = slowing down.
    """
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
            stand_runs = 0
            stand_balls = 0
            recent_runs: list[int] = []  # runs on each of the last `window` legal balls, this stand only
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                overall_rate = 6.0 * stand_runs / stand_balls if stand_balls else 0.0
                recent_balls = len(recent_runs)
                recent_rate = 6.0 * sum(recent_runs) / recent_balls if recent_balls else 0.0
                rows.append(
                    {
                        "source_file": path.name,
                        "innings": innings_number,
                        "over": over_number,
                        "partnership_recent_run_rate": recent_rate,
                        "partnership_trend": recent_rate - overall_rate,
                    }
                )
                for delivery in deliveries:
                    extras = delivery.get("extras", {})
                    is_legal = "wides" not in extras and "noballs" not in extras
                    total_runs = int(delivery.get("runs", {}).get("total", 0))
                    stand_runs += total_runs
                    if is_legal:
                        stand_balls += 1
                        recent_runs.append(total_runs)
                        if len(recent_runs) > window:
                            recent_runs.pop(0)
                    if delivery.get("wickets"):
                        stand_runs = 0
                        stand_balls = 0
                        recent_runs = []

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Partnership trend dataset contains duplicate over keys.")
    return result
