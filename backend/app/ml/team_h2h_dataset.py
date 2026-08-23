"""Leakage-safe team-vs-team head-to-head history.

User-requested direction (2026-08-23): "search for more." Web research
(cricket analytics literature on IPL prediction) specifically flags this
as more predictive in IPL than most competitions: "IPL's consistency of
venue -- each team plays at the same home ground every season -- and the
relative stability of squad identity across multiple seasons means that
head-to-head records carry more predictive value in IPL than in most
other cricket competitions." Distinct from every H2H feature already in
this codebase, which is player-vs-player (batter vs bowler) -- this is
franchise-vs-franchise, something never tested here.

Match-level (known before ball 1, constant for the whole match), same
timing as toss/team composition. Shrunk toward the pooled global rate
the same way every other sparse-cell feature in this codebase is
(H2H_SHRINKAGE_MATCHES "trust" in the prior), since most team pairings
have played only a handful of times.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

H2H_SHRINKAGE_MATCHES = 6.0


def _pair_key(team_a: str, team_b: str) -> str:
    return "|".join(sorted((team_a, team_b)))


def build_team_h2h_dataset(
    project_root: Path, scopes: tuple[str, ...] = ("ipl",)
) -> pd.DataFrame:
    """IPL-only by default: franchise identity (and therefore a
    meaningful head-to-head pairing) doesn't carry over between T20I
    national sides the way it does for IPL franchises -- a T20I pairing
    like India-vs-Australia recurs across formats/eras in a way that
    isn't comparable to the IPL's fixed-franchise, fixed-season
    structure this feature is actually about.
    """
    paths: list[tuple[str, Path]] = []
    for scope in scopes:
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json"):
            raw = json.loads(path.read_text(encoding="utf-8"))
            paths.append((str(raw["info"]["dates"][0]), path))
    paths.sort(key=lambda item: (item[0], item[1].name))

    # Neutral prior for a pairing with no history yet: 0.5 (each team
    # equally likely against an unknown opponent) -- not some other
    # global rate, since the quantity being shrunk is "does THIS team
    # tend to beat THIS specific opponent," which has no natural
    # population-level analog to fall back on other than even odds.
    NEUTRAL_PRIOR = 0.5

    pair_matches: dict[str, int] = defaultdict(int)
    pair_wins: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    rows: list[dict[str, Any]] = []
    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        info = raw["info"]
        teams = list(info.get("teams", []))
        if len(teams) != 2:
            continue
        key = _pair_key(teams[0], teams[1])
        prior_matches = pair_matches[key]

        regular_innings = [
            item for item in raw.get("innings", []) if not bool(item.get("super_over", False))
        ]
        for innings_number, innings in enumerate(regular_innings, start=1):
            batting_team = str(innings.get("team", ""))
            raw_win_rate = (
                pair_wins[key][batting_team] / prior_matches if prior_matches else NEUTRAL_PRIOR
            )
            shrunk_win_rate = (
                prior_matches * raw_win_rate + H2H_SHRINKAGE_MATCHES * NEUTRAL_PRIOR
            ) / (prior_matches + H2H_SHRINKAGE_MATCHES)
            for source_over in innings.get("overs", []):
                if not source_over.get("deliveries"):
                    continue
                rows.append(
                    {
                        "source_file": path.name,
                        "innings": innings_number,
                        "over": int(source_over["over"]) + 1,
                        "h2h_matches_played": prior_matches,
                        "h2h_batting_team_win_rate_shrunk": shrunk_win_rate,
                    }
                )

        outcome = info.get("outcome", {})
        winner = str(outcome.get("winner", ""))
        pair_matches[key] += 1
        if winner:
            pair_wins[key][winner] += 1

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Team H2H dataset contains duplicate over keys.")
    return result
