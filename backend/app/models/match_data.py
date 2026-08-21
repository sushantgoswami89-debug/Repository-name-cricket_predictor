"""
CricketBaba Match Data Models.

Defines the internal data structures used after parsing
Cricsheet JSON files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
@dataclass(slots=True)
class Delivery:
    """
    Represents a single ball in a cricket match.
    """

    ball: str
    batter: str
    bowler: str
    non_striker: str
    runs: dict[str, int]
    extras: dict[str, int] = field(default_factory=dict)
    wickets: list[dict[str, Any]] = field(default_factory=list)
@dataclass(slots=True)
class Over:
    """
    Represents one over in an innings.
    """

    over_number: int
    deliveries: list[Delivery] = field(default_factory=list)


@dataclass(slots=True)
class Innings:
    """
    Represents one innings.
    """

    team: str
    overs: list[Over] = field(default_factory=list)
    super_over: bool = False


@dataclass(slots=True)
class MatchInfo:
    """
    Basic metadata about a match.
    """

    match_type: str
    teams: list[str]
    venue: str | None = None
    city: str | None = None
    dates: list[str] = field(default_factory=list)
    event_name: str | None = None
    match_type_number: int | None = None
    team_type: str | None = None


@dataclass(slots=True)
class Match:
    """
    Complete parsed match.
    """

    info: MatchInfo
    innings: list[Innings] = field(default_factory=list)
