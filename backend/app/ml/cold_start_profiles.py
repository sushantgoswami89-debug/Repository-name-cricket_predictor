"""Governed temporary profiles for players without sufficient IPL history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


IMPORTANT_RUNS = 50
IMPORTANT_CAMEO_RUNS = 30
IMPORTANT_CAMEO_STRIKE_RATE = 175.0
REGULAR_STARTS = 5
ESTABLISHED_SEASONS = 2


@dataclass(frozen=True)
class PromotionEvidence:
    seasons: int
    max_consecutive_starts: int
    first_five_innings: tuple[tuple[int, int], ...]  # (runs, balls)
    external_source_verified: bool = False

    @property
    def important_first_five(self) -> bool:
        for runs, balls in self.first_five_innings[:5]:
            strike_rate = 100 * runs / balls if balls else 0.0
            if runs >= IMPORTANT_RUNS or (
                runs >= IMPORTANT_CAMEO_RUNS
                and strike_rate >= IMPORTANT_CAMEO_STRIKE_RATE
            ):
                return True
        return False

    @property
    def eligible(self) -> bool:
        return (
            self.seasons >= ESTABLISHED_SEASONS
            or self.max_consecutive_starts >= REGULAR_STARTS
            or self.important_first_five
        )

    @property
    def status(self) -> str:
        if self.eligible and self.external_source_verified:
            return "MAIN"
        if self.eligible:
            return "ELIGIBLE_REVIEW"
        return "TEMP"


BATTER_ROLES = {
    "BATTER",
    "TOP_ORDER_BATTER",
    "MIDDLE_ORDER_BATTER",
    "WICKETKEEPER_BATTER",
    "ALL_ROUNDER",
}


@dataclass(frozen=True)
class TemporaryProfile:
    """A point-in-time external prior with auditable provenance."""

    player_id: str
    player_name: str
    role: str
    usual_position: str
    batting_average: float | None
    strike_rate: float | None
    experience: tuple[str, ...]
    source_url: str
    source_title: str
    source_published: date
    verified_at: date

    @property
    def is_batter(self) -> bool:
        return self.role in BATTER_ROLES

    def available_on(self, match_date: date) -> bool:
        return self.source_published < match_date and self.verified_at < match_date

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "TemporaryProfile":
        return cls(
            player_id=str(row["player_id"]),
            player_name=str(row["player_name"]),
            role=str(row["role"]),
            usual_position=str(row.get("usual_position", "UNKNOWN")),
            batting_average=(
                float(row["batting_average"])
                if row.get("batting_average") is not None
                else None
            ),
            strike_rate=(
                float(row["strike_rate"])
                if row.get("strike_rate") is not None
                else None
            ),
            experience=tuple(str(item) for item in row.get("experience", [])),
            source_url=str(row["source_url"]),
            source_title=str(row["source_title"]),
            source_published=date.fromisoformat(str(row["source_published"])),
            verified_at=date.fromisoformat(str(row["verified_at"])),
        )


def route_profile(
    *,
    role: str,
    evidence: PromotionEvidence,
    temporary_profile: TemporaryProfile | None,
    match_date: date,
) -> str:
    """Choose a profile without allowing a batter prior onto a specialist bowler."""
    if role == "BOWLER":
        return "BOWLER_SAFEGUARD"
    if evidence.status == "MAIN":
        return "MAIN"
    if (
        temporary_profile is not None
        and temporary_profile.is_batter
        and temporary_profile.available_on(match_date)
    ):
        return "TEMP"
    return "NEUTRAL"


def aggressiveness_band(strike_rate: float | None) -> str:
    """Return a transparent T20 batting-style prior from a sourced strike rate."""
    if strike_rate is None:
        return "UNKNOWN"
    if strike_rate >= 160:
        return "VERY_AGGRESSIVE"
    if strike_rate >= 140:
        return "AGGRESSIVE"
    if strike_rate >= 120:
        return "BALANCED"
    return "ANCHOR"
