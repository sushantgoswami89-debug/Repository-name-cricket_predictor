"""Explicit routing boundaries for CricketBaba limited-over engines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EngineFamily(StrEnum):
    T20 = "t20"
    IPL = "ipl"
    ODI = "odi"


class UnsupportedCricketFormat(ValueError):
    """Raised when a match is outside CricketBaba's limited-over scope."""


@dataclass(frozen=True, slots=True)
class EngineSpecification:
    family: EngineFamily
    innings_overs: int
    powerplay_overs: int
    death_starts: int
    base_features: tuple[str, ...]
    extension_features: tuple[str, ...] = ()
    separately_trained_model_required: bool = True


T20_BASE_FEATURES = (
    "score_state",
    "phase",
    "recent_form",
    "player_profiles",
    "bowler_type",
    "venue_track",
    "boundary_risk",
    "commentary_signals",
)


class EngineRouter:
    """Choose one engine without silently applying a model across formats."""

    SPECIFICATIONS = {
        EngineFamily.T20: EngineSpecification(
            family=EngineFamily.T20,
            innings_overs=20,
            powerplay_overs=6,
            death_starts=16,
            base_features=T20_BASE_FEATURES,
        ),
        EngineFamily.IPL: EngineSpecification(
            family=EngineFamily.IPL,
            innings_overs=20,
            powerplay_overs=6,
            death_starts=16,
            base_features=T20_BASE_FEATURES,
            # impact_player_state was implemented and tested against both
            # live models (run-range and wicket) on real 2025+ holdout data
            # and found to add no signal to either -- see
            # docs/finding_ipl_impact_player_no_signal.md. Removed from
            # this tuple rather than left as a stale placeholder.
            extension_features=(
                "strategic_timeout_state",
                "ipl_team_strategy",
            ),
        ),
        EngineFamily.ODI: EngineSpecification(
            family=EngineFamily.ODI,
            innings_overs=50,
            powerplay_overs=10,
            death_starts=41,
            base_features=(
                "score_state",
                "odi_phase",
                "recent_form",
                "player_profiles",
                "bowler_type",
                "venue_track",
                "old_ball_state",
                "commentary_signals",
            ),
        ),
    }

    @classmethod
    def resolve(cls, match_format: str, competition: str = "") -> EngineSpecification:
        normalized_format = match_format.strip().upper()
        normalized_competition = competition.strip().upper()
        if normalized_format in {"TEST", "TEST MATCH"}:
            raise UnsupportedCricketFormat(
                "Test cricket is intentionally outside CricketBaba's scope."
            )
        if normalized_format == "ODI":
            return cls.SPECIFICATIONS[EngineFamily.ODI]
        if normalized_format == "IPL" or normalized_competition in {
            "IPL",
            "INDIAN PREMIER LEAGUE",
        }:
            return cls.SPECIFICATIONS[EngineFamily.IPL]
        if normalized_format in {"T20", "T20I", "IT20", "TWENTY20"}:
            return cls.SPECIFICATIONS[EngineFamily.T20]
        raise UnsupportedCricketFormat(
            f"Unsupported cricket format: {match_format or 'unknown'}."
        )
