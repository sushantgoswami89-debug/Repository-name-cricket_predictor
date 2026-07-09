"""
Match Context Model.

This module defines the MatchContext dataclass used throughout
the CricketBaba prediction engine.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from app.models.pitch_context import PitchContext
from app.models.weather_context import WeatherContext

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MatchContext:
    """
    Represents all contextual information required for a match prediction.
    """

    match_id: str | None = None

    team1: str = ""
    team2: str = ""

    venue: str = ""
    city: str = ""
    country: str = ""

    format: str = ""

    toss_winner: str | None = None
    toss_decision: str | None = None

    batting_first: str | None = None
    bowling_first: str | None = None

    date: str | None = None

    weather: WeatherContext = field(default_factory=WeatherContext)

    pitch: PitchContext = field(default_factory=PitchContext)

    venue_stats: dict[str, Any] = field(default_factory=dict)

    team1_players: list[str] = field(default_factory=list)
    team2_players: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """
        Validate MatchContext.
        """

        logger.info("Validating MatchContext")

        if not self.team1:
            raise ValueError("team1 cannot be empty.")

        if not self.team2:
            raise ValueError("team2 cannot be empty.")

        if self.team1 == self.team2:
            raise ValueError("Both teams cannot be the same.")

        if not self.venue:
            raise ValueError("venue cannot be empty.")

        self.weather.validate()
        self.pitch.validate()

        logger.info("Validation successful")

        return True

    def to_dict(self) -> dict[str, Any]:
        """
        Convert MatchContext into a dictionary.
        """

        logger.info("Converting MatchContext to dictionary")

        data = asdict(self)

        data["weather"] = self.weather.to_dict()
        data["pitch"] = self.pitch.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchContext":
        """
        Create MatchContext from a dictionary.
        """

        logger.info("Creating MatchContext from dictionary")

        payload = dict(data)

        payload["weather"] = WeatherContext(**payload.get("weather", {}))

        payload["pitch"] = PitchContext(**payload.get("pitch", {}))

        return cls(**payload)

    def summary(self) -> str:
        """
        Return a human-readable match summary.
        """

        logger.info("Generating match summary")

        return (
            f"{self.team1} vs {self.team2} | "
            f"{self.format} | "
            f"{self.venue}, {self.city}"
        )
