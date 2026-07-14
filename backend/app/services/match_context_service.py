"""
Match Context Service.

Responsible for constructing and enriching MatchContext objects.
"""

from __future__ import annotations

import logging
from typing import Any

from app.models.match_context import MatchContext
from app.models.pitch_context import PitchContext
from app.models.toss_context import TossContext
from app.models.weather_context import WeatherContext
from app.services.venue_service import VenueService
from app.utils.normalizer import Normalizer

logger = logging.getLogger(__name__)


class MatchContextService:
    """Service responsible for creating MatchContext instances."""

    @staticmethod
    def build(match_data: dict[str, Any]) -> MatchContext:
        """
        Build and enrich a MatchContext.

        Parameters
        ----------
        match_data : dict[str, Any]
            Raw match information.

        Returns
        -------
        MatchContext
            A validated and enriched MatchContext instance.
        """

        logger.info("Building MatchContext")

        venue_name = match_data.get("venue", "")

        weather_data = match_data.get("weather", {})

        weather = WeatherContext(
            temperature=weather_data.get("temperature", 0.0),
            humidity=weather_data.get("humidity", 0.0),
            wind_speed=weather_data.get("wind_speed", 0.0),
            rain_probability=weather_data.get("rain_probability", 0.0),
            cloud_cover=weather_data.get("cloud_cover", 0.0),
            dew_probability=weather_data.get("dew_probability", 0.0),
        )

        pitch = PitchContext()

        toss = TossContext(
            winner=match_data.get("toss_winner"),
            decision=match_data.get("toss_decision"),
        )

        context = MatchContext(
            match_id=match_data.get("match_id"),
            team1=Normalizer.normalize_team(match_data.get("team1", "")),
            team2=Normalizer.normalize_team(match_data.get("team2", "")),
            venue=venue_name,
            city=match_data.get("city", ""),
            country=match_data.get("country", ""),
            format=Normalizer.normalize_format(match_data.get("format", "")),
            toss=toss,
            batting_first=match_data.get("batting_first"),
            bowling_first=match_data.get("bowling_first"),
            date=match_data.get("date"),
            weather=weather,
            pitch=pitch,
            venue_stats=match_data.get("venue_stats", {}),
            team1_players=match_data.get("team1_players", []),
            team2_players=match_data.get("team2_players", []),
            metadata=match_data.get("metadata", {}),
        )

        if context.venue:
            logger.info("Enriching venue information")

            venue = VenueService.get_venue(context.venue)

            if not context.city:
                context.city = str(venue["city"])

            if not context.country:
                context.country = str(venue["country"])

            context.pitch.pitch_type = str(venue["pitch_type"])

            context.venue_stats = {
                "average_first_innings": venue["average_first_innings"],
                "average_second_innings": venue["average_second_innings"],
                "boundary_size": venue["boundary_size"],
                "dew": venue["dew"],
                "timezone": venue["timezone"],
            }

        context.validate()

        logger.info("MatchContext created successfully")

        return context
