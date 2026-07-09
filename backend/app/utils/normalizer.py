"""
Normalization utilities for CricketBaba.

This module provides helper methods for normalizing
team names, match formats and generic text values.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Normalizer:
    """Utility class for data normalization."""

    TEAM_NAMES = {
        "india": "India",
        "ind": "India",
        "team india": "India",
        "australia": "Australia",
        "aus": "Australia",
        "team australia": "Australia",
        "england": "England",
        "eng": "England",
        "new zealand": "New Zealand",
        "nz": "New Zealand",
        "pakistan": "Pakistan",
        "pak": "Pakistan",
        "south africa": "South Africa",
        "sa": "South Africa",
    }

    MATCH_FORMATS = {
        "t20": "T20",
        "it20": "T20",
        "twenty20": "T20",
        "odi": "ODI",
        "test": "Test",
        "test match": "Test",
    }

    @staticmethod
    def normalize_text(value: str) -> str:
        """
        Normalize generic text.

        Removes extra spaces and converts to lowercase.
        """

        if not value:
            return ""

        return " ".join(value.strip().split()).lower()

    @classmethod
    def normalize_team(cls, team: str) -> str:
        """
        Normalize a team name.
        """

        normalized = cls.normalize_text(team)

        result = cls.TEAM_NAMES.get(normalized, team.strip())

        logger.info("Normalized team '%s' -> '%s'", team, result)

        return result

    @classmethod
    def normalize_format(cls, match_format: str) -> str:
        """
        Normalize match format.
        """

        normalized = cls.normalize_text(match_format)

        result = cls.MATCH_FORMATS.get(normalized, match_format.strip())

        logger.info(
            "Normalized format '%s' -> '%s'",
            match_format,
            result,
        )

        return result
