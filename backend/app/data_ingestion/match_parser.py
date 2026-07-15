"""
CricketBaba Match Parser.

Production-ready parser for Cricsheet JSON v1.2.
"""

from __future__ import annotations

import logging
from typing import Any

from app.models.match_data import Delivery, Innings, Match, MatchInfo, Over

logger = logging.getLogger(__name__)


class MatchParser:
    """Parse Cricsheet JSON into Match objects."""

    def parse(self, data: dict[str, Any]) -> Match:
        """Parse a Cricsheet JSON dictionary."""
        if "info" not in data:
            raise ValueError("Missing 'info' section in match JSON.")

        info = self._parse_info(data["info"])

        innings = [
            self._parse_innings(innings_data)
            for innings_data in data.get("innings", [])
        ]

        logger.info("Parsed %d innings.", len(innings))

        return Match(
            info=info,
            innings=innings,
        )

    def _parse_info(self, info: dict[str, Any]) -> MatchInfo:
        return MatchInfo(
            match_type=info.get("match_type", "Unknown"),
            teams=info.get("teams", []),
            venue=info.get("venue"),
            city=info.get("city"),
            dates=[str(x) for x in info.get("dates", [])],
        )

    def _parse_innings(self, innings_data: dict[str, Any]) -> Innings:
        overs = [
            self._parse_over(over_data)
            for over_data in innings_data.get("overs", [])
        ]

        return Innings(
            team=innings_data.get("team", "Unknown"),
            overs=overs,
        )

    def _parse_over(self, over_data: dict[str, Any]) -> Over:
        deliveries = [
            self._parse_delivery(delivery)
            for delivery in over_data.get("deliveries", [])
        ]

        return Over(
            over_number=over_data.get("over", 0),
            deliveries=deliveries,
        )

    def _parse_delivery(self, delivery: dict[str, Any]) -> Delivery:
        return Delivery(
            ball=str(delivery.get("actual_delivery", "")),
            batter=delivery.get("batter", ""),
            bowler=delivery.get("bowler", ""),
            non_striker=delivery.get("non_striker", ""),
            runs=delivery.get("runs", {}),
            extras=delivery.get("extras", {}),
            wickets=delivery.get("wickets", []),
        )
