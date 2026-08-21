"""Parse Cricsheet JSON match files into CricketBaba domain objects.

The parser deliberately performs structural validation only.  Business
validation is handled by :class:`app.data_ingestion.validator.MatchValidator`
in the next ingestion step.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from app.models.match_data import Delivery, Innings, Match, MatchInfo, Over

logger = logging.getLogger(__name__)


class MatchParseError(ValueError):
    """Raised when input is not a structurally valid Cricsheet match."""


class MatchParser:
    """Convert Cricsheet JSON v1.x dictionaries into internal match objects."""

    def parse(self, data: Mapping[str, Any]) -> Match:
        """Parse one decoded Cricsheet match document.

        Parameters
        ----------
        data
            The dictionary returned by :class:`JsonLoader`.

        Returns
        -------
        Match
            A match ready to be passed to ``MatchValidator``.

        Raises
        ------
        MatchParseError
            If required Cricsheet sections have an invalid structure.
        """

        if not isinstance(data, Mapping):
            raise MatchParseError("Match JSON must be an object.")

        info_data = self._mapping(data.get("info"), "info")
        innings_data = self._sequence(data.get("innings", []), "innings")

        match = Match(
            info=self._parse_info(info_data),
            innings=[
                self._parse_innings(item, index)
                for index, item in enumerate(innings_data)
            ],
        )

        logger.info(
            "Parsed Cricsheet match: type=%s, teams=%s, innings=%d",
            match.info.match_type,
            ", ".join(match.info.teams),
            len(match.innings),
        )
        return match

    def _parse_info(self, info: Mapping[str, Any]) -> MatchInfo:
        teams = self._string_list(info.get("teams", []), "info.teams")
        dates = [
            str(date) for date in self._sequence(info.get("dates", []), "info.dates")
        ]

        event = info.get("event", {})
        if event is None:
            event = {}
        event_data = self._mapping(event, "info.event")

        return MatchInfo(
            match_type=self._string_value(
                info.get("match_type", "Unknown"), "info.match_type"
            ),
            teams=teams,
            venue=self._optional_string(info.get("venue"), "info.venue"),
            city=self._optional_string(info.get("city"), "info.city"),
            dates=dates,
            event_name=self._optional_string(
                event_data.get("name"), "info.event.name"
            ),
            match_type_number=self._optional_integer(
                info.get("match_type_number"), "info.match_type_number"
            ),
            team_type=self._optional_string(info.get("team_type"), "info.team_type"),
        )

    def _parse_innings(self, data: Any, index: int) -> Innings:
        innings = self._mapping(data, f"innings[{index}]")
        overs_data = self._sequence(innings.get("overs", []), f"innings[{index}].overs")

        return Innings(
            team=self._string_value(
                innings.get("team", "Unknown"), f"innings[{index}].team"
            ),
            overs=[
                self._parse_over(over, index, over_index)
                for over_index, over in enumerate(overs_data)
            ],
            super_over=self._boolean_value(
                innings.get("super_over", False),
                f"innings[{index}].super_over",
            ),
        )

    def _parse_over(self, data: Any, innings_index: int, index: int) -> Over:
        location = f"innings[{innings_index}].overs[{index}]"
        over = self._mapping(data, location)
        over_number = over.get("over", 0)
        if isinstance(over_number, bool) or not isinstance(over_number, int):
            raise MatchParseError(f"{location}.over must be an integer.")

        deliveries_data = self._sequence(
            over.get("deliveries", []), f"{location}.deliveries"
        )
        return Over(
            over_number=over_number,
            deliveries=[
                self._parse_delivery(delivery, over_number, delivery_index, location)
                for delivery_index, delivery in enumerate(deliveries_data)
            ],
        )

    @staticmethod
    def _boolean_value(value: Any, location: str) -> bool:
        if not isinstance(value, bool):
            raise MatchParseError(f"{location} must be a boolean.")
        return value

    @staticmethod
    def _optional_integer(value: Any, location: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise MatchParseError(f"{location} must be an integer or null.")
        return value

    def _parse_delivery(
        self,
        data: Any,
        over_number: int,
        index: int,
        over_location: str,
    ) -> Delivery:
        location = f"{over_location}.deliveries[{index}]"
        delivery = self._mapping(data, location)
        runs = self._int_mapping(
            delivery.get("runs", {}),
            f"{location}.runs",
            ignored_boolean_keys=frozenset({"non_boundary"}),
        )
        extras = self._int_mapping(delivery.get("extras", {}), f"{location}.extras")
        wickets = self._mapping_list(delivery.get("wickets", []), f"{location}.wickets")

        # ``actual_delivery`` is supplied by CricketBaba's archived data.  Raw
        # Cricsheet files do not always include it, so retain a stable display
        # identifier based on the delivery's position as a safe fallback.
        ball = delivery.get("actual_delivery")
        if ball is None or ball == "":
            ball = f"{over_number}.{index + 1}"

        return Delivery(
            ball=self._string_value(ball, f"{location}.actual_delivery"),
            batter=self._string_value(delivery.get("batter", ""), f"{location}.batter"),
            bowler=self._string_value(delivery.get("bowler", ""), f"{location}.bowler"),
            non_striker=self._string_value(
                delivery.get("non_striker", ""), f"{location}.non_striker"
            ),
            runs=runs,
            extras=extras,
            wickets=wickets,
        )

    @staticmethod
    def _mapping(value: Any, location: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise MatchParseError(f"{location} must be an object.")
        return value

    @staticmethod
    def _sequence(value: Any, location: str) -> Sequence[Any]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise MatchParseError(f"{location} must be an array.")
        return value

    def _string_list(self, value: Any, location: str) -> list[str]:
        return [
            self._string_value(item, f"{location}[{index}]")
            for index, item in enumerate(self._sequence(value, location))
        ]

    @staticmethod
    def _string_value(value: Any, location: str) -> str:
        if not isinstance(value, str):
            raise MatchParseError(f"{location} must be a string.")
        return value

    def _optional_string(self, value: Any, location: str) -> str | None:
        if value is None:
            return None
        return self._string_value(value, location)

    def _int_mapping(
        self,
        value: Any,
        location: str,
        ignored_boolean_keys: frozenset[str] = frozenset(),
    ) -> dict[str, int]:
        mapping = self._mapping(value, location)
        result: dict[str, int] = {}
        for key, amount in mapping.items():
            if key in ignored_boolean_keys and isinstance(amount, bool):
                continue
            if (
                not isinstance(key, str)
                or isinstance(amount, bool)
                or not isinstance(amount, int)
            ):
                raise MatchParseError(f"{location} must map strings to integers.")
            result[key] = amount
        return result

    def _mapping_list(self, value: Any, location: str) -> list[dict[str, Any]]:
        return [
            dict(self._mapping(item, f"{location}[{index}]"))
            for index, item in enumerate(self._sequence(value, location))
        ]
