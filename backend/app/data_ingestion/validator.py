"""
CricketBaba Match Validator.

Validates parsed Match objects before persistence.
"""

from __future__ import annotations

import logging

from app.models.match_data import (
    Delivery,
    Innings,
    Match,
    Over,
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when match validation fails."""


class MatchValidator:
    """
    Validates parsed Match objects.
    """

    def validate(self, match: Match) -> Match:
        """
        Validate an entire match.

        Parameters
        ----------
        match
            Parsed match object.

        Returns
        -------
        Match
            Validated match.

        Raises
        ------
        ValidationError
            If validation fails.
        """

        logger.info("Validating match.")

        self._validate_match(match)

        for innings in match.innings:
            self._validate_innings(innings)

        logger.info("Match validation successful.")

        return match

    def _validate_match(self, match: Match) -> None:
        if not match.info.match_type:
            raise ValidationError("Missing match type.")

        if len(match.info.teams) != 2:
            raise ValidationError("A match must contain exactly two teams.")

        if not match.innings:
            raise ValidationError("Match contains no innings.")

    def _validate_innings(self, innings: Innings) -> None:
        if not innings.team:
            raise ValidationError("Innings has no batting team.")

        if not innings.overs:
            raise ValidationError(
                f"{innings.team} innings contains no overs."
            )

        for over in innings.overs:
            self._validate_over(over)

    def _validate_over(self, over: Over) -> None:
        if over.over_number < 0:
            raise ValidationError("Negative over number found.")

        if not over.deliveries:
            raise ValidationError(
                f"Over {over.over_number} contains no deliveries."
            )

        for delivery in over.deliveries:
            self._validate_delivery(delivery)

    def _validate_delivery(self, delivery: Delivery) -> None:
        if not delivery.ball:
            raise ValidationError("Delivery missing ball number.")

        if not delivery.batter:
            raise ValidationError("Delivery missing batter.")

        if not delivery.bowler:
            raise ValidationError("Delivery missing bowler.")

        if not delivery.non_striker:
            raise ValidationError(
                "Delivery missing non-striker."
            )

        if "total" not in delivery.runs:
            raise ValidationError(
                "Delivery missing total runs."
            )