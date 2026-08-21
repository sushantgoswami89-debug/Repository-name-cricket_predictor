"""Canonical, rule-aware state for one live cricket innings.

This model is deliberately independent from prediction code.  A prediction may
only be requested from a confirmed :class:`MatchState`; it must never invent
scorecard, player, or bowling information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class MatchFormat(StrEnum):
    """Supported limited-over formats."""

    IPL = "IPL"
    T20I = "T20I"
    ODI = "ODI"


@dataclass(frozen=True, slots=True)
class FormatRules:
    """The core innings and bowling constraints for a match format."""

    innings_overs: int
    powerplay_overs: int
    maximum_overs_per_bowler: int

    @classmethod
    def for_format(cls, match_format: MatchFormat) -> "FormatRules":
        if match_format is MatchFormat.ODI:
            return cls(
                innings_overs=50,
                powerplay_overs=10,
                maximum_overs_per_bowler=10,
            )
        return cls(innings_overs=20, powerplay_overs=6, maximum_overs_per_bowler=4)


@dataclass(slots=True)
class BowlerFigures:
    """Legal-ball, run, and wicket record for one bowler in this innings."""

    legal_balls: int = 0
    runs_conceded: int = 0
    wickets: int = 0

    @property
    def completed_overs(self) -> int:
        return self.legal_balls // 6


@dataclass(frozen=True, slots=True)
class DeliveryEvent:
    """A confirmed ball-by-ball event from an authoritative live feed."""

    striker: str
    non_striker: str
    bowler: str
    total_runs: int
    batter_runs: int = 0
    extras: dict[str, int] = field(default_factory=dict)
    wicket_kind: str | None = None

    @property
    def is_legal_delivery(self) -> bool:
        """Wides and no-balls do not count towards the six legal balls."""
        return "wides" not in self.extras and "noballs" not in self.extras


class MatchStateError(ValueError):
    """Raised when an event would create an impossible match state."""


@dataclass(slots=True)
class MatchState:
    """Confirmed live state for one innings, updated only by delivery events."""

    match_format: MatchFormat
    batting_team: str
    bowling_team: str
    striker: str
    non_striker: str
    score: int = 0
    wickets: int = 0
    completed_overs: int = 0
    legal_balls_in_over: int = 0
    current_bowler: str | None = None
    previous_over_bowler: str | None = None
    bowler_figures: dict[str, BowlerFigures] = field(default_factory=dict)
    prediction_locked_for_over: int | None = None

    @property
    def rules(self) -> FormatRules:
        return FormatRules.for_format(self.match_format)

    @property
    def over_number(self) -> int:
        """One-based over number currently being bowled."""
        return self.completed_overs + 1

    @property
    def is_over_complete(self) -> bool:
        return self.legal_balls_in_over == 6

    def begin_over(self, bowler: str) -> None:
        """Confirm a new over and reject illegal bowler selections."""
        if not bowler:
            raise MatchStateError(
                "A confirmed bowler is required before an over starts."
            )
        if self.current_bowler is not None and not self.is_over_complete:
            raise MatchStateError("The current over is not complete.")
        if self.completed_overs >= self.rules.innings_overs:
            raise MatchStateError("The innings has reached its over limit.")
        if bowler == self.previous_over_bowler:
            raise MatchStateError("A bowler cannot bowl consecutive overs.")
        figures = self.bowler_figures.setdefault(bowler, BowlerFigures())
        if figures.completed_overs >= self.rules.maximum_overs_per_bowler:
            raise MatchStateError(
                f"{bowler} has already bowled the "
                f"{self.rules.maximum_overs_per_bowler}-over limit."
            )
        self.current_bowler = bowler
        self.legal_balls_in_over = 0

    def lock_prediction(self) -> int:
        """Reserve the current over for one prediction until it is resolved."""
        if self.current_bowler is None:
            raise MatchStateError("Cannot predict until the next bowler is confirmed.")
        if self.prediction_locked_for_over is not None:
            raise MatchStateError(
                "A prediction is already locked for the current over."
            )
        self.prediction_locked_for_over = self.over_number
        return self.over_number

    def apply_delivery(self, event: DeliveryEvent) -> None:
        """Apply one confirmed delivery and update score, figures, and strike."""
        if self.current_bowler is None:
            raise MatchStateError(
                "Cannot apply a delivery before the bowler is confirmed."
            )
        if event.bowler != self.current_bowler:
            raise MatchStateError(
                "Delivery bowler does not match the confirmed over bowler."
            )
        if event.striker != self.striker or event.non_striker != self.non_striker:
            raise MatchStateError(
                "Delivery batting pair does not match the confirmed live state."
            )
        if event.total_runs < 0 or event.batter_runs < 0:
            raise MatchStateError("Runs cannot be negative.")

        self.score += event.total_runs
        figures = self.bowler_figures[self.current_bowler]
        figures.runs_conceded += (
            event.total_runs
            - event.extras.get("byes", 0)
            - event.extras.get("legbyes", 0)
        )
        if event.is_legal_delivery:
            self.legal_balls_in_over += 1
            figures.legal_balls += 1
        if event.wicket_kind and event.wicket_kind != "retired hurt":
            self.wickets += 1
            figures.wickets += 1

        if event.batter_runs % 2:
            self.striker, self.non_striker = self.non_striker, self.striker

        if self.is_over_complete:
            self.completed_overs += 1
            self.previous_over_bowler = self.current_bowler
            self.current_bowler = None
            self.prediction_locked_for_over = None
            self.striker, self.non_striker = self.non_striker, self.striker
