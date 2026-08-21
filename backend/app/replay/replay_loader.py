"""Load historical matches as prediction-safe replay frames.

The loader exposes the state before an over starts and keeps the over result
separate so callers can predict first, then reveal the actual outcome.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.models.match_data import Delivery, Innings, Match, Over
from app.models.match_state import (
    BowlerFigures,
    MatchFormat,
    MatchState,
    MatchStateError,
)

logger = logging.getLogger(__name__)


class ReplayLoaderError(ValueError):
    """Raised when a historical match cannot be replayed safely."""


class ReplayRuleMode(StrEnum):
    """How recorded historical rule violations are handled."""

    STRICT = "strict"
    ARCHIVAL = "archival"


@dataclass(frozen=True, slots=True)
class ReplayRuleAnomaly:
    """A recorded event that conflicts with the selected standard rules."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ActualOverResult:
    """The hidden result of an over, revealed only after prediction."""

    runs: int
    wickets: int
    deliveries: tuple[Delivery, ...]


@dataclass(frozen=True, slots=True)
class ReplayOverFrame:
    """Prediction-ready state before an over and its eventual actual result."""

    match_id: str
    innings_index: int
    innings_team: str
    is_super_over: bool
    venue: str | None
    city: str | None
    over_number: int
    state: MatchState
    actual: ActualOverResult
    rule_anomalies: tuple[ReplayRuleAnomaly, ...] = ()


class ReplayLoader:
    """Load one historical match and stream overs in chronological order."""

    _NON_DISMISSAL_KINDS = frozenset({"retired hurt"})

    def __init__(
        self,
        json_loader: JsonLoader | None = None,
        match_parser: MatchParser | None = None,
        rule_mode: ReplayRuleMode = ReplayRuleMode.STRICT,
    ) -> None:
        self._json_loader = json_loader or JsonLoader()
        self._match_parser = match_parser or MatchParser()
        self._rule_mode = rule_mode

    def load_match(self, path: str | Path) -> Match:
        """Load and parse one Cricsheet JSON match file."""

        match_path = Path(path)
        logger.info("Loading replay match: %s", match_path)
        return self._match_parser.parse(self._json_loader.load(match_path))

    def frames_from_file(self, path: str | Path) -> Iterator[ReplayOverFrame]:
        """Yield replay frames for every innings in one match file."""

        match_path = Path(path)
        match = self.load_match(match_path)
        yield from self.frames(match, match_id=match_path.stem)

    def frames(self, match: Match, match_id: str = "unknown") -> Iterator[ReplayOverFrame]:
        """Yield overs from a parsed match without exposing future delivery data."""

        match_format = self._match_format(match.info.match_type)

        for innings_index, innings in enumerate(match.innings):
            bowling_team = self._bowling_team(match.info.teams, innings.team)
            yield from self._innings_frames(
                match_id=match_id,
                innings_index=innings_index,
                innings=innings,
                bowling_team=bowling_team,
                match_format=match_format,
                is_super_over=innings.super_over,
                venue=match.info.venue,
                city=match.info.city,
            )

    def _innings_frames(
        self,
        match_id: str,
        innings_index: int,
        innings: Innings,
        bowling_team: str,
        match_format: MatchFormat,
        is_super_over: bool,
        venue: str | None,
        city: str | None,
    ) -> Iterator[ReplayOverFrame]:
        first_delivery = self._first_delivery(innings)
        if first_delivery is None:
            logger.info("Skipping empty innings %d for %s.", innings_index, match_id)
            return

        state = MatchState(
            match_format=match_format,
            batting_team=innings.team,
            bowling_team=bowling_team,
            striker=first_delivery.batter,
            non_striker=first_delivery.non_striker,
        )

        for over in sorted(innings.overs, key=lambda item: item.over_number):
            if not over.deliveries:
                logger.debug("Skipping empty over %d for %s.", over.over_number, match_id)
                continue

            bowler = over.deliveries[0].bowler
            state.completed_overs = over.over_number
            state.striker = over.deliveries[0].batter
            state.non_striker = over.deliveries[0].non_striker
            rule_anomalies = self._begin_recorded_over(state, bowler)
            prediction_state = self._copy_state(state)
            actual = self._actual_over_result(over)

            logger.debug(
                "Replay frame %s innings=%d over=%d score=%d/%d.",
                match_id,
                innings_index,
                prediction_state.over_number,
                prediction_state.score,
                prediction_state.wickets,
            )
            yield ReplayOverFrame(
                match_id=match_id,
                innings_index=innings_index,
                innings_team=innings.team,
                is_super_over=is_super_over,
                venue=venue,
                city=city,
                over_number=prediction_state.over_number,
                state=prediction_state,
                actual=actual,
                rule_anomalies=rule_anomalies,
            )

            self._apply_over_result(state, over, actual)

    def _begin_recorded_over(
        self, state: MatchState, bowler: str
    ) -> tuple[ReplayRuleAnomaly, ...]:
        try:
            state.begin_over(bowler)
            return ()
        except MatchStateError as error:
            if self._rule_mode is ReplayRuleMode.STRICT:
                raise
            anomaly = ReplayRuleAnomaly(
                code=self._rule_anomaly_code(str(error)),
                message=str(error),
            )
            logger.warning("Archival replay rule exception: %s", error)
            state.current_bowler = bowler
            state.legal_balls_in_over = 0
            state.bowler_figures.setdefault(bowler, BowlerFigures())
            return (anomaly,)

    @staticmethod
    def _rule_anomaly_code(message: str) -> str:
        if "over limit" in message:
            return "bowler_over_limit"
        if "consecutive overs" in message:
            return "consecutive_over_bowler"
        if "innings has reached" in message:
            return "innings_over_limit"
        return "recorded_rule_exception"

    def _actual_over_result(self, over: Over) -> ActualOverResult:
        return ActualOverResult(
            runs=sum(delivery.runs.get("total", 0) for delivery in over.deliveries),
            wickets=sum(self._dismissal_count(delivery) for delivery in over.deliveries),
            deliveries=tuple(over.deliveries),
        )

    def _apply_over_result(
        self,
        state: MatchState,
        over: Over,
        actual: ActualOverResult,
    ) -> None:
        state.score += actual.runs
        state.wickets += actual.wickets
        for delivery in over.deliveries:
            figures = state.bowler_figures.setdefault(
                delivery.bowler, BowlerFigures()
            )
            figures.legal_balls += int(self._is_legal_delivery(delivery))
            figures.runs_conceded += self._bowler_runs(delivery)
            figures.wickets += self._bowler_wickets(delivery)

        finishing_bowler = over.deliveries[-1].bowler
        state.completed_overs = over.over_number + 1
        state.legal_balls_in_over = 0
        state.previous_over_bowler = finishing_bowler
        state.current_bowler = None
        state.prediction_locked_for_over = None

        last_delivery = over.deliveries[-1]
        state.striker = last_delivery.batter
        state.non_striker = last_delivery.non_striker

    @staticmethod
    def _is_legal_delivery(delivery: Delivery) -> bool:
        return "wides" not in delivery.extras and "noballs" not in delivery.extras

    @staticmethod
    def _bowler_runs(delivery: Delivery) -> int:
        return (
            delivery.runs.get("total", 0)
            - delivery.extras.get("byes", 0)
            - delivery.extras.get("legbyes", 0)
        )

    def _dismissal_count(self, delivery: Delivery) -> int:
        return sum(
            wicket.get("kind") not in self._NON_DISMISSAL_KINDS
            for wicket in delivery.wickets
        )

    @staticmethod
    def _bowler_wickets(delivery: Delivery) -> int:
        """Count only dismissals credited to the bowler's figures."""

        not_credited = frozenset(
            {"retired hurt", "retired out", "run out", "obstructing the field"}
        )
        return sum(wicket.get("kind") not in not_credited for wicket in delivery.wickets)

    @staticmethod
    def _first_delivery(innings: Innings) -> Delivery | None:
        for over in sorted(innings.overs, key=lambda item: item.over_number):
            if over.deliveries:
                return over.deliveries[0]
        return None

    @staticmethod
    def _bowling_team(teams: list[str], batting_team: str) -> str:
        for team in teams:
            if team != batting_team:
                return team
        raise ReplayLoaderError(f"Could not determine bowling team for {batting_team}.")

    @staticmethod
    def _match_format(match_type: str) -> MatchFormat:
        normalized = match_type.upper()
        if normalized == "ODI":
            return MatchFormat.ODI
        if normalized == "T20":
            return MatchFormat.T20I
        return MatchFormat.IPL

    @staticmethod
    def _copy_state(state: MatchState) -> MatchState:
        return MatchState(
            match_format=state.match_format,
            batting_team=state.batting_team,
            bowling_team=state.bowling_team,
            striker=state.striker,
            non_striker=state.non_striker,
            score=state.score,
            wickets=state.wickets,
            completed_overs=state.completed_overs,
            legal_balls_in_over=state.legal_balls_in_over,
            current_bowler=state.current_bowler,
            previous_over_bowler=state.previous_over_bowler,
            bowler_figures={
                bowler: BowlerFigures(
                    legal_balls=figures.legal_balls,
                    runs_conceded=figures.runs_conceded,
                    wickets=figures.wickets,
                )
                for bowler, figures in state.bowler_figures.items()
            },
            prediction_locked_for_over=state.prediction_locked_for_over,
        )
