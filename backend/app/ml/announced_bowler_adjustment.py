"""Leakage-safe inputs and bounded adjustment for an announced bowler.

This module is candidate-only.  In particular, it never discovers a bowler
from a delivery.  Callers must supply explicit pre-over scoreboard evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, log
from typing import Literal


BowlerSource = Literal["scoreboard_pre_over", "unannounced"]


@dataclass(frozen=True, slots=True)
class BowlerAnnouncement:
    """Evidence captured before the first ball of an explicitly numbered over."""

    bowler: str
    source: BowlerSource
    over: int
    expected_over: int
    row_is_empty: bool
    captured_before_first_ball: bool

    def accepted_bowler(self) -> str:
        if (
            self.source == "scoreboard_pre_over"
            and self.bowler.strip()
            and self.over == self.expected_over
            and self.row_is_empty
            and self.captured_before_first_ball
        ):
            return self.bowler.strip()
        return ""


@dataclass(frozen=True, slots=True)
class BowlerSpellFeatures:
    """Chronological bowler state available immediately before an over."""

    bowler_source: BowlerSource
    match_balls: int = 0
    match_runs_conceded: int = 0
    match_wickets: int = 0
    match_dot_rate: float = 0.0
    match_boundary_concession_rate: float = 0.0
    match_strike_rate: float = 0.0
    match_economy: float = 0.0
    overs_since_previous: int = -1
    consecutive_overs: int = 0
    spell_number: int = 0
    current_spell_balls: int = 0
    history_phase_balls: int = 0
    history_phase_economy: float = 0.0
    history_phase_strike_rate: float = 0.0
    history_supported: bool = False
    h2h_balls: int = 0
    h2h_runs: int = 0
    h2h_wickets: int = 0
    h2h_supported: bool = False

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)


@dataclass(slots=True)
class _MatchBowlerState:
    balls: int = 0
    runs: int = 0
    wickets: int = 0
    dots: int = 0
    boundaries: int = 0
    last_over: int = -1
    consecutive_overs: int = 0
    spell_number: int = 0
    spell_balls: int = 0


class CurrentSpellTracker:
    """Track completed legal deliveries and expose pre-over state.

    A new spell begins after two or more intervening overs.  One intervening
    over is normal when a bowler operates from one end and remains in the same
    spell.
    """

    def __init__(self) -> None:
        self._states: dict[str, _MatchBowlerState] = {}

    def features(
        self,
        announcement: BowlerAnnouncement,
        *,
        history_phase_balls: int = 0,
        history_phase_runs: int = 0,
        history_phase_wickets: int = 0,
        h2h_balls: int = 0,
        h2h_runs: int = 0,
        h2h_wickets: int = 0,
        history_support_balls: int = 120,
        h2h_support_balls: int = 24,
    ) -> BowlerSpellFeatures:
        bowler = announcement.accepted_bowler()
        if not bowler:
            return BowlerSpellFeatures(bowler_source="unannounced")

        state = self._states.get(bowler, _MatchBowlerState())
        gap = (
            announcement.expected_over - state.last_over - 1
            if state.last_over >= 0
            else -1
        )
        new_spell = state.last_over < 0 or gap >= 2
        spell_number = state.spell_number + int(new_spell)
        spell_balls = 0 if new_spell else state.spell_balls
        history_supported = history_phase_balls >= history_support_balls
        h2h_supported = h2h_balls >= h2h_support_balls
        return BowlerSpellFeatures(
            bowler_source="scoreboard_pre_over",
            match_balls=state.balls,
            match_runs_conceded=state.runs,
            match_wickets=state.wickets,
            match_dot_rate=state.dots / state.balls if state.balls else 0.0,
            match_boundary_concession_rate=(
                state.boundaries / state.balls if state.balls else 0.0
            ),
            match_strike_rate=(
                state.balls / state.wickets if state.wickets else 0.0
            ),
            match_economy=(
                6.0 * state.runs / state.balls if state.balls else 0.0
            ),
            overs_since_previous=gap,
            consecutive_overs=(
                state.consecutive_overs + 1 if gap == 0 else 1
            ),
            spell_number=spell_number,
            current_spell_balls=spell_balls,
            history_phase_balls=history_phase_balls,
            history_phase_economy=(
                6.0 * history_phase_runs / history_phase_balls
                if history_supported
                else 0.0
            ),
            history_phase_strike_rate=(
                history_phase_balls / history_phase_wickets
                if history_supported and history_phase_wickets
                else 0.0
            ),
            history_supported=history_supported,
            h2h_balls=h2h_balls if h2h_supported else 0,
            h2h_runs=h2h_runs if h2h_supported else 0,
            h2h_wickets=h2h_wickets if h2h_supported else 0,
            h2h_supported=h2h_supported,
        )

    def record_completed_over(
        self,
        *,
        bowler: str,
        over: int,
        legal_balls: int,
        runs_conceded: int,
        wickets: int,
        dots: int,
        boundaries: int,
    ) -> None:
        """Update state only after the over outcome is no longer future data."""

        if legal_balls < 0 or min(runs_conceded, wickets, dots, boundaries) < 0:
            raise ValueError("completed-over counts cannot be negative")
        state = self._states.setdefault(bowler, _MatchBowlerState())
        gap = over - state.last_over - 1 if state.last_over >= 0 else -1
        if state.last_over < 0 or gap >= 2:
            state.spell_number += 1
            state.spell_balls = 0
        state.consecutive_overs = state.consecutive_overs + 1 if gap == 0 else 1
        state.balls += legal_balls
        state.runs += runs_conceded
        state.wickets += wickets
        state.dots += dots
        state.boundaries += boundaries
        state.last_over = over
        state.spell_balls += legal_balls


@dataclass(frozen=True, slots=True)
class AdjustedPrediction:
    runs: float
    wicket_probability: float
    bowler_source: BowlerSource
    run_delta: float
    wicket_logit_delta: float


class BoundedBowlerAdjustment:
    """Apply a residual adjustment without letting identity dominate situation."""

    def __init__(
        self,
        *,
        maximum_run_delta: float = 1.5,
        maximum_wicket_logit_delta: float = 0.45,
    ) -> None:
        self.maximum_run_delta = maximum_run_delta
        self.maximum_wicket_logit_delta = maximum_wicket_logit_delta

    def apply(
        self,
        *,
        base_runs: float,
        base_wicket_probability: float,
        features: BowlerSpellFeatures,
        proposed_run_delta: float,
        proposed_wicket_logit_delta: float,
    ) -> AdjustedPrediction:
        if features.bowler_source != "scoreboard_pre_over":
            return AdjustedPrediction(
                base_runs, base_wicket_probability, "unannounced", 0.0, 0.0
            )

        # Current-match evidence earns weight quickly. Supported history can
        # contribute, but is capped so an identity prior cannot dominate.
        match_weight = features.match_balls / (features.match_balls + 18.0)
        history_weight = (
            min(0.35, features.history_phase_balls / 600.0)
            if features.history_supported
            else 0.0
        )
        h2h_weight = (
            min(0.15, features.h2h_balls / 240.0)
            if features.h2h_supported
            else 0.0
        )
        evidence_weight = min(1.0, match_weight + history_weight + h2h_weight)
        run_delta = _clip(
            proposed_run_delta * evidence_weight,
            -self.maximum_run_delta,
            self.maximum_run_delta,
        )
        wicket_delta = _clip(
            proposed_wicket_logit_delta * evidence_weight,
            -self.maximum_wicket_logit_delta,
            self.maximum_wicket_logit_delta,
        )
        probability = _logistic(_logit(base_wicket_probability) + wicket_delta)
        return AdjustedPrediction(
            runs=base_runs + run_delta,
            wicket_probability=probability,
            bowler_source="scoreboard_pre_over",
            run_delta=run_delta,
            wicket_logit_delta=wicket_delta,
        )


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _logit(probability: float) -> float:
    probability = _clip(probability, 1e-6, 1 - 1e-6)
    return log(probability / (1 - probability))


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))
