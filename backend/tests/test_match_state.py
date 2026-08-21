"""Tests for the canonical live innings state."""

from __future__ import annotations

import pytest

from app.models.match_state import (
    BowlerFigures,
    DeliveryEvent,
    MatchFormat,
    MatchState,
    MatchStateError,
)


def _ball(state: MatchState, runs: int = 0, wicket: str | None = None) -> None:
    state.apply_delivery(
        DeliveryEvent(
            striker=state.striker,
            non_striker=state.non_striker,
            bowler=state.current_bowler or "",
            total_runs=runs,
            batter_runs=runs,
            wicket_kind=wicket,
        )
    )


def test_state_confirms_one_prediction_and_updates_an_over() -> None:
    state = MatchState(MatchFormat.IPL, "India", "England", "R Sharma", "S Gill")

    state.begin_over("J Archer")
    assert state.lock_prediction() == 1
    with pytest.raises(MatchStateError, match="already locked"):
        state.lock_prediction()

    _ball(state, runs=1)
    _ball(state, wicket="bowled")
    for _ in range(4):
        _ball(state)

    assert state.score == 1
    assert state.wickets == 1
    assert state.completed_overs == 1
    assert state.current_bowler is None
    assert state.prediction_locked_for_over is None
    assert state.bowler_figures["J Archer"].legal_balls == 6


def test_state_enforces_format_specific_bowling_limits() -> None:
    state = MatchState(MatchFormat.IPL, "India", "England", "R Sharma", "S Gill")
    state.bowler_figures["J Archer"] = BowlerFigures(legal_balls=24)
    with pytest.raises(MatchStateError, match="4-over limit"):
        state.begin_over("J Archer")

    state = MatchState(MatchFormat.ODI, "India", "England", "R Sharma", "S Gill")
    state.bowler_figures["J Archer"] = BowlerFigures(legal_balls=54)
    state.begin_over("J Archer")
    assert state.current_bowler == "J Archer"


def test_state_rejects_consecutive_bowlers_and_non_legal_balls() -> None:
    state = MatchState(MatchFormat.T20I, "India", "England", "R Sharma", "S Gill")
    state.begin_over("J Archer")
    state.apply_delivery(
        DeliveryEvent(
            "R Sharma",
            "S Gill",
            "J Archer",
            total_runs=1,
            extras={"wides": 1},
        )
    )
    assert state.legal_balls_in_over == 0
    for _ in range(6):
        _ball(state)
    with pytest.raises(MatchStateError, match="consecutive"):
        state.begin_over("J Archer")
