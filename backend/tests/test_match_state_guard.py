"""Regression tests for the self-checking live-pipeline guard.

Each case codifies a real bug that previously reached production and was
only caught by a human watching a live match by hand (docs/
live_field_test_2026-09-15.md, docs/live_field_test_2026-09-17.md).
"""

from __future__ import annotations

import pytest

from app.live.pipeline import MatchStateAnomaly, _check_match_state
from app.live.live_snapshot import LiveDelivery, LiveSnapshot
from app.models.live_match_state import LiveMatchState
from app.models.match_context import MatchContext


def _snapshot(**overrides) -> LiveSnapshot:
    values = dict(
        match_id="match",
        match_format="T20",
        batting_team="India",
        bowling_team="Afghanistan",
        innings=1,
        score=100,
        wickets=2,
        overs="10.0",
        is_live=True,
        deliveries=(),
        scheduled_overs=20,
        competition="",
    )
    values.update(overrides)
    return LiveSnapshot(**values)


def _context(team1: str, team2: str, competition: str = "") -> MatchContext:
    return MatchContext(
        match_id="match",
        team1=team1,
        team2=team2,
        format="T20",
        competition=competition,
        live=LiveMatchState(over=11),
    )


def test_valid_state_does_not_raise():
    _check_match_state(_snapshot(), _context("India", "Afghanistan"), next_over=11)


def test_swapped_identical_teams_raises():
    with pytest.raises(MatchStateAnomaly, match="batting_team == bowling_team"):
        _check_match_state(_snapshot(), _context("India", "India"), next_over=11)


def test_empty_team_name_raises():
    with pytest.raises(MatchStateAnomaly, match="empty team name"):
        _check_match_state(_snapshot(), _context("India", ""), next_over=11)


def test_over_past_scheduled_overs_raises():
    snapshot = _snapshot(scheduled_overs=20)
    with pytest.raises(MatchStateAnomaly, match="scheduled_overs"):
        _check_match_state(snapshot, _context("India", "Afghanistan"), next_over=22)


def test_over_within_one_over_grace_does_not_raise():
    snapshot = _snapshot(scheduled_overs=20)
    _check_match_state(snapshot, _context("India", "Afghanistan"), next_over=21)


def test_ipl_unresolved_team_name_raises():
    context = _context("Chennai Super Kings", "Not A Real Team", competition="IPL")
    with pytest.raises(MatchStateAnomaly, match="does not resolve"):
        _check_match_state(_snapshot(competition="IPL"), context, next_over=11)


def test_ipl_known_franchises_do_not_raise():
    context = _context("Chennai Super Kings", "Mumbai Indians", competition="IPL")
    _check_match_state(_snapshot(competition="IPL"), context, next_over=11)


def test_t20i_unlisted_team_names_do_not_raise():
    # Only IPL has a fixed 10-team universe -- T20I/associate nations have
    # no equivalent canonical registry, so this check is IPL-only.
    context = _context("Nepal", "Papua New Guinea")
    _check_match_state(_snapshot(), context, next_over=11)
