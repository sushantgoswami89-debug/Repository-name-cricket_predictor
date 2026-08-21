"""Tests for independent historical replay verification."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.models.match_data import Delivery, Innings, Match, MatchInfo, Over
from app.replay.match_state_verifier import (
    MatchStateVerificationError,
    MatchStateVerifier,
)
from app.replay.replay_loader import ReplayLoader


def _match() -> Match:
    return Match(
        info=MatchInfo(
            match_type="T20",
            teams=["Team A", "Team B"],
            venue="Eden Gardens",
            city="Kolkata",
        ),
        innings=[
            Innings(
                team="Team A",
                overs=[
                    Over(
                        0,
                        [
                            Delivery(
                                "0.1",
                                "Batter A",
                                "Bowler A",
                                "Batter B",
                                {"total": 4, "batter": 4},
                            )
                        ],
                    ),
                    Over(
                        1,
                        [
                            Delivery(
                                "1.1",
                                "Batter A",
                                "Bowler B",
                                "Batter B",
                                {"total": 0, "batter": 0},
                                wickets=[{"kind": "bowled"}],
                            )
                        ],
                    ),
                ],
            )
        ],
    )


def test_verifier_accepts_matching_replay_and_metadata() -> None:
    match = _match()
    frames = list(ReplayLoader().frames(match, match_id="sample"))

    result = MatchStateVerifier().assert_valid(match, frames, match_id="sample")

    assert result.is_valid
    assert result.frames_checked == 2
    assert frames[0].venue == "Eden Gardens"
    assert frames[0].city == "Kolkata"


def test_verifier_reports_a_precise_state_mismatch() -> None:
    match = _match()
    frames = list(ReplayLoader().frames(match, match_id="sample"))
    altered_state = ReplayLoader._copy_state(frames[1].state)
    altered_state.score = 99
    frames[1] = replace(frames[1], state=altered_state)

    result = MatchStateVerifier().verify(match, frames, match_id="sample")

    assert not result.is_valid
    assert result.issues[0].field == "state.score"
    assert result.issues[0].expected == 4
    assert result.issues[0].actual == 99
    with pytest.raises(MatchStateVerificationError, match="state.score"):
        result.raise_for_errors()


def test_verifier_detects_missing_frames() -> None:
    match = _match()
    frames = list(ReplayLoader().frames(match, match_id="sample"))

    result = MatchStateVerifier().verify(match, frames[:1], match_id="sample")

    assert not result.is_valid
    assert result.issues[0].field == "frame"
    assert result.issues[0].actual == "missing"
