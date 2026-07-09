"""
Tests for MatchFormat and PitchType enums.
"""

from app.enums.match_format import MatchFormat
from app.enums.pitch_type import PitchType


def test_match_format_enum() -> None:
    assert MatchFormat.from_string("t20") == MatchFormat.T20
    assert MatchFormat.from_string("ODI") == MatchFormat.ODI
    assert MatchFormat.from_string("Test Match") == MatchFormat.TEST


def test_pitch_type_enum() -> None:
    assert PitchType.from_string("balanced") == PitchType.BALANCED
    assert PitchType.from_string("Batting") == PitchType.BATTING
    assert PitchType.from_string("PACE") == PitchType.PACE
