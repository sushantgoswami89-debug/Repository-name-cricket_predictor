"""
Integration tests for PitchContext inside MatchContext.
"""

from app.services.match_context_service import MatchContextService


def test_pitch_is_attached() -> None:
    context = MatchContextService.build(
        {
            "team1": "India",
            "team2": "Australia",
            "venue": "MCG",
            "format": "T20",
        }
    )

    assert context.pitch.pitch_type == "Balanced"

    assert context.pitch.batting_rating == 50
    assert context.pitch.pace_assistance == 50
    assert context.pitch.spin_assistance == 50

    assert context.pitch.bounce == 50
    assert context.pitch.turn == 50
    assert context.pitch.wear_factor == 50
