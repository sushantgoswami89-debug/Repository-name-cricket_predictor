"""
Tests for MatchContextService.
"""

from app.services.match_context_service import MatchContextService


def test_match_context_build() -> None:
    context = MatchContextService.build(
        {
            "team1": " ind ",
            "team2": "TEAM AUSTRALIA",
            "venue": "MCG",
            "format": "twenty20",
        }
    )

    assert context.team1 == "India"
    assert context.team2 == "Australia"
    assert context.city == "Melbourne"
    assert context.country == "Australia"

    assert context.pitch.pitch_type == "Balanced"

    assert context.format == "T20"
