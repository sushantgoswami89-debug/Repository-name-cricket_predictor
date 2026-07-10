"""
Tests for ContextBuilder.
"""

from app.services.context_builder import ContextBuilder


def test_context_builder() -> None:
    """
    Ensure ContextBuilder creates a valid MatchContext.
    """

    context = ContextBuilder.build(
        {
            "team1": "india",
            "team2": "TEAM AUSTRALIA",
            "venue": "MCG",
            "format": "twenty20",
            "toss_winner": "India",
            "toss_decision": "Bat",
            "weather": {
                "temperature": 24,
                "humidity": 58,
                "wind_speed": 12,
                "rain_probability": 5,
                "cloud_cover": 30,
                "dew_probability": 40,
            },
        }
    )

    assert context.team1 == "India"
    assert context.team2 == "Australia"

    assert context.format == "T20"

    assert context.city == "Melbourne"
    assert context.country == "Australia"

    assert context.pitch.pitch_type == "Balanced"

    assert context.toss.winner == "India"
    assert context.toss.decision == "bat"

    assert context.weather.temperature == 24
