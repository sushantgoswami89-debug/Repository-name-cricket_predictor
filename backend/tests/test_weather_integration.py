"""
Integration tests for WeatherContext inside MatchContext.
"""

from app.services.match_context_service import MatchContextService


def test_weather_is_attached() -> None:
    context = MatchContextService.build(
        {
            "team1": "India",
            "team2": "Australia",
            "venue": "MCG",
            "format": "T20",
            "weather": {
                "temperature": 24,
                "humidity": 70,
                "wind_speed": 15,
                "rain_probability": 10,
                "cloud_cover": 25,
                "dew_probability": 65,
            },
        }
    )

    assert context.weather.temperature == 24
    assert context.weather.humidity == 70
    assert context.weather.dew_probability == 65
