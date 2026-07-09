"""
Tests for WeatherContext.
"""

from app.models.weather_context import WeatherContext


def test_weather_context_validation() -> None:
    weather = WeatherContext(
        temperature=24.5,
        humidity=68,
        wind_speed=12,
        rain_probability=20,
        cloud_cover=40,
        dew_probability=65,
    )

    assert weather.validate() is True


def test_weather_context_dictionary() -> None:
    weather = WeatherContext(
        temperature=30,
        humidity=50,
        wind_speed=15,
        rain_probability=10,
        cloud_cover=30,
        dew_probability=20,
    )

    data = weather.to_dict()

    assert data["temperature"] == 30
    assert data["humidity"] == 50
