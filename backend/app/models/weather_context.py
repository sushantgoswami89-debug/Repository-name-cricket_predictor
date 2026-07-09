"""
Weather Context Model.

Stores weather information required by the prediction engine.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WeatherContext:
    """
    Represents weather conditions for a match.
    """

    temperature: float = 0.0
    humidity: float = 0.0
    wind_speed: float = 0.0
    rain_probability: float = 0.0
    cloud_cover: float = 0.0
    dew_probability: float = 0.0

    def validate(self) -> bool:
        """
        Validate weather values.
        """

        logger.info("Validating WeatherContext")

        percentage_fields = {
            "humidity": self.humidity,
            "rain_probability": self.rain_probability,
            "cloud_cover": self.cloud_cover,
            "dew_probability": self.dew_probability,
        }

        for field_name, value in percentage_fields.items():
            if not 0 <= value <= 100:
                raise ValueError(f"{field_name} must be between 0 and 100.")

        logger.info("WeatherContext validation successful")

        return True

    def to_dict(self) -> Dict[str, float]:
        """
        Convert WeatherContext into a dictionary.
        """

        return asdict(self)
