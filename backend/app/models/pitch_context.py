"""
Pitch Context Model.

Stores structured pitch information for prediction models.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PitchContext:
    """
    Represents pitch conditions.
    """

    pitch_type: str = "Balanced"

    batting_rating: int = 50
    pace_assistance: int = 50
    spin_assistance: int = 50

    bounce: int = 50
    turn: int = 50

    wear_factor: int = 50

    def validate(self) -> bool:
        """
        Validate pitch ratings.
        """

        logger.info("Validating PitchContext")

        ratings = {
            "batting_rating": self.batting_rating,
            "pace_assistance": self.pace_assistance,
            "spin_assistance": self.spin_assistance,
            "bounce": self.bounce,
            "turn": self.turn,
            "wear_factor": self.wear_factor,
        }

        for field_name, value in ratings.items():
            if not 0 <= value <= 100:
                raise ValueError(f"{field_name} must be between 0 and 100.")

        logger.info("PitchContext validation successful")

        return True

    def to_dict(self) -> Dict[str, int | str]:
        """
        Convert PitchContext to dictionary.
        """

        return asdict(self)
