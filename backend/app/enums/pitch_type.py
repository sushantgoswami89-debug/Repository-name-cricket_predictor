"""
Pitch type enumeration.
"""

from enum import Enum


class PitchType(str, Enum):
    """Supported pitch types."""

    BATTING = "Batting"
    BOWLING = "Bowling"
    BALANCED = "Balanced"
    SPIN = "Spin"
    PACE = "Pace"

    @classmethod
    def from_string(cls, value: str) -> "PitchType":
        """
        Convert a string into a PitchType enum.

        Raises
        ------
        ValueError
            If the pitch type is unsupported.
        """

        normalized = value.strip().lower()

        mapping = {
            "batting": cls.BATTING,
            "bowling": cls.BOWLING,
            "balanced": cls.BALANCED,
            "spin": cls.SPIN,
            "pace": cls.PACE,
        }

        if normalized not in mapping:
            raise ValueError(f"Unsupported pitch type: {value}")

        return mapping[normalized]
