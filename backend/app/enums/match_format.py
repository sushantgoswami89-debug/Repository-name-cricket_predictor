"""
Match format enumeration.
"""

from enum import Enum


class MatchFormat(str, Enum):
    """Supported cricket match formats."""

    TEST = "Test"
    ODI = "ODI"
    T20 = "T20"

    @classmethod
    def from_string(cls, value: str) -> "MatchFormat":
        """
        Convert a string into a MatchFormat enum.

        Raises
        ------
        ValueError
            If the format is unsupported.
        """

        normalized = value.strip().lower()

        mapping = {
            "test": cls.TEST,
            "test match": cls.TEST,
            "odi": cls.ODI,
            "t20": cls.T20,
            "it20": cls.T20,
            "twenty20": cls.T20,
        }

        if normalized not in mapping:
            raise ValueError(f"Unsupported match format: {value}")

        return mapping[normalized]
