"""
Toss Context Model.

Stores structured toss information for a cricket match.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TossContext:
    """
    Represents toss information for a cricket match.
    """

    winner: str | None = None
    decision: str | None = None

    def validate(self) -> bool:
        """
        Validate toss information.
        """

        logger.info("Validating TossContext")

        valid_decisions = {"bat", "bowl", None}

        if self.decision is not None:
            self.decision = self.decision.lower().strip()

        if self.decision not in valid_decisions:
            raise ValueError("Toss decision must be either 'bat' or 'bowl'.")

        logger.info("TossContext validation successful")

        return True

    def to_dict(self) -> dict[str, str | None]:
        """
        Convert TossContext into a dictionary.
        """

        logger.info("Converting TossContext to dictionary")

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, str | None]) -> "TossContext":
        """
        Create TossContext from a dictionary.
        """

        logger.info("Creating TossContext from dictionary")

        return cls(**data)
