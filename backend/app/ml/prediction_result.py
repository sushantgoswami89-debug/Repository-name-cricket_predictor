"""
Prediction Result Model.

Represents the output of the CricketBaba prediction engine.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PredictionResult:
    """
    Represents the output of the prediction engine.
    """

    predicted_runs: float

    wicket_probability: float

    confidence: float

    analysis: list[str] = field(default_factory=list)

    metadata: dict[str, str] = field(default_factory=dict)

    def validate(self) -> bool:
        """
        Validate the prediction result.
        """

        logger.info("Validating PredictionResult")

        if self.predicted_runs < 0:
            raise ValueError("Predicted runs cannot be negative.")

        if not 0.0 <= self.wicket_probability <= 1.0:
            raise ValueError("Wicket probability must be between 0 and 1.")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")

        logger.info("PredictionResult validation successful")

        return True

    def to_dict(self) -> dict[str, object]:
        """
        Convert PredictionResult to a dictionary.
        """

        logger.info("Converting PredictionResult to dictionary")

        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> "PredictionResult":
        """
        Create PredictionResult from a dictionary.
        """

        logger.info("Creating PredictionResult from dictionary")

        return cls(**data)

    def summary(self) -> str:
        """
        Return a human-readable summary.
        """

        return (
            f"Runs: {self.predicted_runs:.1f} | "
            f"Wicket Probability: "
            f"{self.wicket_probability:.1%} | "
            f"Confidence: "
            f"{self.confidence:.1%}"
        )
