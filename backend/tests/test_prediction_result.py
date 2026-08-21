"""
Tests for PredictionResult.
"""

import pytest

from app.ml.prediction_result import PredictionResult


def test_prediction_result() -> None:
    """
    Test PredictionResult creation.
    """

    result = PredictionResult(
        predicted_runs=184.5,
        wicket_probability=0.27,
        confidence=0.91,
        analysis=["Balanced pitch"],
    )

    assert result.validate()

    assert result.predicted_runs == 184.5
    assert result.wicket_probability == 0.27
    assert result.confidence == 0.91
    assert result.confidence_percent == 91
    assert result.confidence_level == "HIGH"
    assert result.confidence_meter == "█████████░"

    output = result.to_dict()
    assert output["expected_range"] == ""
    assert output["confidence_percent"] == 91
    assert output["confidence_level"] == "HIGH"
    assert output["confidence_meter"] == "█████████░"

    assert result.analysis == ["Balanced pitch"]


def test_prediction_result_invalid_probability() -> None:
    """
    Wicket probability must be between 0 and 1.
    """

    with pytest.raises(ValueError):
        PredictionResult(
            predicted_runs=180,
            wicket_probability=1.5,
            confidence=0.9,
        ).validate()


def test_prediction_result_invalid_confidence() -> None:
    """
    Confidence must be between 0 and 1.
    """

    with pytest.raises(ValueError):
        PredictionResult(
            predicted_runs=180,
            wicket_probability=0.3,
            confidence=2.0,
        ).validate()


@pytest.mark.parametrize(
    ("confidence", "level"),
    [(0.59, "LOW"), (0.60, "MEDIUM"), (0.79, "MEDIUM"), (0.80, "HIGH")],
)
def test_confidence_level_boundaries(confidence: float, level: str) -> None:
    result = PredictionResult(7.0, 0.2, confidence, expected_range="3-12")
    assert result.confidence_level == level
