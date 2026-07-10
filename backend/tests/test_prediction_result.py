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
