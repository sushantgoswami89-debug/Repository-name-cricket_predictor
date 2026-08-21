"""Tests for inference-compatible probability calibration."""

from __future__ import annotations

import numpy as np

from app.ml.calibrated_model import CalibratedBinaryClassifier


class _Model:
    def predict_proba(self, features: object) -> np.ndarray:
        return np.array([[0.2, 0.8], [0.7, 0.3]])


class _Calibrator:
    def predict(self, probability: np.ndarray) -> np.ndarray:
        return probability / 2


def test_calibrated_classifier_preserves_two_column_probabilities() -> None:
    probability = CalibratedBinaryClassifier(_Model(), _Calibrator()).predict_proba(
        object()
    )

    assert np.allclose(probability, [[0.6, 0.4], [0.85, 0.15]])
    assert np.allclose(probability.sum(axis=1), [1.0, 1.0])
