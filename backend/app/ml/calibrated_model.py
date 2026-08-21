"""Inference-compatible probability calibration wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np


class CalibratedBinaryClassifier:
    """Apply a fitted calibrator while preserving ``predict_proba`` semantics."""

    def __init__(self, model: Any, calibrator: Any) -> None:
        self.model = model
        self.calibrator = calibrator

    def predict_proba(self, features: Any) -> np.ndarray:
        raw_probability = self.model.predict_proba(features)[:, 1]
        probability = np.asarray(self.calibrator.predict(raw_probability))
        return np.column_stack((1.0 - probability, probability))
