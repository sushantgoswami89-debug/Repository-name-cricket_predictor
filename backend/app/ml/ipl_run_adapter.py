"""Small inference-only run adapter for an IPL-specific candidate."""

from __future__ import annotations

from typing import Any

import numpy as np


class PhaseAdjustedRegressor:
    """Apply calibration-only phase corrections to a fitted runs model."""

    def __init__(
        self,
        model: Any,
        corrections: dict[str, float],
        correction_columns: tuple[str, ...] = ("phase",),
    ) -> None:
        self.model = model
        self.corrections = dict(corrections)
        self.correction_columns = correction_columns

    def predict(self, features: Any) -> np.ndarray:
        prediction = np.asarray(self.model.predict(features), dtype=float)
        keys = features[list(self.correction_columns)].astype(str).agg("|".join, axis=1)
        adjustment = np.asarray(
            [
                self.corrections.get(
                    key,
                    self.corrections.get(str(features.iloc[index]["phase"]), 0.0),
                )
                for index, key in enumerate(keys)
            ]
        )
        return np.clip(prediction + adjustment, 0, None)
