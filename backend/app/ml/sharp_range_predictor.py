"""Runtime two-run Sharp Range prediction for CricketBaba Candidate v3.2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ml.model_repository import ModelRepository


def select_inclusive_bands(
    probabilities: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the strongest inclusive run band and its probability per row."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Probabilities must be a non-empty two-dimensional array.")
    if not isinstance(width, int) or width < 0:
        raise ValueError("width must be a non-negative integer.")
    band_size = width + 1
    if band_size > values.shape[1]:
        raise ValueError("width exceeds the model outcome range.")
    windows = np.column_stack(
        [
            values[:, start : start + band_size].sum(axis=1)
            for start in range(values.shape[1] - width)
        ]
    )
    low = np.argmax(windows, axis=1)
    return low, windows[np.arange(len(values)), low]


@dataclass(frozen=True, slots=True)
class SharpRange:
    """The most likely inclusive two-run band from the outcome distribution."""

    low: int
    high: int

    @property
    def display(self) -> str:
        suffix = "+" if self.high >= 30 else ""
        return f"{self.low}-{self.high}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["display"] = self.display
        result["width"] = self.high - self.low
        return result


class SharpRangePredictor:
    """Load the validated runs distribution and select its best two-run band."""

    def __init__(self, candidate_dir: Path) -> None:
        repository = ModelRepository(candidate_dir)
        self._model: Any = repository.load_artifact("sharp_range_model.pkl")
        self._features: list[str] = repository.load_artifact("feature_cols.pkl")
        self._categorical: list[str] = repository.load_artifact("cat_cols.pkl")

    def predict(self, state: pd.DataFrame) -> SharpRange:
        if len(state) != 1:
            raise ValueError("Sharp Range requires exactly one pre-over state.")
        x = state[self._features].copy()
        for column in self._categorical:
            x[column] = x[column].astype("category")
        probability = self._model.predict_proba(x)[0]
        low, _ = select_inclusive_bands(probability.reshape(1, -1), width=2)
        low = int(low[0])
        return SharpRange(low=low, high=low + 2)
