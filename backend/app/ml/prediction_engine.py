"""
Prediction Engine.

Runs the CricketBaba ML models and returns a PredictionResult.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.ml.feature_builder import FeatureBuilder
from app.ml.model_repository import ModelRepository
from app.ml.prediction_result import PredictionResult
from app.models.match_context import MatchContext


class PredictionEngine:
    """
    Executes CricketBaba ML predictions.
    """

    def __init__(
        self,
        repository: ModelRepository | None = None,
        feature_builder: FeatureBuilder | None = None,
    ) -> None:
        self._repository = repository or ModelRepository()
        self._builder = feature_builder or FeatureBuilder()

    def predict(self, context: MatchContext) -> PredictionResult:
        """
        Predict runs and wicket probability.
        """

        features = self._builder.build(context)

        feature_order = self._repository.feature_columns

        missing = [f for f in feature_order if f not in features]

        if missing:
            raise ValueError(f"Missing model features: {missing}")

        ordered = {k: features[k] for k in feature_order}

        df = pd.DataFrame([ordered])

        runs = float(self._repository.runs_model.predict(df)[0])

        wicket_probability = float(
            self._repository.wicket_model.predict_proba(df)[0][1]
        )

        confidence = 0.90

        result = PredictionResult(
            predicted_runs=runs,
            wicket_probability=wicket_probability,
            confidence=confidence,
        )

        result.validate()

        return result
