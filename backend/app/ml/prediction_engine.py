"""
Prediction Engine.

Runs the CricketBaba ML models and returns a PredictionResult.
"""

from __future__ import annotations

import time

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

        start = time.perf_counter()

        features = self._builder.build(context)

        feature_order = self._repository.get_feature_columns()

        missing = [name for name in feature_order if name not in features]

        if missing:
            raise ValueError(f"Missing model features: {missing}")

        ordered = {name: features[name] for name in feature_order}

        df = pd.DataFrame([ordered])

        # Ensure categorical columns have categorical dtype
        for column in self._repository.get_categorical_columns():
            if column in df.columns:
                df[column] = df[column].astype("category")

        runs_model = self._repository.get_runs_model()
        wicket_model = self._repository.get_wicket_model()

        predicted_runs = float(runs_model.predict(df)[0])

        wicket_probability = float(wicket_model.predict_proba(df)[0][1])

        latency_ms = (time.perf_counter() - start) * 1000

        confidence = 0.90

        analysis: list[str] = [
            f"Prediction completed in {latency_ms:.2f} ms",
        ]

        result = PredictionResult(
            predicted_runs=predicted_runs,
            wicket_probability=wicket_probability,
            confidence=confidence,
            analysis=analysis,
            metadata={
                "latency_ms": f"{latency_ms:.2f}",
            },
        )

        result.validate()

        return result
