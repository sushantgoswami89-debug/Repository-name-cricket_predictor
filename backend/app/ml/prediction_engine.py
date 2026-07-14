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

    def _build_analysis(
        self,
        context: MatchContext,
        predicted_runs: float,
        latency_ms: float,
    ) -> list[str]:
        """
        Build human-readable explanations for the prediction.
        """

        analysis: list[str] = []

        over = context.live.over

        if over <= 6:
            analysis.append("Powerplay generally favors scoring.")
        elif over <= 15:
            analysis.append("Middle overs usually balance risk and reward.")
        else:
            analysis.append("Death overs often produce higher scoring.")

        if context.pitch.batting_rating >= 70:
            analysis.append("Pitch conditions favour batting.")
        elif context.pitch.bowling_rating >= 70:
            analysis.append("Pitch conditions favour bowling.")

        venue_avg = context.venue_stats.get("average_score")

        if isinstance(venue_avg, (int, float)):
            analysis.append(f"Venue average score: {venue_avg:.0f}.")

        analysis.append(f"Predicted runs this over: {predicted_runs:.1f}.")

        analysis.append(f"Prediction generated in {latency_ms:.2f} ms.")

        return analysis

    def predict(self, context: MatchContext) -> PredictionResult:
        """
        Predict runs and wicket probability.
        """

        start = time.perf_counter()

        features = self._builder.build(context)

        feature_order = self._repository.get_feature_columns()

        missing = [f for f in feature_order if f not in features]

        if missing:
            raise ValueError(f"Missing model features: {missing}")

        ordered = {k: features[k] for k in feature_order}

        df = pd.DataFrame([ordered])

        for column in self._repository.get_categorical_columns():
            if column in df.columns:
                df[column] = df[column].astype("category")

        runs_model = self._repository.get_runs_model()
        wicket_model = self._repository.get_wicket_model()

        predicted_runs = float(runs_model.predict(df)[0])

        wicket_probability = float(wicket_model.predict_proba(df)[0][1])

        latency_ms = (time.perf_counter() - start) * 1000

        analysis = self._build_analysis(
            context=context,
            predicted_runs=predicted_runs,
            latency_ms=latency_ms,
        )

        result = PredictionResult(
            predicted_runs=predicted_runs,
            wicket_probability=wicket_probability,
            confidence=0.90,
            analysis=analysis,
            metadata={
                "latency_ms": f"{latency_ms:.2f}",
            },
        )

        result.validate()

        return result
