"""Runtime conditional range and confidence enhancement for Candidate v3.1."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.ml.historical_analogue import HistoricalAnalogueEngine

RUN_BUCKETS = ("0-4", "5-8", "9-12", "13-16", "17+")


@dataclass(frozen=True, slots=True)
class EnhancedPrediction:
    """Candidate v3 point predictions enriched with historical evidence."""

    expected_runs: float
    wicket_probability: float
    range_low: float
    range_high: float
    confidence: float
    similar_situations: int
    effective_sample_size: float
    similarity_percent: int
    historical_expected_runs: float
    historical_wicket_probability: float
    most_common_runs_bucket: str
    most_common_bucket_probability: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnaloguePredictionEnhancer:
    """Load the validated v3.1 layer and enhance one pre-over prediction."""

    def __init__(self, candidate_dir: Path) -> None:
        self._analogue: HistoricalAnalogueEngine = joblib.load(
            candidate_dir / "historical_analogue_engine.pkl"
        )
        self._confidence: Any = joblib.load(candidate_dir / "confidence_calibrator.pkl")
        self._config = json.loads(
            (candidate_dir / "analogue_config.json").read_text(encoding="utf-8")
        )

    def enhance(
        self,
        state: pd.DataFrame,
        model_runs: float,
        model_wicket_probability: float,
    ) -> EnhancedPrediction:
        """Return conditional range, calibrated confidence, and analogues."""

        if len(state) != 1:
            raise ValueError("Runtime enhancement requires exactly one match state.")
        lower_quantile, upper_quantile = self._config["analogue_quantiles"]
        batch = self._analogue.query(
            state,
            lower_quantile=float(lower_quantile),
            upper_quantile=float(upper_quantile),
        )
        reliability = self._reliability(batch)
        run_weight = reliability * float(self._config["selected_run_maximum_weight"])
        wicket_weight = reliability * float(
            self._config["selected_wicket_maximum_weight"]
        )
        expected_runs = float(
            model_runs * (1.0 - run_weight) + batch.expected_runs[0] * run_weight
        )
        wicket_probability = float(
            model_wicket_probability * (1.0 - wicket_weight)
            + batch.wicket_probability[0] * wicket_weight
        )
        padding = float(self._config["selected_interval_padding"])
        shift = expected_runs - batch.expected_runs[0]
        low = max(0.0, float(batch.lower_runs[0] + shift - padding))
        high = float(batch.upper_runs[0] + shift + padding)
        confidence_features = self._confidence_features(
            state, model_runs, batch, low, high
        )
        confidence = float(self._confidence.predict_proba(confidence_features)[0, 1])
        probabilities = batch.run_bucket_probabilities[0]
        bucket_index = int(np.argmax(probabilities))
        return EnhancedPrediction(
            expected_runs=round(expected_runs, 1),
            wicket_probability=round(wicket_probability, 4),
            range_low=round(low, 1),
            range_high=round(high, 1),
            confidence=round(confidence, 4),
            similar_situations=int(batch.neighbor_count[0]),
            effective_sample_size=round(float(batch.effective_sample_size[0]), 1),
            similarity_percent=round(float(batch.similarity[0]) * 100),
            historical_expected_runs=round(float(batch.expected_runs[0]), 1),
            historical_wicket_probability=round(float(batch.wicket_probability[0]), 4),
            most_common_runs_bucket=RUN_BUCKETS[bucket_index],
            most_common_bucket_probability=round(float(probabilities[bucket_index]), 4),
        )

    @staticmethod
    def _reliability(batch: Any) -> float:
        sample = np.clip(batch.effective_sample_size[0] / 180.0, 0.0, 1.0)
        consistency = np.clip(1.0 - batch.outcome_std[0] / 10.0, 0.20, 1.0)
        return float(batch.similarity[0] * np.sqrt(sample) * consistency)

    def _confidence_features(
        self,
        state: pd.DataFrame,
        model_runs: float,
        batch: Any,
        low: float,
        high: float,
    ) -> pd.DataFrame:
        values = pd.DataFrame(
            {
                "similarity": [batch.similarity[0]],
                "log_effective_sample": [np.log1p(batch.effective_sample_size[0])],
                "outcome_std": [batch.outcome_std[0]],
                "model_analogue_gap": [abs(model_runs - batch.expected_runs[0])],
                "range_width": [high - low],
                "recent_wicket_rate": [float(state.iloc[0]["recent_wicket_rate"])],
                "required_run_rate": [float(state.iloc[0]["required_run_rate"])],
            }
        )
        phase = str(state.iloc[0]["phase"])
        values[f"phase_{phase}"] = 1.0
        columns = list(self._confidence.feature_names_in_)
        return values.reindex(columns=columns, fill_value=0.0)
