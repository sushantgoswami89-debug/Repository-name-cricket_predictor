"""Bounded, explainable match-level adaptation for shadow predictions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


class MatchIntelligenceError(ValueError):
    """Raised when prediction and reveal order is violated."""


@dataclass(frozen=True, slots=True)
class IntervalProfile:
    """Signed residual quantiles calibrated on historical unseen overs."""

    lower_residual: float
    upper_residual: float
    calibration_rows: int = 0


@dataclass(frozen=True, slots=True)
class AdaptivePrediction:
    """One adjusted prediction awaiting its actual result."""

    base_runs: float
    expected_runs: float
    base_wicket_probability: float
    wicket_probability: float
    range_low: float
    range_high: float
    phase: str


@dataclass(slots=True)
class MatchLearningState:
    """Small online state updated only after an over is revealed."""

    run_bias: float = 0.0
    wicket_offset: float = 0.0
    volatility: float = 0.0
    overs_observed: int = 0
    balls_observed: int = 0
    current_phase: str | None = None
    last_run_error: float | None = None
    last_wicket_residual: float | None = None
    last_adjustment: str = "No completed overs observed."


class MatchIntelligence:
    """Adjust a static model cautiously as confirmed match evidence arrives."""

    def __init__(
        self,
        interval_profiles: dict[str, IntervalProfile],
        run_learning_rate: float = 0.20,
        wicket_learning_rate: float = 0.15,
        maximum_current_match_weight: float = 0.50,
    ) -> None:
        if "default" not in interval_profiles:
            raise ValueError("A default interval profile is required.")
        self._profiles = interval_profiles
        self._run_learning_rate = run_learning_rate
        self._wicket_learning_rate = wicket_learning_rate
        self._maximum_current_match_weight = maximum_current_match_weight
        self.state = MatchLearningState()
        self._pending: AdaptivePrediction | None = None

    @classmethod
    def from_profile_data(cls, data: dict[str, dict[str, Any]]) -> "MatchIntelligence":
        """Construct from the persisted candidate interval-profile JSON."""

        return cls(
            {
                phase: IntervalProfile(
                    lower_residual=float(values["lower_residual"]),
                    upper_residual=float(values["upper_residual"]),
                    calibration_rows=int(values.get("calibration_rows", 0)),
                )
                for phase, values in data.items()
            }
        )

    @property
    def current_match_weight(self) -> float:
        return min(
            self._maximum_current_match_weight,
            self.state.balls_observed / 60.0,
        )

    def predict(
        self,
        base_runs: float,
        base_wicket_probability: float,
        phase: str,
    ) -> AdaptivePrediction:
        """Return one adjusted prediction and lock until its result is supplied."""

        if self._pending is not None:
            raise MatchIntelligenceError(
                "The previous prediction must be resolved with feed_actual_data first."
            )
        if base_runs < 0:
            raise ValueError("Base runs cannot be negative.")
        if not 0 <= base_wicket_probability <= 1:
            raise ValueError("Wicket probability must be between zero and one.")

        self._handle_phase_transition(phase)
        expected_runs = max(0.0, base_runs + self.state.run_bias)
        wicket_probability = self._adjust_probability(
            base_wicket_probability, self.state.wicket_offset
        )
        profile = self._profiles.get(phase, self._profiles["default"])
        volatility_padding = min(2.0, self.state.volatility * 0.10)
        prediction = AdaptivePrediction(
            base_runs=base_runs,
            expected_runs=expected_runs,
            base_wicket_probability=base_wicket_probability,
            wicket_probability=wicket_probability,
            range_low=max(
                0.0,
                expected_runs + profile.lower_residual - volatility_padding,
            ),
            range_high=(
                expected_runs + profile.upper_residual + volatility_padding
            ),
            phase=phase,
        )
        self._pending = prediction
        return prediction

    def feed_actual_data(
        self,
        actual_runs: int,
        actual_wickets: int,
        legal_balls: int = 6,
    ) -> dict[str, Any]:
        """Reveal a completed over and update only future predictions."""

        if self._pending is None:
            raise MatchIntelligenceError("No prediction is waiting for an actual result.")
        if actual_runs < 0 or actual_wickets < 0 or legal_balls < 0:
            raise ValueError("Actual match values cannot be negative.")

        prediction = self._pending
        run_error = actual_runs - prediction.expected_runs
        wicket_outcome = float(actual_wickets > 0)
        wicket_residual = wicket_outcome - prediction.wicket_probability

        self.state.run_bias = self._clamp(
            (1.0 - self._run_learning_rate) * self.state.run_bias
            + self._run_learning_rate * run_error,
            -3.0,
            3.0,
        )
        self.state.wicket_offset = self._clamp(
            0.85 * self.state.wicket_offset
            + self._wicket_learning_rate * wicket_residual,
            -0.8,
            0.8,
        )
        self.state.volatility = (
            0.80 * self.state.volatility + 0.20 * abs(run_error)
        )
        self.state.overs_observed += 1
        self.state.balls_observed += legal_balls
        self.state.last_run_error = run_error
        self.state.last_wicket_residual = wicket_residual
        self.state.last_adjustment = self._describe_adjustment(
            run_error, wicket_residual
        )
        self._pending = None
        return self.learning_snapshot()

    def learning_snapshot(self) -> dict[str, Any]:
        """Return a JSON-ready explanation of the current adaptive state."""

        snapshot = asdict(self.state)
        snapshot.update(
            {
                "current_match_weight": self.current_match_weight,
                "historical_weight": 1.0 - self.current_match_weight,
                "prediction_pending": self._pending is not None,
                "confidence": self._confidence(),
                "interval_target": 0.90,
            }
        )
        return {"self_learning": snapshot}

    def _handle_phase_transition(self, phase: str) -> None:
        previous = self.state.current_phase
        if previous is not None and previous != phase:
            self.state.run_bias *= 0.70
            self.state.wicket_offset *= 0.70
            self.state.last_adjustment = (
                f"Partial recalibration after phase changed from {previous} to {phase}."
            )
        self.state.current_phase = phase

    @staticmethod
    def _adjust_probability(probability: float, offset: float) -> float:
        clipped = min(max(probability, 1e-6), 1.0 - 1e-6)
        logit = math.log(clipped / (1.0 - clipped))
        return 1.0 / (1.0 + math.exp(-(logit + offset)))

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _describe_adjustment(run_error: float, wicket_residual: float) -> str:
        run_direction = "increased" if run_error > 0 else "decreased"
        wicket_direction = "increased" if wicket_residual > 0 else "reduced"
        return (
            f"Run expectation {run_direction} after {run_error:+.2f} run error; "
            f"wicket risk {wicket_direction} gradually."
        )

    def _confidence(self) -> str:
        if self.state.overs_observed < 2:
            return "low"
        if self.state.overs_observed < 6 or self.state.volatility > 4.0:
            return "medium"
        return "high"
