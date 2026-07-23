"""
Prediction Engine v3 (Precision & Momentum Update).
Features: Adaptive Learning Rate, Match Rhythm Blending, and Ultra-Tight Bracketing.
"""

from __future__ import annotations

import time

import pandas as pd

from app.ml.engine_router import EngineFamily, EngineRouter
from app.ml.feature_builder import FeatureBuilder
from app.ml.model_repository import ModelRepository
from app.ml.prediction_result import PredictionResult
from app.models.match_context import MatchContext


class PredictionEngine:
    def __init__(
        self,
        repository: ModelRepository | None = None,
        feature_builder: FeatureBuilder | None = None,
    ) -> None:
        self._repository = repository or ModelRepository()
        self._builder = feature_builder or FeatureBuilder()

        # --- STATE MANAGEMENT ---
        self._match_bias = 1.0
        self._wicket_multiplier = 1.0
        self._last_predicted_over = -1
        self._is_awaiting_actuals = False
        self._last_point_prediction = 0.0

        # --- TRACKERS ---
        self._bowler_stats = {}
        self._error_history = []

    def _get_format_rules(self, style: str):
        specification = EngineRouter.resolve(style)
        return {
            "pp": specification.powerplay_overs,
            "max_overs": 10 if specification.family is EngineFamily.ODI else 4,
            "death": specification.death_starts,
            "engine_family": specification.family.value,
        }

    def _build_analysis(
        self, context, pred_runs, wkt_prob, latency, rules
    ) -> list[str]:
        analysis: list[str] = []
        over = context.live.over
        if over <= rules["pp"]:
            analysis.append("Phase: Powerplay (High Volatility).")
        elif over >= rules["death"]:
            analysis.append("Phase: Death Overs (Extreme Momentum).")

        analysis.append(
            f"Match Bias: {self._match_bias:.2f}x | "
            f"Risk: {self._wicket_multiplier:.2f}x."
        )
        return analysis

    @staticmethod
    def _dynamic_confidence(
        context: MatchContext,
        raw_runs: float,
        evolved_runs: float,
        range_width: int,
    ) -> tuple[float, dict[str, float | str]]:
        """Estimate per-over confidence from information available at prediction time.

        This is a bounded stability score rather than a probability that the
        prediction will be exactly correct. It deliberately falls during
        volatile phases and when current match rhythm disagrees with history.
        """

        over = context.live.over
        recent_runs = context.live.runs_last_3_overs
        recent_wickets = context.live.wickets_last_3_overs
        observed_overs = max(0, over - 1)

        maturity_bonus = min(0.10, observed_overs * 0.0125)
        if over <= 6:
            phase_penalty = 0.06
            phase_name = "powerplay"
        elif over >= 16:
            phase_penalty = 0.09
            phase_name = "death"
        else:
            phase_penalty = 0.02
            phase_name = "middle"

        if recent_runs > 0:
            recent_rate = recent_runs / min(3, max(1, observed_overs))
            disagreement = abs(raw_runs - recent_rate)
            rhythm_penalty = min(0.14, disagreement * 0.018)
        else:
            disagreement = 0.0
            rhythm_penalty = 0.07 if observed_overs > 0 else 0.10

        wicket_penalty = min(0.08, max(0, recent_wickets) * 0.025)
        range_penalty = min(0.06, max(0, range_width - 2) * 0.02)
        adaptation_penalty = min(0.08, abs(evolved_runs - raw_runs) * 0.015)
        confidence = (
            0.84
            + maturity_bonus
            - (
                phase_penalty
                + rhythm_penalty
                + wicket_penalty
                + range_penalty
                + adaptation_penalty
            )
        )
        confidence = round(min(0.92, max(0.50, confidence)), 2)
        factors: dict[str, float | str] = {
            "method": "dynamic_match_stability_v1",
            "phase": phase_name,
            "maturity_bonus": round(maturity_bonus, 3),
            "phase_penalty": round(phase_penalty, 3),
            "rhythm_disagreement": round(disagreement, 3),
            "rhythm_penalty": round(rhythm_penalty, 3),
            "wicket_penalty": round(wicket_penalty, 3),
            "range_penalty": round(range_penalty, 3),
            "adaptation_penalty": round(adaptation_penalty, 3),
        }
        return confidence, factors

    def predict(self, context: MatchContext) -> PredictionResult:
        current_over = context.live.over
        style = context.format or context.live.match_style
        rules = self._get_format_rules(style)

        # 1. GATEKEEPER
        if self._is_awaiting_actuals and current_over > self._last_predicted_over:
            raise ValueError(
                f"FEED REQUIRED: Over {self._last_predicted_over} missing."
            )

        start = time.perf_counter()

        # 2. CORE ML INFERENCE
        features = self._builder.build(context)
        feature_order = self._repository.get_feature_columns()
        missing = [name for name in feature_order if name not in features]
        if missing:
            raise ValueError(
                "Live feature contract is incomplete; missing required model "
                f"features: {', '.join(missing)}."
            )
        ordered = {name: features[name] for name in feature_order}
        df = pd.DataFrame([ordered])

        for col in self._repository.get_categorical_columns():
            if col in df.columns:
                df[col] = df[col].astype("category")

        raw_runs = float(self._repository.get_runs_model().predict(df)[0])
        raw_wkt_prob = float(
            self._repository.get_wicket_model().predict_proba(df)[0][1]
        )

        # 3. MOMENTUM BLENDING (Centering the range)
        # We blend the AI prediction with the 'Match Rhythm' (Last 3 overs RR)
        recent_runs = context.live.runs_last_3_overs
        if recent_runs > 0:
            match_rhythm = recent_runs / 3.0
            # 50/50 blend of AI history and current Match Momentum
            blended_runs = (raw_runs + match_rhythm) / 2
        else:
            blended_runs = raw_runs

        # 4. EVOLUTION (Apply learned Match Bias)
        evolved_runs = blended_runs * self._match_bias
        evolved_wkt = min(1.0, max(0.0, raw_wkt_prob * self._wicket_multiplier))

        # 5. ULTRA-PRECISION BRACKET
        pivot = round(evolved_runs, 1)

        # Determine the spread based on match phase
        if 7 <= current_over <= 15:
            # Middle Over Precision: ~2 run window
            low_bound = int(pivot - 1.1)
            high_bound = int(pivot + 1.1)
        else:
            # Aggressive Phase: ~3 run window
            low_bound = int(pivot - 1.4)
            high_bound = int(pivot + 1.6)

        low_bound = max(0, low_bound)
        # Ensure range doesn't exceed 3 runs to keep it tight
        if (high_bound - low_bound) > 3:
            high_bound = low_bound + 3

        confidence, confidence_factors = self._dynamic_confidence(
            context,
            raw_runs,
            evolved_runs,
            high_bound - low_bound,
        )

        # Update Internal State
        self._last_predicted_over = current_over
        self._last_point_prediction = evolved_runs
        self._is_awaiting_actuals = True

        latency_ms = (time.perf_counter() - start) * 1000
        analysis = self._build_analysis(
            context, evolved_runs, evolved_wkt, latency_ms, rules
        )

        return PredictionResult(
            predicted_runs=pivot,
            expected_range=f"{low_bound}-{high_bound}",
            wicket_probability=round(evolved_wkt, 3),
            confidence=confidence,
            analysis=analysis,
            metadata={
                "match_bias": f"{self._match_bias:.2f}",
                "momentum_blend": "Active" if recent_runs > 0 else "Inactive",
                "confidence_type": "dynamic stability indicator",
                "confidence_factors": confidence_factors,
                "engine_family": rules["engine_family"],
            },
        )

    def update_actuals(self, actual_runs: int, actual_wickets: int, bowler_name: str):
        if not self._is_awaiting_actuals:
            return "No pending over."

        error = actual_runs - self._last_point_prediction

        # ADAPTIVE LEARNING: Learn 2x faster if the error is significant (>4 runs)
        learning_rate = 0.12 if abs(error) > 4 else 0.06

        # Keep the adaptive correction stable.  The old additive update could
        # make the multiplier negative after a single expensive over.
        self._match_bias = min(
            1.5,
            max(0.5, self._match_bias + (error * learning_rate / 10.0)),
        )

        # Evolution: Wicket Risk Dampener
        if actual_wickets == 0 and self._wicket_multiplier > 0.5:
            self._wicket_multiplier -= 0.05
        elif actual_wickets > 0:
            self._wicket_multiplier += 0.05

        self._bowler_stats[bowler_name] = self._bowler_stats.get(bowler_name, 0) + 1
        self._is_awaiting_actuals = False
        return f"Evolved (LR: {learning_rate})"
