"""
Prediction Engine v3 (Precision & Momentum Update).
Features: Adaptive Learning Rate, Match Rhythm Blending, and Ultra-Tight Bracketing.
"""

from __future__ import annotations

import time

import pandas as pd

from app.ml.bowler_spell_adjuster import BowlerSpellAdjuster
from app.ml.engine_router import EngineFamily, EngineRouter
from app.ml.feature_builder import FeatureBuilder
from app.ml.model_repository import ModelRepository
from app.ml.prediction_result import PredictionResult
from app.models.match_context import MatchContext


class PredictionEngine:
    # The +1.0 correction here was tuned against the pre-fix model, which had
    # no batter/bowler identity signal (see historical_feature_store.py) and
    # under-predicted as a result. With real features flowing, a 20-match
    # IPL replay grid search (tune_momentum_blend.py) found the raw model
    # now runs close to unbiased (measured mean signed error ~-0.34 runs/over,
    # within noise of a flat MAE plateau from -0.5 to 0.0), so the fixed
    # additive correction is dropped; the adaptive match_bias multiplier
    # below handles residual per-match drift.
    RUN_CENTERING_CORRECTION = 0.0

    def __init__(
        self,
        repository: ModelRepository | None = None,
        feature_builder: FeatureBuilder | None = None,
        momentum_blend_weight: float = 0.3,
        run_centering_correction: float | None = None,
    ) -> None:
        self._repository = repository or ModelRepository()
        self._builder = feature_builder or FeatureBuilder()
        self._momentum_blend_weight = momentum_blend_weight
        self._run_centering_correction = (
            run_centering_correction
            if run_centering_correction is not None
            else self.RUN_CENTERING_CORRECTION
        )
        self._bowler_adjuster = BowlerSpellAdjuster()

        # --- STATE MANAGEMENT ---
        self._match_bias = 1.0
        self._wicket_multiplier = 1.0
        self._last_predicted_over = -1
        self._is_awaiting_actuals = False
        self._last_point_prediction = 0.0

        # --- TRACKERS ---
        self._bowler_stats = {}
        self._error_history = []

    def export_runtime_state(self) -> dict[str, object]:
        """Return the bounded adaptive state needed for a safe restart."""
        return {
            "match_bias": self._match_bias,
            "wicket_multiplier": self._wicket_multiplier,
            "last_predicted_over": self._last_predicted_over,
            "is_awaiting_actuals": self._is_awaiting_actuals,
            "last_point_prediction": self._last_point_prediction,
            "bowler_stats": dict(self._bowler_stats),
        }

    def reset_runtime_state(self) -> None:
        """Clear match-adaptive state before rebuilding from corrected facts."""
        self._match_bias = 1.0
        self._wicket_multiplier = 1.0
        self._last_predicted_over = -1
        self._is_awaiting_actuals = False
        self._last_point_prediction = 0.0
        self._bowler_stats = {}
        self._error_history = []

    def restore_runtime_state(self, state: dict[str, object]) -> None:
        """Restore validated adaptive state from live persistence."""
        match_bias = float(state.get("match_bias", 1.0))
        wicket_multiplier = float(state.get("wicket_multiplier", 1.0))
        if not 0.5 <= match_bias <= 1.5:
            raise ValueError("Persisted match bias is outside safe bounds.")
        if not 0.0 <= wicket_multiplier <= 2.0:
            raise ValueError("Persisted wicket multiplier is outside safe bounds.")
        self._match_bias = match_bias
        self._wicket_multiplier = wicket_multiplier
        self._last_predicted_over = int(state.get("last_predicted_over", -1))
        self._is_awaiting_actuals = bool(state.get("is_awaiting_actuals", False))
        self._last_point_prediction = float(
            state.get("last_point_prediction", 0.0)
        )
        bowler_stats = state.get("bowler_stats", {})
        if not isinstance(bowler_stats, dict):
            raise ValueError("Persisted bowler stats must be an object.")
        self._bowler_stats = {
            str(name): int(value) for name, value in bowler_stats.items()
        }

    def _get_format_rules(self, style: str, competition: str = ""):
        specification = EngineRouter.resolve(style, competition)
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
        rules = self._get_format_rules(style, context.competition)

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
        calibrators = self._repository.get_wicket_calibrator()
        if calibrators is not None:
            calibrator = calibrators.get(features["phase"]) if isinstance(
                calibrators, dict
            ) else calibrators
            if calibrator is not None:
                raw_wkt_prob = float(calibrator.predict([raw_wkt_prob])[0])

        # 3. MOMENTUM BLENDING (Centering the range)
        # We blend the AI prediction with the 'Match Rhythm' (Last 3 overs RR)
        recent_runs = context.live.runs_last_3_overs
        if recent_runs > 0:
            observed_recent_overs = min(3, max(1, current_over - 1))
            match_rhythm = recent_runs / observed_recent_overs
            w = self._momentum_blend_weight
            blended_runs = (1 - w) * raw_runs + w * match_rhythm
        else:
            blended_runs = raw_runs

        # 4. EVOLUTION (Apply learned Match Bias)
        centered_runs = blended_runs + self._run_centering_correction
        evolved_runs = centered_runs * self._match_bias
        evolved_wkt = min(1.0, max(0.0, raw_wkt_prob * self._wicket_multiplier))

        # 4b. BOWLER SPELL ADJUSTMENT (announced_bowler_current_spell_v3)
        # Independently validated against the production baseline (see
        # bowler_spell_adjuster.py docstring): Run MAE 3.87->3.75, Wicket
        # Brier 0.193->0.190, Wicket AUC 0.519->0.550. Applies whenever the
        # current bowler is known, regardless of whether that came from an
        # advance live announcement or simply being the actual bowler in a
        # replay -- it does not require prediction-before-the-over to add
        # value, only bowler identity.
        bowler_adjustment = self._bowler_adjuster.adjust(
            bowler=context.live.bowler,
            over=current_over,
            phase=features["phase"],
            wickets_in_hand=context.live.wickets_in_hand,
            recent_wicket_rate=context.live.recent_wicket_rate,
            current_run_rate=context.live.current_run_rate,
            required_run_rate=context.live.required_run_rate,
            base_runs=evolved_runs,
            base_wicket_probability=evolved_wkt,
        )
        if bowler_adjustment is not None:
            evolved_runs = bowler_adjustment.runs
            evolved_wkt = bowler_adjustment.wicket_probability

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
        raw_confidence = confidence
        confidence_calibrator = self._repository.get_confidence_calibrator()
        if confidence_calibrator is not None:
            confidence = round(
                float(confidence_calibrator.predict([confidence])[0]), 3
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
                "raw_runs": round(raw_runs, 3),
                "match_rhythm": (
                    round(match_rhythm, 3) if recent_runs > 0 else None
                ),
                "blended_runs": round(blended_runs, 3),
                "centering_correction": self.RUN_CENTERING_CORRECTION,
                "adjusted_runs": round(evolved_runs, 3),
                "display_runs": pivot,
                "bowler_adjustment_applied": bowler_adjustment is not None,
                "confidence_type": "dynamic stability indicator",
                "raw_confidence": round(raw_confidence, 3),
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
        self._bowler_adjuster.record_completed_over(
            bowler=bowler_name,
            over=self._last_predicted_over,
            actual_runs=actual_runs,
            actual_wickets=actual_wickets,
        )
        self._is_awaiting_actuals = False
        return f"Evolved (LR: {learning_rate})"
