"""Connect verified live deliveries to Candidate v3 predictions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol

from app.live.toi_reader import ToiSnapshot
from app.live.verification import LiveDeliveryVerifier
from app.ml.prediction_engine import PredictionEngine
from app.models.live_match_state import LiveMatchState
from app.models.match_context import MatchContext
from app.services.prediction_rating import rate_prediction


class PredictionPublisher(Protocol):
    def publish(self, key: str, text: str) -> bool: ...


class VerifiedLivePredictionPipeline:
    """Publish only predictions derived from fully reconciled deliveries."""

    def __init__(
        self,
        engine: PredictionEngine | None = None,
        publisher: PredictionPublisher | None = None,
        output_file: Path | None = None,
    ) -> None:
        self._engine = engine or PredictionEngine()
        self._publisher = publisher
        self._output = output_file
        self._verifier: LiveDeliveryVerifier | None = None
        self._last_resolved_over: int | None = None
        self._pending_prediction: dict[str, object] | None = None

    def process(self, snapshot: ToiSnapshot) -> dict[str, object] | None:
        if self._verifier is None or self._verifier.innings != snapshot.innings:
            self._verifier = LiveDeliveryVerifier(snapshot.match_id, snapshot.innings)
            self._last_resolved_over = None
            self._pending_prediction = None
        verifier = self._verifier
        new_deliveries = verifier.apply_snapshot(snapshot)
        completed = verifier.completed_over
        if completed is None or completed == self._last_resolved_over:
            return None
        # A prediction is valid only at the over boundary.  If the feed first
        # becomes consistent after the next over has started, do not backfill a
        # forecast using mid-over state and present it as pre-over intelligence.
        if verifier.last_key is None or verifier.last_key[1] != completed:
            return None

        actual_runs = verifier.over_runs[completed]
        actual_wickets = verifier.over_wickets[completed]
        last_delivery = (
            verifier.accepted[verifier.last_key] if verifier.last_key else None
        )
        previous_evaluation: dict[str, object] | None = None
        if self._last_resolved_over is not None:
            if self._pending_prediction is not None:
                previous_evaluation = rate_prediction(
                    predicted_runs=float(self._pending_prediction["expected_runs"]),
                    expected_range=str(self._pending_prediction["expected_range"]),
                    wicket_probability=float(
                        self._pending_prediction["wicket_probability"]
                    ),
                    actual_runs=actual_runs,
                    actual_wickets=actual_wickets,
                ).to_dict()
            self._engine.update_actuals(
                actual_runs,
                actual_wickets,
                last_delivery.bowler if last_delivery else "",
            )
        self._last_resolved_over = completed
        if not snapshot.is_live or completed >= self._innings_limit(
            snapshot.match_format
        ):
            return None

        next_over = completed + 1
        recent_overs = range(max(1, completed - 2), completed + 1)
        recent_runs = sum(verifier.over_runs.get(over, 0) for over in recent_overs)
        recent_wickets = sum(
            verifier.over_wickets.get(over, 0) for over in recent_overs
        )
        legal_deliveries = [
            delivery
            for delivery in verifier.accepted.values()
            if delivery.is_legal
        ]
        recent_deliveries = legal_deliveries[-12:]
        recent_count = len(recent_deliveries)
        recent_delivery_runs = sum(
            delivery.total_runs for delivery in recent_deliveries
        )
        recent_dots = sum(
            delivery.total_runs == 0 for delivery in recent_deliveries
        )
        recent_singles = sum(
            delivery.total_runs == 1 for delivery in recent_deliveries
        )
        recent_boundaries = sum(
            delivery.batter_runs in {4, 6} for delivery in recent_deliveries
        )
        recent_delivery_wickets = sum(
            delivery.wicket_kind is not None for delivery in recent_deliveries
        )
        legal_balls = len(legal_deliveries)
        innings_balls = self._innings_limit(snapshot.match_format) * 6
        balls_remaining = max(0, innings_balls - legal_balls)
        runs_required = (
            max(0, snapshot.target - verifier.score) if snapshot.target else 0
        )
        context = MatchContext(
            match_id=snapshot.match_id,
            team1=snapshot.batting_team,
            team2=snapshot.bowling_team,
            format=snapshot.match_format,
            live=LiveMatchState(
                over=next_over,
                striker=last_delivery.striker if last_delivery else "",
                non_striker="",
                bowler="",
                score_before_over=verifier.score,
                wkts_down_before_over=verifier.wickets,
                balls_faced_before_over=0,
                runs_last_3_overs=recent_runs,
                wickets_last_3_overs=recent_wickets,
                wickets_in_hand=max(0, 10 - verifier.wickets),
                legal_balls_bowled=legal_balls,
                balls_remaining=balls_remaining,
                current_run_rate=(
                    verifier.score * 6 / legal_balls if legal_balls else 0.0
                ),
                is_chase=int(snapshot.target > 0),
                runs_required=runs_required,
                required_run_rate=(
                    runs_required * 6 / balls_remaining
                    if snapshot.target and balls_remaining
                    else 0.0
                ),
                recent_legal_balls=recent_count,
                recent_runs_per_ball=(
                    recent_delivery_runs / recent_count if recent_count else 0.0
                ),
                recent_dot_rate=recent_dots / recent_count if recent_count else 0.0,
                recent_single_rate=(
                    recent_singles / recent_count if recent_count else 0.0
                ),
                recent_boundary_rate=(
                    recent_boundaries / recent_count if recent_count else 0.0
                ),
                recent_wicket_rate=(
                    recent_delivery_wickets / recent_count
                    if recent_count
                    else 0.0
                ),
                match_style=snapshot.match_format,
            ),
        )
        result = self._engine.predict(context)
        prediction = result.to_dict()
        self._pending_prediction = prediction
        output: dict[str, object] = {
            "candidate_version": "v3_verified_live",
            "match_id": snapshot.match_id,
            "innings": snapshot.innings,
            "over": next_over,
            "verified_through": f"{completed}.6",
            "score_before_over": verifier.score,
            "wickets_before_over": verifier.wickets,
            "new_deliveries_verified": len(new_deliveries),
            "prediction": prediction,
            "previous_over_evaluation": previous_evaluation,
        }
        if self._output:
            self._atomic_write(output)
        if self._publisher:
            key = f"{snapshot.match_id}:{snapshot.innings}:{next_over}"
            self._publisher.publish(key, self._telegram_text(output))
        return output

    def _atomic_write(self, output: dict[str, object]) -> None:
        assert self._output is not None
        self._output.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self._output.parent, delete=False
        ) as handle:
            json.dump(output, handle, indent=2)
            temporary = Path(handle.name)
        os.replace(temporary, self._output)

    @staticmethod
    def _telegram_text(output: dict[str, object]) -> str:
        prediction = output["prediction"]
        assert isinstance(prediction, dict)
        probability = float(prediction["wicket_probability"]) * 100
        message = (
            f"🏏 Candidate v3 — verified over {output['over']} prediction\n"
            f"Score: {output['score_before_over']}/{output['wickets_before_over']}\n"
            f"Runs: {prediction['expected_range']} "
            f"(point {prediction['expected_runs']})\n"
            f"Wicket probability: {probability:.1f}%\n"
            f"Confidence: {prediction['confidence_meter']} "
            f"{prediction['confidence_percent']}% "
            f"({prediction['confidence_level']})\n"
            f"State verified through {output['verified_through']}"
        )
        evaluation = output.get("previous_over_evaluation")
        if isinstance(evaluation, dict):
            coverage = "✅ covered" if evaluation["range_covered"] else "❌ missed"
            message += (
                f"\n\nPrevious over review: {evaluation['star_meter']} "
                f"{evaluation['rating']}\n"
                f"Actual: {evaluation['actual_runs']} | "
                f"Error: {evaluation['absolute_run_error']} | "
                f"Range: {coverage}"
            )
        return message

    @staticmethod
    def _innings_limit(match_format: str) -> int:
        return 50 if match_format.upper() == "ODI" else 20
