"""Connect verified live deliveries to Candidate v3 predictions."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol

from app.live.persistence import LiveStateStore
from app.live.toi_reader import ToiDelivery, ToiSnapshot
from app.live.verification import LiveDeliveryVerifier, VerificationError
from app.ml.engine_router import EngineRouter
from app.ml.bowler_shadow_predictor import BowlerShadowPredictor
from app.ml.prediction_engine import PredictionEngine
from app.models.live_match_state import LiveMatchState
from app.models.match_context import MatchContext
from app.services.prediction_rating import rate_prediction


class PredictionPublisher(Protocol):
    def publish(self, key: str, text: str) -> bool: ...


def _team_player_names(snapshot: ToiSnapshot, team_name: str) -> list[str]:
    # team_players is best-effort (see ToiSnapshot's own docstring -- a
    # missing/malformed block degrades to the empty default rather than
    # raising), so a team with no roster fetched yet just yields an empty
    # list -- team-composition features degrade to zero counts, not a crash.
    return [player.name for player in snapshot.team_players.get(team_name, ())]


class VerifiedLivePredictionPipeline:
    """Publish only predictions derived from fully reconciled deliveries."""

    def __init__(
        self,
        engine: PredictionEngine | None = None,
        publisher: PredictionPublisher | None = None,
        output_file: Path | None = None,
        state_file: Path | None = None,
        restore_verified_state: bool = True,
        shadow_predictor: BowlerShadowPredictor | None = None,
        publish_bowler_shadow: bool = False,
    ) -> None:
        self._engine = engine or PredictionEngine()
        self._shadow_predictor = shadow_predictor or BowlerShadowPredictor()
        # announced_bowler_current_spell_v3 is validated on a proper
        # 2025-2026 holdout (every promotion gate passes -- see
        # models/candidates/announced_bowler_current_spell_v3/validation_report.json)
        # but its own decision field says "research_pass_await_live_availability":
        # the one thing never confirmed is whether TOI actually
        # pre-announces the next bowler often enough, and early enough,
        # to be worth using live (see monitor_announced_bowler.py /
        # docs/live_match_test_checklist.md item #1). Defaults to False so
        # nothing changes until that's actually been checked against a
        # real match -- flip explicitly (run_toi_live.py --publish-bowler-shadow)
        # once it has.
        self._publish_bowler_shadow = publish_bowler_shadow
        self._publisher = publisher
        self._output = output_file
        self._verifier: LiveDeliveryVerifier | None = None
        self._last_resolved_over: int | None = None
        self._pending_prediction: dict[str, object] | None = None
        self._pending_over: int | None = None
        self._review_history: list[dict[str, object]] = []
        self._publication_history: list[dict[str, object]] = []
        resolved_state_file = state_file
        if resolved_state_file is None and output_file is not None:
            resolved_state_file = output_file.with_suffix(".state.json")
        self._state_store = (
            LiveStateStore(resolved_state_file) if resolved_state_file else None
        )
        self._restored_scope: tuple[str, int] | None = None
        self._published_keys: set[str] = set()
        self._restore_verified_state = restore_verified_state

    def process(self, snapshot: ToiSnapshot) -> dict[str, object] | None:
        scope = (snapshot.match_id, snapshot.innings)
        if self._restored_scope != scope:
            self._verifier = None
            self._last_resolved_over = None
            self._pending_prediction = None
            self._pending_over = None
            self._review_history = []
            self._publication_history = []
            self._published_keys = set()
            self._restore(snapshot)
            self._restored_scope = scope
            self._retry_pending_publications(snapshot)
        if self._verifier is None or self._verifier.innings != snapshot.innings:
            self._verifier = LiveDeliveryVerifier(snapshot.match_id, snapshot.innings)
        verifier = self._verifier
        try:
            new_deliveries = verifier.apply_snapshot(snapshot)
        except VerificationError:
            # A fully reconciled snapshot is authoritative after a score
            # correction. Rebuild rather than carrying stale delivery/engine
            # state into the next prediction.
            verifier = LiveDeliveryVerifier(snapshot.match_id, snapshot.innings)
            new_deliveries = verifier.apply_snapshot(snapshot)
            self._verifier = verifier
            self._last_resolved_over = (
                (verifier.completed_over or 1) - 1
                if verifier.completed_over is not None
                else None
            )
            self._pending_prediction = None
            self._pending_over = None
            if hasattr(self._engine, "reset_runtime_state"):
                self._engine.reset_runtime_state()
        self._persist(snapshot)
        if self._is_complete(snapshot, verifier):
            self._persist(snapshot)
            return None
        completed = verifier.completed_over
        if (
            completed is None
            and not verifier.accepted
            and snapshot.is_live
            and self._last_resolved_over is None
        ):
            return self._pre_innings_prediction(snapshot)
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
        if (
            self._pending_prediction is not None
            and self._pending_over == completed
        ):
            previous_evaluation = rate_prediction(
                predicted_runs=float(self._pending_prediction["expected_runs"]),
                expected_range=str(self._pending_prediction["expected_range"]),
                wicket_probability=float(
                    self._pending_prediction["wicket_probability"]
                ),
                actual_runs=actual_runs,
                actual_wickets=actual_wickets,
            ).to_dict()
            self._record_review(
                snapshot, completed, self._pending_prediction, previous_evaluation
            )
        if self._last_resolved_over is not None:
            self._engine.update_actuals(
                actual_runs,
                actual_wickets,
                last_delivery.bowler if last_delivery else "",
            )
        self._last_resolved_over = completed
        next_over = completed + 1
        announced_bowler = (
            snapshot.announced_bowler
            if snapshot.announced_bowler_over == next_over
            else ""
        )
        recent_overs = range(max(1, completed - 2), completed + 1)
        recent_runs = sum(verifier.over_runs.get(over, 0) for over in recent_overs)
        recent_wickets = sum(
            verifier.over_wickets.get(over, 0) for over in recent_overs
        )
        all_deliveries = list(verifier.accepted.values())
        legal_deliveries = [
            delivery
            for delivery in all_deliveries
            if delivery.is_legal
        ]
        recent_deliveries: list[ToiDelivery] = []
        legal_in_window = 0
        for delivery in reversed(all_deliveries):
            recent_deliveries.append(delivery)
            legal_in_window += int(delivery.is_legal)
            if legal_in_window == 12:
                break
        recent_deliveries.reverse()
        recent_count = legal_in_window
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
        innings_balls = self._innings_limit_for(snapshot) * 6
        balls_remaining = max(0, innings_balls - legal_balls)
        runs_required = (
            max(0, snapshot.target - verifier.score) if snapshot.target else 0
        )
        context = MatchContext(
            match_id=snapshot.match_id,
            team1=snapshot.batting_team,
            team2=snapshot.bowling_team,
            team1_players=_team_player_names(snapshot, snapshot.batting_team),
            team2_players=_team_player_names(snapshot, snapshot.bowling_team),
            format=snapshot.match_format,
            competition=snapshot.competition,
            live=LiveMatchState(
                over=next_over,
                striker=last_delivery.striker if last_delivery else "",
                non_striker="",
                bowler=announced_bowler,
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
            # Threads the current innings' verified deliveries through to
            # PredictionEngine's wicket model (contract22_wicket_v2_batter_state
            # needs these for partnership-age/new-batter tracking -- see
            # WicketContract22FeatureComputer). No registry key: genuine
            # live TOI has no Cricsheet-style registry, so player identity
            # resolution falls back to the name-alias snapshot, same as
            # everywhere else this session handles that gap.
            metadata={"deliveries": all_deliveries},
        )
        result = self._engine.predict(context)
        prediction = result.to_dict()
        try:
            shadow = self._shadow_predictor.predict(
                context=context,
                baseline=prediction,
                announced_bowler=announced_bowler,
                deliveries=all_deliveries,
            )
        except Exception as exc:
            shadow = {
                "status": "not_applied",
                "candidate_version": "announced_bowler_current_spell_v3",
                "publishing_enabled": False,
                "reason": "shadow_inference_failed",
                "detail": type(exc).__name__,
            }
        if self._publish_bowler_shadow and shadow.get("status") == "applied":
            self._apply_bowler_shadow(prediction, shadow)
        metadata = prediction.get("metadata")
        if isinstance(metadata, dict):
            metadata["bowler_shadow"] = shadow
        self._pending_prediction = prediction
        self._pending_over = next_over
        output: dict[str, object] = {
            "candidate_version": "v3_verified_live",
            "match_id": snapshot.match_id,
            "innings": snapshot.innings,
            "over": next_over,
            "verified_through": f"{completed}.6",
            "score_before_over": verifier.score,
            "wickets_before_over": verifier.wickets,
            "new_deliveries_verified": len(new_deliveries),
            "bowler": announced_bowler,
            "bowler_source": (
                "scoreboard_pre_over" if announced_bowler else "unannounced"
            ),
            "prediction": prediction,
            "shadow_prediction": shadow,
            "previous_over_evaluation": previous_evaluation,
        }
        if self._output:
            self._atomic_write(output)
        self._persist(snapshot)
        self._publish_output(snapshot, next_over, output)
        self._persist(snapshot)
        return output

    def _pre_innings_prediction(
        self, snapshot: ToiSnapshot
    ) -> dict[str, object]:
        innings_balls = self._innings_limit_for(snapshot) * 6
        context = MatchContext(
            match_id=snapshot.match_id,
            team1=snapshot.batting_team,
            team2=snapshot.bowling_team,
            team1_players=_team_player_names(snapshot, snapshot.batting_team),
            team2_players=_team_player_names(snapshot, snapshot.bowling_team),
            format=snapshot.match_format,
            competition=snapshot.competition,
            live=LiveMatchState(
                over=1,
                wickets_in_hand=10,
                balls_remaining=innings_balls,
                is_chase=int(snapshot.target > 0),
                runs_required=snapshot.target,
                required_run_rate=(
                    snapshot.target * 6 / innings_balls if snapshot.target else 0.0
                ),
                match_style=snapshot.match_format,
            ),
        )
        prediction = self._engine.predict(context).to_dict()
        self._pending_prediction = prediction
        self._pending_over = 1
        self._last_resolved_over = 0
        output: dict[str, object] = {
            "candidate_version": "v3_verified_live",
            "match_id": snapshot.match_id,
            "innings": snapshot.innings,
            "over": 1,
            "score_before_over": 0,
            "wickets_before_over": 0,
            "new_deliveries_verified": 0,
            "prediction": prediction,
            "previous_over_evaluation": None,
            "prediction_basis": "pre_innings_context",
        }
        if self._output:
            self._atomic_write(output)
        self._persist(snapshot)
        self._publish_output(snapshot, 1, output)
        self._persist(snapshot)
        return output

    @staticmethod
    def _apply_bowler_shadow(prediction: dict[str, object], shadow: dict[str, object]) -> None:
        """Overwrite the baseline prediction in place with the announced-
        bowler-adjusted numbers. Only called when publish_bowler_shadow is
        explicitly enabled and the shadow predictor actually produced an
        adjustment (status == "applied") -- see the constructor docstring
        for why this defaults off."""
        run_shift = float(shadow["expected_runs"]) - float(shadow["baseline_expected_runs"])
        predicted_runs = round(float(prediction["expected_runs"]) + run_shift, 1)
        prediction["expected_runs"] = predicted_runs
        try:
            low_str, high_str = str(prediction["expected_range"]).split("-")
            new_low = round(float(low_str) + run_shift)
            new_high = round(float(high_str) + run_shift)
            prediction["expected_range"] = f"{new_low}-{new_high}"
        except (ValueError, KeyError):
            pass  # Malformed range string -- leave it, the run/wicket fields still update.
        prediction["wicket_probability"] = round(float(shadow["wicket_probability"]), 3)
        metadata = prediction.get("metadata")
        if isinstance(metadata, dict):
            metadata["display_runs"] = predicted_runs
        shadow["status"] = "published"

    def _restore(self, snapshot: ToiSnapshot) -> None:
        if self._state_store is None:
            return
        payload = self._state_store.load(snapshot.match_id, snapshot.innings)
        if payload is None:
            return
        try:
            if self._restore_verified_state:
                verifier = LiveDeliveryVerifier(snapshot.match_id, snapshot.innings)
                for item in payload.get("accepted_deliveries", []):
                    if not isinstance(item, dict):
                        raise ValueError("Invalid persisted delivery.")
                    verifier.apply(ToiDelivery(**item))
                self._verifier = verifier
            last = payload.get("last_resolved_over")
            self._last_resolved_over = int(last) if last is not None else None
            pending = payload.get("pending_prediction")
            self._pending_prediction = pending if isinstance(pending, dict) else None
            pending_over = payload.get("pending_over")
            self._pending_over = (
                int(pending_over) if pending_over is not None else None
            )
            published = payload.get("published_keys", [])
            if not isinstance(published, list):
                raise ValueError("Invalid persisted publication keys.")
            self._published_keys = {str(value) for value in published}
            reviews = payload.get("review_history", [])
            publications = payload.get("publication_history", [])
            if not isinstance(reviews, list) or not isinstance(publications, list):
                raise ValueError("Invalid persisted audit history.")
            self._review_history = [
                value for value in reviews if isinstance(value, dict)
            ]
            self._publication_history = [
                value for value in publications if isinstance(value, dict)
            ]
            if self._publisher and hasattr(self._publisher, "restore_published"):
                self._publisher.restore_published(self._published_keys)
            engine_state = payload.get("engine_state")
            if (
                self._restore_verified_state
                and isinstance(engine_state, dict)
                and hasattr(
                self._engine, "restore_runtime_state"
                )
            ):
                self._engine.restore_runtime_state(engine_state)
        except (TypeError, ValueError):
            self._state_store.quarantine()
            self._verifier = None
            self._last_resolved_over = None
            self._pending_prediction = None
            self._pending_over = None
            self._published_keys = set()
            self._review_history = []
            self._publication_history = []

    def _retry_pending_publications(self, snapshot: ToiSnapshot) -> None:
        if self._publisher is None:
            return
        for record in self._publication_history:
            if record.get("status") not in {"pending", "failed_retryable"}:
                continue
            key = str(record.get("publication_key", ""))
            text = str(record.get("text", ""))
            if not key or not text:
                continue
            try:
                result = self._publisher.publish(key, text)
            except Exception as exc:
                record["status"] = (
                    "failed_retryable"
                    if getattr(exc, "retryable", True)
                    else "failed_permanent"
                )
                record["error"] = str(exc)
                continue
            if bool(getattr(result, "sent", result)):
                record["status"] = "sent"
                record["telegram_message_id"] = getattr(
                    result, "message_id", None
                )
                self._published_keys.add(str(record["identity"]))
        self._persist(snapshot)

    def _persist(self, snapshot: ToiSnapshot) -> None:
        if self._state_store is None or self._verifier is None:
            return
        engine_state: dict[str, object] = {}
        if hasattr(self._engine, "export_runtime_state"):
            engine_state = self._engine.export_runtime_state()
        self._state_store.save(
            {
                "match_id": snapshot.match_id,
                "innings": snapshot.innings,
                "accepted_deliveries": [
                    asdict(delivery)
                    for delivery in self._verifier.accepted.values()
                ],
                "last_resolved_over": self._last_resolved_over,
                "pending_prediction": self._pending_prediction,
                "pending_over": self._pending_over,
                "published_keys": sorted(self._published_keys),
                "review_history": self._bounded_history(self._review_history),
                "publication_history": self._bounded_history(
                    self._publication_history
                ),
                "engine_state": engine_state,
            }
        )

    def _record_review(
        self,
        snapshot: ToiSnapshot,
        over: int,
        prediction: dict[str, object],
        evaluation: dict[str, object],
    ) -> None:
        actual = {
            "runs": self._verifier.over_runs[over],
            "wickets": self._verifier.over_wickets[over],
        }
        fingerprint = self._fingerprint({"prediction": prediction, "actual": actual})
        identity = f"{snapshot.match_id}:{snapshot.innings}:{over}"
        prior = [
            item for item in self._review_history if item.get("identity") == identity
        ]
        if prior and prior[-1].get("fingerprint") == fingerprint:
            return
        self._review_history.append(
            {
                "identity": identity,
                "version": len(prior) + 1,
                "fingerprint": fingerprint,
                "prediction": prediction,
                "actual": actual,
                "evaluation": evaluation,
                "status": "resolved",
                "timestamp": self._now(),
                "correction_reason": (
                    "verified delivery rebase" if prior else None
                ),
            }
        )

    def _publish_output(
        self, snapshot: ToiSnapshot, over: int, output: dict[str, object]
    ) -> None:
        if self._publisher is None:
            return
        identity = f"{snapshot.match_id}:{snapshot.innings}:{over}"
        text = self._telegram_text(output)
        fingerprint = self._fingerprint({"identity": identity, "text": text})
        sent = [
            item
            for item in self._publication_history
            if item.get("identity") == identity and item.get("status") == "sent"
        ]
        if sent and sent[-1].get("fingerprint") == fingerprint:
            self._published_keys.add(identity)
            return
        is_correction = bool(sent)
        key = (
            f"{identity}:correction:{fingerprint[:12]}"
            if is_correction
            else identity
        )
        record: dict[str, object] = {
            "identity": identity,
            "publication_key": key,
            "fingerprint": fingerprint,
            "timestamp": self._now(),
            "status": "pending",
            "telegram_message_id": None,
            "correction_reason": (
                "material prediction or review change after verified rebase"
                if is_correction
                else None
            ),
            "text": text,
        }
        self._publication_history.append(record)
        self._persist(snapshot)
        publish = (
            self._publisher.publish_correction
            if is_correction and hasattr(self._publisher, "publish_correction")
            else self._publisher.publish
        )
        try:
            result = publish(key, f"Correction\n{text}" if is_correction else text)
        except Exception as exc:
            record["status"] = (
                "failed_retryable"
                if getattr(exc, "retryable", True)
                else "failed_permanent"
            )
            record["error"] = str(exc)
            return
        was_sent = bool(getattr(result, "sent", result))
        if was_sent:
            record["status"] = "sent"
            record["telegram_message_id"] = getattr(result, "message_id", None)
            self._published_keys.add(identity)

    @staticmethod
    def _fingerprint(value: object) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _bounded_history(
        history: list[dict[str, object]], limit: int = 500
    ) -> list[dict[str, object]]:
        if len(history) <= limit:
            return history
        unresolved = [
            item for item in history if item.get("status") not in {"sent", "resolved"}
        ]
        required_keys = {
            str(item.get("identity"))
            for item in history
            if item.get("status") == "sent"
        }
        newest_by_identity: dict[str, dict[str, object]] = {}
        for item in history:
            if str(item.get("identity")) in required_keys:
                newest_by_identity[str(item.get("identity"))] = item
        protected_ids = {
            id(item)
            for item in unresolved + list(newest_by_identity.values())
        }
        resolved = [item for item in history if id(item) not in protected_ids]
        keep = max(0, limit - len(protected_ids))
        return resolved[-keep:] + [
            item for item in history if id(item) in protected_ids
        ]

    def _is_complete(
        self, snapshot: ToiSnapshot, verifier: LiveDeliveryVerifier
    ) -> bool:
        legal_balls = sum(delivery.is_legal for delivery in verifier.accepted.values())
        return (
            snapshot.match_complete
            or snapshot.innings_complete
            or not snapshot.is_live
            or verifier.wickets >= 10
            or (snapshot.target > 0 and verifier.score >= snapshot.target)
            or legal_balls >= self._innings_limit_for(snapshot) * 6
        )

    @classmethod
    def _innings_limit_for(cls, snapshot: ToiSnapshot) -> int:
        if snapshot.is_super_over or snapshot.innings > 2:
            return 1
        if snapshot.scheduled_overs > 0:
            return snapshot.scheduled_overs
        return cls._innings_limit(snapshot.match_format, snapshot.competition)

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
        health_color = {
            "HIGH": "🟢",
            "MEDIUM": "🟡",
            "LOW": "🔴",
        }.get(str(prediction["confidence_level"]), "🟡")
        message = (
            f"🏏 Candidate v3 — Over {output['over']} Prediction\n"
            f"Score: {output['score_before_over']}/{output['wickets_before_over']}\n"
            f"Runs: {prediction['expected_range']} "
            f"(point {prediction['expected_runs']})\n"
            f"Wicket risk (phase baseline): {probability:.1f}%\n"
            f"System Health: {health_color} "
            f"{prediction['confidence_percent']}%"
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
    def _innings_limit(match_format: str, competition: str = "") -> int:
        return EngineRouter.resolve(match_format, competition).innings_overs
