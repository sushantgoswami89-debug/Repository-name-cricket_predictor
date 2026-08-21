"""Integration tests for verified prediction publication."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from app.live.pipeline import VerifiedLivePredictionPipeline
from app.live.toi_reader import ToiDelivery, ToiSnapshot
from app.ml.prediction_result import PredictionResult


class FakeEngine:
    def __init__(self) -> None:
        self.contexts = []
        self.actuals = []

    def predict(self, context):
        self.contexts.append(context)
        return PredictionResult(7.0, 0.2, 0.9, expected_range="5-9")

    def update_actuals(self, *actuals):
        self.actuals.append(actuals)


class FakePublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, key, text):
        self.messages.append((key, text))
        return True


class FakeShadowPredictor:
    def __init__(self, fail: bool = False) -> None:
        self.calls = []
        self.fail = fail

    def predict(self, **values):
        self.calls.append(values)
        if self.fail:
            raise RuntimeError("shadow failure")
        return {
            "status": "applied",
            "candidate_version": "announced_bowler_current_spell_v3",
            "publishing_enabled": False,
            "expected_runs": 6.8,
            "wicket_probability": 0.24,
        }


def _snapshot(is_live: bool = True) -> ToiSnapshot:
    deliveries = tuple(
        ToiDelivery(
            "match",
            1,
            1,
            ball,
            "Bowler",
            "Batter",
            1,
            1,
            {},
            None,
            ball,
            0,
            ball,
            "",
        )
        for ball in range(1, 7)
    )
    return ToiSnapshot("match", "T20", "A", "B", 1, 6, 0, "1.0", is_live, deliveries)


def _two_over_snapshot() -> ToiSnapshot:
    first = _snapshot().deliveries
    second = tuple(
        ToiDelivery(
            "match",
            1,
            2,
            ball,
            "Other Bowler",
            "Batter",
            1,
            1,
            {},
            None,
            6 + ball,
            0,
            10 + ball,
            "",
        )
        for ball in range(1, 7)
    )
    return ToiSnapshot(
        "match", "T20", "A", "B", 1, 12, 0, "2.0", True, first + second
    )


def _empty_snapshot() -> ToiSnapshot:
    return ToiSnapshot("match", "T20", "A", "B", 1, 0, 0, "0.0", True, ())


def test_only_verified_completed_over_is_published() -> None:
    engine = FakeEngine()
    publisher = FakePublisher()
    pipeline = VerifiedLivePredictionPipeline(engine=engine, publisher=publisher)

    output = pipeline.process(_snapshot())

    assert output is not None
    assert output["candidate_version"] == "v3_verified_live"
    assert output["over"] == 2
    assert engine.contexts[0].live.score_before_over == 6
    assert len(publisher.messages) == 1
    message = publisher.messages[0][1]
    assert "Candidate v3 — Over 2 Prediction" in message
    assert "Runs: 5-9 (point 7.0)" in message
    assert "System Health: 🟢 90%" in message
    assert "verified" not in message.lower()
    assert pipeline.process(_snapshot()) is None
    assert len(publisher.messages) == 1


def test_bowler_candidate_is_recorded_in_shadow_but_not_published() -> None:
    shadow = FakeShadowPredictor()
    publisher = FakePublisher()
    snapshot = replace(
        _snapshot(),
        announced_bowler="Other Bowler",
        announced_bowler_over=2,
    )
    pipeline = VerifiedLivePredictionPipeline(
        engine=FakeEngine(),
        publisher=publisher,
        shadow_predictor=shadow,
    )

    output = pipeline.process(snapshot)

    assert output is not None
    assert output["prediction"]["expected_runs"] == 7.0
    assert output["shadow_prediction"]["expected_runs"] == 6.8
    assert output["shadow_prediction"]["publishing_enabled"] is False
    assert shadow.calls[0]["announced_bowler"] == "Other Bowler"
    assert "6.8" not in publisher.messages[0][1]
    assert "24.0%" not in publisher.messages[0][1]


def test_shadow_failure_never_blocks_production_prediction() -> None:
    output = VerifiedLivePredictionPipeline(
        engine=FakeEngine(),
        shadow_predictor=FakeShadowPredictor(fail=True),
    ).process(_snapshot())

    assert output is not None
    assert output["prediction"]["expected_runs"] == 7.0
    assert output["shadow_prediction"]["status"] == "not_applied"
    assert output["shadow_prediction"]["reason"] == "shadow_inference_failed"


def test_chase_features_match_candidate_v3_training_definitions() -> None:
    engine = FakeEngine()
    pipeline = VerifiedLivePredictionPipeline(engine=engine)
    chase = replace(_snapshot(), target=11)

    output = pipeline.process(chase)

    assert output is not None
    live = engine.contexts[0].live
    assert live.wickets_in_hand == 10
    assert live.legal_balls_bowled == 6
    assert live.balls_remaining == 114
    assert live.current_run_rate == 6.0
    assert live.is_chase == 1
    assert live.runs_required == 5
    assert live.required_run_rate == 5 * 6 / 114
    assert live.recent_legal_balls == 6
    assert live.recent_runs_per_ball == 1.0
    assert live.recent_dot_rate == 0.0
    assert live.recent_single_rate == 1.0
    assert live.recent_boundary_rate == 0.0
    assert live.recent_wicket_rate == 0.0


def test_locked_candidate_v3_accepts_complete_live_feature_contract() -> None:
    from app.ml.feature_builder import FeatureBuilder
    from app.ml.model_repository import ModelRepository
    from app.ml.prediction_engine import PredictionEngine
    from app.models.live_match_state import LiveMatchState
    from app.models.match_context import MatchContext

    model_dir = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "locked"
        / "cricketbaba_candidate_v3"
    )
    repository = ModelRepository(model_dir)
    context = MatchContext(
        team1="India",
        team2="Zimbabwe",
        format="T20",
        live=LiveMatchState(
            over=8,
            score_before_over=55,
            wkts_down_before_over=2,
            wickets_in_hand=8,
            legal_balls_bowled=42,
            balls_remaining=78,
            current_run_rate=55 * 6 / 42,
            is_chase=1,
            runs_required=71,
            required_run_rate=71 * 6 / 78,
            recent_legal_balls=12,
            recent_runs_per_ball=1.25,
            recent_dot_rate=0.25,
            recent_single_rate=0.5,
            recent_boundary_rate=0.25,
            recent_wicket_rate=1 / 12,
        ),
    )
    features = FeatureBuilder().build(context)

    assert set(repository.get_feature_columns()) <= set(features)
    result = PredictionEngine(repository=repository).predict(context)
    assert 0.0 <= result.wicket_probability <= 1.0
    assert len(str(result.wicket_probability).split(".")[-1]) <= 3


def test_completed_match_does_not_publish_prediction() -> None:
    publisher = FakePublisher()
    pipeline = VerifiedLivePredictionPipeline(engine=FakeEngine(), publisher=publisher)

    assert pipeline.process(_snapshot(is_live=False)) is None
    assert publisher.messages == []


def test_pre_innings_prediction_is_emitted_once_before_any_delivery() -> None:
    engine = FakeEngine()
    publisher = FakePublisher()
    pipeline = VerifiedLivePredictionPipeline(engine=engine, publisher=publisher)

    output = pipeline.process(_empty_snapshot())

    assert output is not None
    assert output["over"] == 1
    assert output["prediction_basis"] == "pre_innings_context"
    assert pipeline.process(_empty_snapshot()) is None
    assert len(publisher.messages) == 1


def test_completed_chase_does_not_publish_next_over() -> None:
    engine = FakeEngine()
    publisher = FakePublisher()
    pipeline = VerifiedLivePredictionPipeline(engine=engine, publisher=publisher)
    completed = replace(_snapshot(), target=6)

    assert pipeline.process(completed) is None
    assert publisher.messages == []


def test_target_reached_mid_over_stops_prediction() -> None:
    deliveries = _snapshot().deliveries[:3]
    terminal = replace(
        _snapshot(),
        score=3,
        overs="0.3",
        deliveries=deliveries,
        target=3,
    )
    engine = FakeEngine()

    assert VerifiedLivePredictionPipeline(engine=engine).process(terminal) is None
    assert engine.contexts == []


def test_super_over_uses_one_over_limit_and_never_predicts_over_two() -> None:
    super_over = replace(
        _snapshot(),
        innings=3,
        scheduled_overs=1,
        is_super_over=True,
    )
    super_over = replace(
        super_over,
        deliveries=tuple(
            replace(delivery, innings=3) for delivery in super_over.deliveries
        ),
    )
    engine = FakeEngine()

    assert VerifiedLivePredictionPipeline(engine=engine).process(super_over) is None
    assert engine.contexts == []


def test_late_snapshot_does_not_backfill_prediction_after_next_over_starts() -> None:
    publisher = FakePublisher()
    engine = FakeEngine()
    pipeline = VerifiedLivePredictionPipeline(engine=engine, publisher=publisher)
    completed = _snapshot()
    wide = ToiDelivery(
        "match",
        1,
        2,
        1,
        "Bowler",
        "Batter",
        1,
        0,
        {"wides": 1},
        None,
        7,
        0,
        7,
        "Wide",
    )
    late = ToiSnapshot(
        "match",
        "T20",
        "A",
        "B",
        1,
        7,
        0,
        "1.0",
        True,
        completed.deliveries + (wide,),
    )

    assert pipeline.process(late) is None
    assert engine.contexts == []
    assert publisher.messages == []


def test_dynamic_confidence_changes_with_match_stability() -> None:
    from app.ml.prediction_engine import PredictionEngine
    from app.models.live_match_state import LiveMatchState
    from app.models.match_context import MatchContext

    stable_context = MatchContext(
        format="T20",
        live=LiveMatchState(
            over=12,
            runs_last_3_overs=24,
            wickets_last_3_overs=0,
        ),
    )
    volatile_context = MatchContext(
        format="T20",
        live=LiveMatchState(
            over=18,
            runs_last_3_overs=45,
            wickets_last_3_overs=3,
        ),
    )

    stable, stable_factors = PredictionEngine._dynamic_confidence(
        stable_context, raw_runs=8.0, evolved_runs=8.0, range_width=2
    )
    volatile, volatile_factors = PredictionEngine._dynamic_confidence(
        volatile_context, raw_runs=8.0, evolved_runs=12.0, range_width=3
    )

    assert stable > volatile
    assert stable_factors["method"] == "dynamic_match_stability_v1"
    assert volatile_factors["phase"] == "death"


def test_prediction_metadata_exposes_calibration_stages() -> None:
    from app.ml.prediction_engine import PredictionEngine
    from app.models.live_match_state import LiveMatchState
    from app.models.match_context import MatchContext

    result = PredictionEngine().predict(
        MatchContext(
            format="T20",
            live=LiveMatchState(over=8, runs_last_3_overs=24),
        )
    )

    assert result.metadata["centering_correction"] == 0.0
    assert result.metadata["raw_runs"] >= 0
    assert result.metadata["blended_runs"] >= 0
    assert result.metadata["adjusted_runs"] == pytest.approx(
        result.predicted_runs, abs=0.051
    )


def test_wicket_probability_varies_across_distinct_match_states() -> None:
    """Regression guard for a previously reported "flat 50%" live wicket
    display (docs/CTO_HANDOVER_2026-07-24.md). Root cause was traced to
    serving-time feature starvation (see historical_feature_store.py):
    with near-identical input on every call, the model had no situational
    signal to respond to. This asserts three clearly different match
    states produce three genuinely different, non-identical probabilities.
    """

    from app.ml.prediction_engine import PredictionEngine
    from app.models.live_match_state import LiveMatchState
    from app.models.match_context import MatchContext

    states = [
        LiveMatchState(
            over=1,
            striker="RG Sharma",
            bowler="JJ Bumrah",
            score_before_over=0,
            wkts_down_before_over=0,
            wickets_in_hand=10,
        ),
        LiveMatchState(
            over=12,
            striker="MS Dhoni",
            bowler="YS Chahal",
            score_before_over=95,
            wkts_down_before_over=4,
            wickets_in_hand=6,
            balls_faced_before_over=30,
        ),
        LiveMatchState(
            over=19,
            striker="JJ Bumrah",
            bowler="TA Boult",
            score_before_over=165,
            wkts_down_before_over=8,
            wickets_in_hand=2,
            balls_faced_before_over=3,
        ),
    ]
    probabilities = [
        PredictionEngine()
        .predict(MatchContext(team1="India", team2="NZ", venue="Wankhede Stadium", format="T20", live=live))
        .wicket_probability
        for live in states
    ]
    assert len(set(probabilities)) == len(probabilities)
    assert not all(round(p, 1) == 0.5 for p in probabilities)


def test_restart_restores_pending_prediction_and_previous_over_review(
    tmp_path: Path,
) -> None:
    state_file = tmp_path / "live-state.json"
    first = VerifiedLivePredictionPipeline(
        engine=FakeEngine(), state_file=state_file
    )
    assert first.process(_snapshot()) is not None

    restarted_engine = FakeEngine()
    restarted = VerifiedLivePredictionPipeline(
        engine=restarted_engine, state_file=state_file
    )
    output = restarted.process(_two_over_snapshot())

    assert output is not None
    assert output["over"] == 3
    evaluation = output["previous_over_evaluation"]
    assert isinstance(evaluation, dict)
    assert evaluation["actual_runs"] == 6
    assert evaluation["rating"] == "EXCELLENT"
    assert restarted_engine.actuals == [(6, 0, "Other Bowler")]


def test_corrupt_restart_state_is_quarantined(tmp_path: Path) -> None:
    state_file = tmp_path / "live-state.json"
    state_file.write_text("{broken", encoding="utf-8")
    pipeline = VerifiedLivePredictionPipeline(
        engine=FakeEngine(), state_file=state_file
    )

    assert pipeline.process(_snapshot()) is not None
    assert state_file.with_suffix(".json.corrupt").exists()


def test_incompatible_restart_state_is_quarantined(tmp_path: Path) -> None:
    state_file = tmp_path / "live-state.json"
    state_file.write_text('{"schema_version": 999, "scopes": {}}', encoding="utf-8")

    assert VerifiedLivePredictionPipeline(
        engine=FakeEngine(), state_file=state_file
    ).process(_snapshot()) is not None
    assert state_file.with_suffix(".json.corrupt").exists()


def test_state_file_is_match_and_innings_scoped(tmp_path: Path) -> None:
    import json

    state_file = tmp_path / "live-state.json"
    pipeline = VerifiedLivePredictionPipeline(
        engine=FakeEngine(), state_file=state_file
    )
    pipeline.process(_snapshot())
    second = replace(
        _empty_snapshot(), match_id="other", innings=2, target=20
    )
    pipeline.process(second)

    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert set(payload["scopes"]) == {"match:1", "other:2"}


def test_pending_publication_is_persisted_before_send(tmp_path: Path) -> None:
    import json

    class InspectingPublisher(FakePublisher):
        def publish(self, key, text):
            payload = json.loads(state_file.read_text(encoding="utf-8"))
            records = payload["scopes"]["match:1"]["publication_history"]
            assert records[-1]["status"] == "pending"
            return super().publish(key, text)

    state_file = tmp_path / "live-state.json"
    pipeline = VerifiedLivePredictionPipeline(
        engine=FakeEngine(),
        publisher=InspectingPublisher(),
        state_file=state_file,
    )

    pipeline.process(_snapshot())
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["scopes"]["match:1"]["publication_history"][-1]["status"] == "sent"


def test_material_rebase_sends_one_correction_and_preserves_original(
    tmp_path: Path,
) -> None:
    import json

    class CorrectionPublisher(FakePublisher):
        def publish_correction(self, key, text):
            self.messages.append((key, text))
            return True

    state_file = tmp_path / "live-state.json"
    publisher = CorrectionPublisher()
    pipeline = VerifiedLivePredictionPipeline(
        engine=FakeEngine(), publisher=publisher, state_file=state_file
    )
    pipeline.process(_snapshot())
    corrected_deliveries = tuple(
        replace(delivery, total_runs=0, batter_runs=0, feed_total=0)
        for delivery in _snapshot().deliveries
    )
    corrected = replace(_snapshot(), score=0, deliveries=corrected_deliveries)

    output = pipeline.process(corrected)
    assert output is not None
    assert len(publisher.messages) == 2
    assert publisher.messages[-1][1].startswith("Correction\n")

    payload = json.loads(state_file.read_text(encoding="utf-8"))
    history = payload["scopes"]["match:1"]["publication_history"]
    assert [item["status"] for item in history] == ["sent", "sent"]
    assert history[0]["fingerprint"] != history[1]["fingerprint"]
    assert history[1]["correction_reason"]

    assert pipeline.process(corrected) is None
    assert len(publisher.messages) == 2


def test_retryable_publication_is_restored_and_retried(tmp_path: Path) -> None:
    class RetryableError(RuntimeError):
        retryable = True

    class FailingPublisher(FakePublisher):
        def publish(self, key, text):
            raise RetryableError("temporary")

    state_file = tmp_path / "live-state.json"
    first = VerifiedLivePredictionPipeline(
        engine=FakeEngine(), publisher=FailingPublisher(), state_file=state_file
    )
    assert first.process(_snapshot()) is not None

    recovered = FakePublisher()
    restarted = VerifiedLivePredictionPipeline(
        engine=FakeEngine(), publisher=recovered, state_file=state_file
    )
    assert restarted.process(_snapshot()) is None
    assert [key for key, _ in recovered.messages] == ["match:1:2"]


def test_rebase_rebuilds_deliveries_but_retains_pending_review(
    tmp_path: Path,
) -> None:
    state_file = tmp_path / "live-state.json"
    original = VerifiedLivePredictionPipeline(
        engine=FakeEngine(), state_file=state_file
    )
    original.process(_snapshot())

    corrected_first = tuple(
        replace(delivery, total_runs=0, batter_runs=0, feed_total=0)
        for delivery in _snapshot().deliveries
    )
    corrected_second = tuple(
        replace(
            delivery,
            feed_total=ball,
            timestamp_ms=20 + ball,
        )
        for ball, delivery in enumerate(_two_over_snapshot().deliveries[6:], start=1)
    )
    corrected = ToiSnapshot(
        "match",
        "T20",
        "A",
        "B",
        1,
        6,
        0,
        "2.0",
        True,
        corrected_first + corrected_second,
    )
    rebased = VerifiedLivePredictionPipeline(
        engine=FakeEngine(),
        state_file=state_file,
        restore_verified_state=False,
    )

    output = rebased.process(corrected)

    assert output is not None
    evaluation = output["previous_over_evaluation"]
    assert isinstance(evaluation, dict)
    assert evaluation["actual_runs"] == 6
