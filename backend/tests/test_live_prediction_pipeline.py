"""Integration tests for verified prediction publication."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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
    assert "Runs: 5-9 (point 7.0)" in message
    assert "Confidence: █████████░ 90% (HIGH)" in message
    assert pipeline.process(_snapshot()) is None
    assert len(publisher.messages) == 1


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
