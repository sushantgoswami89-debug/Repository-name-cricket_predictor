"""Tests for prediction-versus-actual evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.feature_builder import FeatureBuilder
from app.models.match_context import MatchContext
from app.models.match_data import Delivery, Innings, Match, MatchInfo, Over
from app.replay.match_state_verifier import MatchStateVerificationError
from app.replay.evaluation_policy import MatchExcludedError
from app.replay.replay_loader import ReplayLoader
from app.services.prediction_evaluator import PredictionEvaluator


class _RunsModel:
    def predict(self, dataframe: object) -> np.ndarray:
        return np.array([5.5] * len(dataframe))


class _WicketModel:
    def predict_proba(self, dataframe: object) -> np.ndarray:
        return np.array([[0.3, 0.7]] * len(dataframe))


class _Repository:
    def get_feature_columns(self) -> list[str]:
        return list(FeatureBuilder().build(MatchContext()))

    def get_categorical_columns(self) -> list[str]:
        return []

    def get_runs_model(self) -> _RunsModel:
        return _RunsModel()

    def get_wicket_model(self) -> _WicketModel:
        return _WicketModel()


def test_evaluator_keeps_predictions_and_actual_results_together() -> None:
    """Every output row is ready for a side-by-side accuracy check."""

    match = Match(
        info=MatchInfo(match_type="T20", teams=["A", "B"], match_type_number=1),
        innings=[
            Innings(
                team="A",
                overs=[
                    Over(
                        over_number=0,
                        deliveries=[
                            Delivery(
                                "0.1",
                                "Batter",
                                "Bowler",
                                "Partner",
                                {"total": 4},
                            )
                        ],
                    )
                ],
            )
        ],
    )

    result = PredictionEvaluator(repository=_Repository()).evaluate_match(
        match, "match.json"
    )

    assert len(result) == 1
    assert result[0].predicted_runs == 5.5
    assert result[0].is_super_over is False
    assert result[0].rule_exception is False
    assert result[0].rule_exception_codes == ""
    assert result[0].eligibility_scope == "icc_recognized_international"
    assert result[0].actual_runs == 4
    assert result[0].run_error == 1.5
    assert result[0].wicket_probability == 0.7
    assert result[0].predicted_wicket is True
    assert result[0].actual_wicket is False
    assert result[0].confidence == 0.90


class _CorruptReplayLoader(ReplayLoader):
    def frames(self, match: Match, match_id: str = "unknown"):  # type: ignore[no-untyped-def]
        frames = list(super().frames(match, match_id))
        frames[0].state.score = 99
        yield from frames


def test_evaluator_rejects_a_match_when_replay_verification_fails() -> None:
    match = Match(
        info=MatchInfo(match_type="T20", teams=["A", "B"], match_type_number=1),
        innings=[
            Innings(
                team="A",
                overs=[
                    Over(
                        0,
                        [Delivery("0.1", "Batter", "Bowler", "Partner", {"total": 1})],
                    )
                ],
            )
        ],
    )
    evaluator = PredictionEvaluator(
        repository=_Repository(), replay_loader=_CorruptReplayLoader()
    )

    with pytest.raises(MatchStateVerificationError, match="state.score"):
        evaluator.evaluate_match(match, "corrupt.json")


def test_evaluator_excludes_a_known_anomaly_before_prediction() -> None:
    match = Match(
        info=MatchInfo(match_type="T20", teams=["A", "B"], match_type_number=1),
        innings=[],
    )

    with pytest.raises(MatchExcludedError, match="1179017"):
        PredictionEvaluator(repository=_Repository()).evaluate_match(
            match, "1179017.json"
        )


def test_evaluator_rejects_an_unknown_competition_before_prediction() -> None:
    match = Match(
        info=MatchInfo(match_type="T20", teams=["A", "B"], event_name="Friendly"),
        innings=[],
    )

    with pytest.raises(MatchExcludedError, match="not an ICC-recognized"):
        PredictionEvaluator(repository=_Repository()).evaluate_match(
            match, "friendly.json"
        )
