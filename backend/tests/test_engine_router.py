import pytest

from app.ml.engine_router import EngineFamily, EngineRouter, UnsupportedCricketFormat
from app.ml.prediction_engine import PredictionEngine
from app.models.match_context import MatchContext


def test_t20_routes_to_base_engine() -> None:
    engine = EngineRouter.resolve("T20I")
    assert engine.family is EngineFamily.T20
    assert engine.innings_overs == 20
    assert not engine.extension_features


def test_ipl_inherits_t20_core_and_adds_competition_features() -> None:
    engine = EngineRouter.resolve("T20", "Indian Premier League")
    assert engine.family is EngineFamily.IPL
    assert engine.base_features == EngineRouter.resolve("T20I").base_features
    assert "strategic_timeout_state" in engine.extension_features
    assert "ipl_team_strategy" in engine.extension_features


def test_odi_is_separate_fifty_over_engine() -> None:
    engine = EngineRouter.resolve("ODI")
    assert engine.family is EngineFamily.ODI
    assert engine.innings_overs == 50
    assert engine.death_starts == 41
    assert "old_ball_state" in engine.base_features


def test_test_cricket_is_explicitly_rejected() -> None:
    with pytest.raises(UnsupportedCricketFormat, match="outside"):
        EngineRouter.resolve("Test")


def test_prediction_engine_uses_context_competition_for_ipl_rules() -> None:
    engine = PredictionEngine()

    rules = engine._get_format_rules("T20", "Indian Premier League")

    assert rules["engine_family"] == "ipl"
    context = MatchContext(format="T20", competition="Indian Premier League")
    assert context.competition == "Indian Premier League"
