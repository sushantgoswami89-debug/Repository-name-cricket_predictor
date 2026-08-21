from app.ml.bowler_shadow_predictor import BowlerShadowPredictor
from app.models.live_match_state import LiveMatchState
from app.models.match_context import MatchContext


def test_v3_shadow_predictor_loads_candidate_and_never_publishes():
    result = BowlerShadowPredictor().predict(
        context=MatchContext(
            format="T20",
            live=LiveMatchState(
                over=8,
                wickets_in_hand=8,
                current_run_rate=8.2,
                required_run_rate=9.1,
                recent_wicket_rate=1 / 12,
            ),
        ),
        baseline={
            "expected_runs": 8.0,
            "wicket_probability": 0.28,
        },
        announced_bowler="Arshdeep Singh",
        deliveries=[],
    )

    assert result["status"] == "applied"
    assert result["publishing_enabled"] is False
    assert 6.5 <= result["expected_runs"] <= 9.5
    assert 0 <= result["wicket_probability"] <= 1


def test_unannounced_bowler_keeps_shadow_disabled_without_loading():
    result = BowlerShadowPredictor().predict(
        context=MatchContext(format="T20"),
        baseline={"expected_runs": 8.0, "wicket_probability": 0.28},
        announced_bowler="",
        deliveries=[],
    )

    assert result["status"] == "not_applied"
    assert result["reason"] == "unannounced"
