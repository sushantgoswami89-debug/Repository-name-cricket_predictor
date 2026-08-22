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


def test_resolves_common_first_name_form_via_full_name_fallback():
    """Cricsheet's own identities/players.csv lists bowlers under
    whatever spelling happens to be their Cricsheet convention -- for
    many players that's initials ("JJ Bumrah"), not the common first-name
    form real TOI announcements actually use ("Jasprit Bumrah"). Found
    2026-08-22: every real TOI-style name tried against this predictor
    failed to resolve at all before the full_name-derived fallback was
    added (same root cause as the two feature computers' fix)."""
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
        baseline={"expected_runs": 8.0, "wicket_probability": 0.28},
        announced_bowler="Jasprit Bumrah",
        deliveries=[],
    )

    assert result["status"] == "applied"
    assert result["player_id"] == "player:462411b3"
