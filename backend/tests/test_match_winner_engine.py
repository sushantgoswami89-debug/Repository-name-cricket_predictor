import pytest

from app.ml.match_winner_engine import MatchWinnerEngine
from app.models.live_match_state import LiveMatchState
from app.models.match_context import MatchContext


def _context(is_chase: bool, format: str = "T20", competition: str = "") -> MatchContext:
    return MatchContext(
        match_id="test-match",
        team1="Team Batting",
        team2="Team Bowling",
        team1_players=["Batter One", "Batter Two"],
        team2_players=["Bowler One", "Bowler Two"],
        format=format,
        competition=competition,
        live=LiveMatchState(
            over=10,
            score_before_over=100,
            wkts_down_before_over=3,
            wickets_in_hand=7,
            legal_balls_bowled=60,
            balls_remaining=60,
            current_run_rate=8.0,
            required_run_rate=9.0,
            runs_required=50,
            is_chase=int(is_chase),
        ),
    )


def test_batting_team_resolves_to_team1_when_setting():
    """app/live/pipeline.py always sets context.team1 to whichever team is
    CURRENTLY batting (snapshot.batting_team), for both innings. This must
    hold for the innings-1 (setting) case."""
    result = MatchWinnerEngine().predict(_context(is_chase=False))

    assert result.batting_team == "Team Batting"
    assert result.bowling_team == "Team Bowling"


def test_batting_team_resolves_to_team1_when_chasing():
    """Regression test for the 2026-08-23 bug (see
    docs/finding_win_probability_chase_bug.md): predict() used to fall
    back to context.batting_first/bowling_first (never set anywhere) when
    is_chase was true, which resolved to context.team2 -- the bowling
    team -- for the entire second innings of every live match. context.team1
    is the currently-batting team regardless of is_chase; this must never
    silently swap again."""
    result = MatchWinnerEngine().predict(_context(is_chase=True))

    assert result.batting_team == "Team Batting"
    assert result.bowling_team == "Team Bowling"


def test_gbm_always_serves_the_primary_prediction():
    """2026-08-24, user override: 'dont route now' -- the GBM must remain
    the served prediction everywhere, including IPL chases. Monte Carlo
    only ever appears as a shadow value, never the primary result."""
    for is_chase in (False, True):
        for format_ in ("T20", "T20I", "IPL"):
            result = MatchWinnerEngine().predict(_context(is_chase=is_chase, format=format_))
            assert result.metadata["engine"] == "gbm"
            assert result.metadata["match_winner_model"]


def test_ipl_chase_shadows_monte_carlo_alongside_served_gbm():
    """The served result is the GBM; Monte Carlo runs alongside as a
    logged-only shadow for real live comparison (see
    app/ml/win_probability_shadow_log.py) -- IPL chases only."""
    result = MatchWinnerEngine().predict(_context(is_chase=True, format="IPL"))

    assert result.metadata["engine"] == "gbm"
    shadow = result.metadata["monte_carlo_shadow_win_probability"]
    assert shadow is not None
    assert 0.0 <= shadow <= 1.0
    assert 0.0 <= result.win_probability <= 1.0
    assert "monte_carlo_shadow_error" not in result.metadata


def test_monte_carlo_shadow_absent_outside_ipl_chases():
    """The shadow is IPL-chase-specific -- must not appear for innings-1
    predictions or non-IPL chases, matching the validated scope."""
    for is_chase, format_ in ((False, "IPL"), (True, "T20I"), (False, "T20")):
        result = MatchWinnerEngine().predict(_context(is_chase=is_chase, format=format_))
        assert "monte_carlo_shadow_win_probability" not in result.metadata


def test_monte_carlo_shadow_failure_does_not_affect_served_gbm_result():
    """A broken Monte Carlo shadow call must never take down the served
    GBM prediction -- same fault-isolation standard as everything else
    here."""

    class _BrokenMonteCarloRuntime:
        def predict_ipl_chase_win_probability(self, **kwargs):
            raise RuntimeError("simulated Monte Carlo shadow failure")

    engine = MatchWinnerEngine(monte_carlo_runtime=_BrokenMonteCarloRuntime())
    result = engine.predict(_context(is_chase=True, format="IPL"))

    assert result.metadata["engine"] == "gbm"
    assert result.metadata["monte_carlo_shadow_win_probability"] is None
    assert result.metadata["monte_carlo_shadow_error"] == "shadow_inference_failed"
    assert 0.0 <= result.win_probability <= 1.0


def test_missing_monte_carlo_artifacts_still_serves_gbm():
    """If the Monte Carlo artifacts are missing/broken at construction
    time, the engine must still construct and serve the GBM exactly like
    before this feature existed -- never a hard dependency."""
    import app.ml.match_winner_engine as engine_module

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            engine_module,
            "MonteCarloWinProbabilityRuntime",
            lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no artifacts")),
        )
        engine = MatchWinnerEngine()
        assert engine._monte_carlo_runtime is None
        result = engine.predict(_context(is_chase=True, format="IPL"))
        assert result.metadata["engine"] == "gbm"
        assert "monte_carlo_shadow_win_probability" not in result.metadata
