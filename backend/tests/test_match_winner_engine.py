from app.ml.match_winner_engine import MatchWinnerEngine
from app.models.live_match_state import LiveMatchState
from app.models.match_context import MatchContext


def _context(is_chase: bool) -> MatchContext:
    return MatchContext(
        match_id="test-match",
        team1="Team Batting",
        team2="Team Bowling",
        team1_players=["Batter One", "Batter Two"],
        team2_players=["Bowler One", "Bowler Two"],
        format="T20",
        live=LiveMatchState(
            over=10,
            wickets_in_hand=7,
            current_run_rate=8.0,
            required_run_rate=9.0,
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
