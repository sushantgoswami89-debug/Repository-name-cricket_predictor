"""Win-probability engine, deliberately separate from `PredictionEngine`.

User-requested isolation (2026-08-23): match_winner_v1 (see
docs/finding_match_winner_v1.md) started out wired directly into
`PredictionEngine.predict()`, sharing its call path, return type, and
runtime lifecycle with the run-range and wicket models. That created a
real coupling risk -- a bug or missing artifact in the win-probability
model could crash predictions for the other two, which have nothing to
do with it and are independently validated and already live. This module
extracts it into its own engine with its own result type, so a fault
here can never reach run-range/wicket prediction. `app/live/pipeline.py`
calls this separately (wrapped in its own try/except, mirroring the
existing `BowlerShadowPredictor` isolation pattern) and stores its
output under its own key, never merged into `PredictionResult`.

Uses the exact same validated model/features as before this refactor --
this only changes where the call boundary is, not what gets predicted or
how well it predicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.models.match_context import MatchContext
from runtime_match_winner import MatchWinnerRuntime

MATCH_WINNER_ARTIFACTS = (
    Path(__file__).resolve().parents[3]
    / "models/candidates/match_winner_v1"
)


@dataclass
class MatchWinnerResult:
    """Deliberately its own type, not merged into PredictionResult --
    see this module's docstring for why."""

    win_probability: float
    batting_team: str
    bowling_team: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        if not 0.0 <= self.win_probability <= 1.0:
            raise ValueError("Win probability must be between 0 and 1.")
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "win_probability": self.win_probability,
            "batting_team": self.batting_team,
            "bowling_team": self.bowling_team,
            "metadata": self.metadata,
        }


class MatchWinnerEngine:
    """Standalone engine: its own runtime, its own team-identity
    resolution, its own result type. Deliberately does not share any
    state or lifecycle with PredictionEngine -- construct/call it
    independently."""

    def __init__(self, runtime: MatchWinnerRuntime | None = None) -> None:
        self._runtime = runtime or MatchWinnerRuntime(MATCH_WINNER_ARTIFACTS)

    def predict(self, context: MatchContext) -> MatchWinnerResult:
        current_over = context.live.over
        is_chase = bool(context.live.is_chase)
        batting_team = (
            (context.bowling_first or context.team2)
            if is_chase
            else (context.batting_first or context.team1)
        )
        if batting_team == context.team1:
            batting_team_players, bowling_team_players = context.team1_players, context.team2_players
            bowling_team = context.team2
        else:
            batting_team_players, bowling_team_players = context.team2_players, context.team1_players
            bowling_team = context.team1

        win_probability = self._runtime.predict_win_probability(
            registry=context.metadata.get("registry", {}),
            striker_name=context.live.striker,
            non_striker_name=context.live.non_striker,
            bowler_name=context.live.bowler,
            venue_name=context.venue,
            deliveries=context.metadata.get("deliveries", []),
            over=current_over,
            score_before_over=context.live.score_before_over,
            wkts_down_before_over=context.live.wkts_down_before_over,
            wickets_in_hand=context.live.wickets_in_hand,
            legal_balls_bowled=context.live.legal_balls_bowled,
            balls_remaining=context.live.balls_remaining,
            current_run_rate=context.live.current_run_rate,
            is_chase=is_chase,
            runs_required=context.live.runs_required,
            required_run_rate=context.live.required_run_rate,
            recent_legal_balls=context.live.recent_legal_balls,
            recent_runs_per_ball=context.live.recent_runs_per_ball,
            recent_dot_rate=context.live.recent_dot_rate,
            recent_single_rate=context.live.recent_single_rate,
            recent_boundary_rate=context.live.recent_boundary_rate,
            recent_wicket_rate=context.live.recent_wicket_rate,
            batting_team_players=batting_team_players,
            bowling_team_players=bowling_team_players,
            batting_team_name=batting_team,
            bowling_team_name=bowling_team,
            toss_decision=context.metadata.get("toss_decision", ""),
            batting_team_won_toss=bool(context.metadata.get("batting_team_won_toss", False)),
        )
        result = MatchWinnerResult(
            win_probability=round(win_probability, 3),
            batting_team=batting_team,
            bowling_team=bowling_team,
            metadata={"match_winner_model": self._runtime.artifact_dir.name},
        )
        result.validate()
        return result
