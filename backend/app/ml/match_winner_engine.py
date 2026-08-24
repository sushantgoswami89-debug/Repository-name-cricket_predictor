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

`predict()`'s team-identity resolution was fixed 2026-08-23 -- see the
inline comment there and docs/finding_win_probability_chase_bug.md for
the full story (a 63-point real-holdout accuracy jump from one bug).

**2026-08-24, revised same day**: a Monte Carlo simulation
(`app/simulation/monte_carlo_win_probability.py`) was built and validated
against the GBM on the 2025+ holdout, beating it on all 3 metrics for IPL
chases specifically -- see
docs/finding_monte_carlo_win_probability_ipl_chase_win.md. It briefly
served as the PRIMARY prediction for IPL chases, but the user explicitly
overrode that: "dont route now, it should check when match is live and
after 5 or 10 matches it should show me summary and then remind me to
pick one." **The GBM is the only served prediction here, unchanged from
before this whole investigation.** For IPL chases, Monte Carlo now runs
alongside purely as a **shadow** (`metadata["monte_carlo_shadow_win_probability"]`),
fault-isolated (a shadow failure never touches the served GBM result) --
mirrors this project's existing `BowlerShadowPredictor` pattern.
`app/live/pipeline.py` logs every (Monte Carlo, GBM) shadow pair via
`WinProbabilityShadowLog` so a real live comparison accumulates across
actual matches; once 5+ real matches have a known outcome, the pipeline
sends a one-time summary reminder through the Telegram publisher
prompting a decision on which model to keep. See
`app/ml/win_probability_shadow_log.py`'s docstring for the full mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.ml.engine_router import EngineFamily, EngineRouter, UnsupportedCricketFormat
from app.models.match_context import MatchContext
from runtime_match_winner import MatchWinnerRuntime
from runtime_monte_carlo_win_probability import MonteCarloWinProbabilityRuntime

MATCH_WINNER_ARTIFACTS = (
    Path(__file__).resolve().parents[3]
    / "models/candidates/match_winner_v1"
)
MONTE_CARLO_ARTIFACTS = (
    Path(__file__).resolve().parents[3]
    / "models/candidates/win_probability_monte_carlo_v1"
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

    def __init__(
        self,
        runtime: MatchWinnerRuntime | None = None,
        monte_carlo_runtime: MonteCarloWinProbabilityRuntime | None = None,
    ) -> None:
        self._runtime = runtime or MatchWinnerRuntime(MATCH_WINNER_ARTIFACTS)
        self._monte_carlo_runtime = monte_carlo_runtime
        if self._monte_carlo_runtime is None:
            try:
                self._monte_carlo_runtime = MonteCarloWinProbabilityRuntime(MONTE_CARLO_ARTIFACTS)
            except Exception:
                # Never let a missing/broken Monte Carlo artifact prevent
                # the engine from constructing -- predict() falls back to
                # the GBM path for every call if this stays None.
                self._monte_carlo_runtime = None

    @staticmethod
    def _is_ipl(context: MatchContext) -> bool:
        try:
            return EngineRouter.resolve(context.format, context.competition).family == EngineFamily.IPL
        except UnsupportedCricketFormat:
            return False

    def predict(self, context: MatchContext) -> MatchWinnerResult:
        current_over = context.live.over
        is_chase = bool(context.live.is_chase)
        # BUG FIXED 2026-08-23 (see docs/finding_win_probability_chase_bug.md):
        # this used to branch on is_chase and fall back to
        # context.batting_first/bowling_first when choosing the batting
        # team -- those two fields are never set anywhere in the codebase
        # (pipeline.py never populates them), so the fallback always fired,
        # and it fell back to context.team2 specifically during every
        # chase. But app/live/pipeline.py always sets context.team1 to
        # whichever team is CURRENTLY BATTING (snapshot.batting_team) and
        # context.team2 to whichever is bowling, for both innings alike --
        # the same convention app/ml/match_winner_dataset.py's training
        # label uses (`batting_team = innings.get("team")`, per-innings,
        # not a fixed match role). So during every second-innings chase in
        # live production, this was silently swapping team identity: the
        # bowling team's name and roster were fed in and labeled as the
        # batting team. Correct fix: context.team1/team2 already ARE the
        # right teams in the right roles, always -- no is_chase branch
        # needed.
        batting_team = context.team1
        bowling_team = context.team2
        batting_team_players, bowling_team_players = context.team1_players, context.team2_players

        win_probability = self._gbm_win_probability(
            context, current_over, is_chase,
            batting_team, bowling_team,
            batting_team_players, bowling_team_players,
        )
        metadata: dict[str, Any] = {"engine": "gbm", "match_winner_model": self._runtime.artifact_dir.name}

        # Monte Carlo shadow, IPL chases only -- logged for live
        # comparison (app/live/pipeline.py), never served. Fault-isolated:
        # a shadow failure only omits the shadow value, never touches the
        # served GBM result above.
        if is_chase and self._monte_carlo_runtime is not None and self._is_ipl(context):
            try:
                metadata["monte_carlo_shadow_win_probability"] = round(
                    self._monte_carlo_runtime.predict_ipl_chase_win_probability(
                        score_before_over=context.live.score_before_over,
                        wickets_in_hand=context.live.wickets_in_hand,
                        legal_balls_bowled=context.live.legal_balls_bowled,
                        balls_remaining=context.live.balls_remaining,
                        runs_required=context.live.runs_required,
                    ),
                    3,
                )
            except Exception:
                metadata["monte_carlo_shadow_win_probability"] = None
                metadata["monte_carlo_shadow_error"] = "shadow_inference_failed"

        result = MatchWinnerResult(
            win_probability=round(win_probability, 3),
            batting_team=batting_team,
            bowling_team=bowling_team,
            metadata=metadata,
        )
        result.validate()
        return result

    def _gbm_win_probability(
        self,
        context: MatchContext,
        current_over: int,
        is_chase: bool,
        batting_team: str,
        bowling_team: str,
        batting_team_players: list[str],
        bowling_team_players: list[str],
    ) -> float:
        return self._runtime.predict_win_probability(
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
