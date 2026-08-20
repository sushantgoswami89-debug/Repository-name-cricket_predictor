"""Build the feature dictionary required by the CricketBaba prediction engine."""

from __future__ import annotations

from typing import Any

from app.ml.historical_feature_store import HistoricalFeatureStore
from app.models.match_context import MatchContext


class FeatureBuilder:
    """Builds model features from MatchContext."""

    def __init__(self, history_store: HistoricalFeatureStore | None = None) -> None:
        self._history_store = history_store or HistoricalFeatureStore()

    @staticmethod
    def _value(obj: object, name: str, default: Any = 0) -> Any:
        """Fetch an attribute, returning the supplied default for None or absence."""
        value = getattr(obj, name, default)
        return default if value is None else value

    def build(self, context: MatchContext) -> dict[str, Any]:
        """Build the 22-feature schema expected by the persisted ML models."""
        live = context.live
        phase = self._value(context, "phase", "")
        if not phase:
            if live.over <= 6:
                phase = "powerplay"
            elif live.over <= 15:
                phase = "middle"
            else:
                phase = "death"

        historical = self._history_store.lookup(
            batsman=live.striker,
            bowler=live.bowler,
            venue=context.venue,
            phase=phase,
            exclude_match_key=context.metadata.get("exclude_match_key"),
        )

        return {
            "over": live.over,
            "score_before_over": live.score_before_over,
            "wkts_down_before_over": live.wkts_down_before_over,
            "wickets_in_hand": live.wickets_in_hand,
            "legal_balls_bowled": live.legal_balls_bowled,
            "balls_remaining": live.balls_remaining,
            "current_run_rate": live.current_run_rate,
            "is_chase": live.is_chase,
            "runs_required": live.runs_required,
            "required_run_rate": live.required_run_rate,
            "recent_legal_balls": live.recent_legal_balls,
            "recent_runs_per_ball": live.recent_runs_per_ball,
            "recent_dot_rate": live.recent_dot_rate,
            "recent_single_rate": live.recent_single_rate,
            "recent_boundary_rate": live.recent_boundary_rate,
            "recent_wicket_rate": live.recent_wicket_rate,
            "balls_faced_before_over": live.balls_faced_before_over,
            "venue_avg_score": historical["venue_avg_score"],
            "bat_career_overs_faced": historical["bat_career_overs_faced"],
            "bat_hist_avg_runs_per_over": historical["bat_hist_avg_runs_per_over"],
            "bat_hist_wicket_rate": historical["bat_hist_wicket_rate"],
            "bowl_career_overs_bowled": historical["bowl_career_overs_bowled"],
            "bowl_hist_avg_runs_conceded": historical["bowl_hist_avg_runs_conceded"],
            "bowl_hist_wicket_rate": historical["bowl_hist_wicket_rate"],
            "h2h_overs": historical["h2h_overs"],
            "h2h_avg_runs": historical["h2h_avg_runs"],
            "bat_vs_bowltype_avg_runs": historical["bat_vs_bowltype_avg_runs"],
            "bat_vs_bowltype_wicket_rate": historical["bat_vs_bowltype_wicket_rate"],
            "bowl_phase_avg_runs": historical["bowl_phase_avg_runs"],
            "phase": phase,
            "batsman_style": historical["batsman_style"],
            "batsman_class": historical["batsman_class"],
            "bowler_type": historical["bowler_type"],
            "bowler_quality": historical["bowler_quality"],
            "pitch_type": historical["pitch_type"],
        }
