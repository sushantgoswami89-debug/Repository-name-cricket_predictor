"""
Feature Builder.

Builds the feature dictionary required by the CricketBaba
prediction engine.
"""

from __future__ import annotations

from typing import Any

from app.models.match_context import MatchContext


class FeatureBuilder:
    """
    Builds model features from MatchContext.
    """

    @staticmethod
    def _value(obj: object, name: str, default: Any = 0) -> Any:
        """
        Safely fetch an attribute from an object.

        Returns the supplied default if the attribute does not exist
        or is None.
        """
        value = getattr(obj, name, default)
        return default if value is None else value

    def build(self, context: MatchContext) -> dict[str, Any]:
        """
        Build a dictionary containing all model features.
        """

        live = context.live

        return {
            # Live match state
            "over": live.over,
            "score_before_over": live.score_before_over,
            "wkts_down_before_over": live.wkts_down_before_over,
            "balls_faced_before_over": live.balls_faced_before_over,
            # Historical / contextual values
            "venue_avg_score": self._value(context, "venue_avg_score"),
            "bat_career_overs_faced": self._value(
                context,
                "bat_career_overs_faced",
            ),
            "bat_hist_avg_runs_per_over": self._value(
                context,
                "bat_hist_avg_runs_per_over",
            ),
            "bat_hist_wicket_rate": self._value(
                context,
                "bat_hist_wicket_rate",
            ),
            "bowl_career_overs_bowled": self._value(
                context,
                "bowl_career_overs_bowled",
            ),
            "bowl_hist_avg_runs_conceded": self._value(
                context,
                "bowl_hist_avg_runs_conceded",
            ),
            "bowl_hist_wicket_rate": self._value(
                context,
                "bowl_hist_wicket_rate",
            ),
            "h2h_overs": self._value(context, "h2h_overs"),
            "h2h_avg_runs": self._value(context, "h2h_avg_runs"),
            "bat_vs_bowltype_avg_runs": self._value(
                context,
                "bat_vs_bowltype_avg_runs",
            ),
            "bat_vs_bowltype_wicket_rate": self._value(
                context,
                "bat_vs_bowltype_wicket_rate",
            ),
            "bowl_phase_avg_runs": self._value(
                context,
                "bowl_phase_avg_runs",
            ),
            # Categorical features
            "phase": self._value(context, "phase", ""),
            "batsman_style": self._value(
                context,
                "batsman_style",
                "",
            ),
            "batsman_class": self._value(
                context,
                "batsman_class",
                "",
            ),
            "bowler_type": self._value(
                context,
                "bowler_type",
                "",
            ),
            "bowler_quality": self._value(
                context,
                "bowler_quality",
                "",
            ),
            "pitch_type": self._value(
                context,
                "pitch_type",
                "",
            ),
        }
