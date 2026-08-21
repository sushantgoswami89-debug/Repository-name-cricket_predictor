"""Build live match-state snapshots by replaying parsed innings data."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

from app.models.live_match_state import LiveMatchState
from app.models.match_data import Delivery, Innings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    """The state before an over together with that over's eventual outcome."""

    state: LiveMatchState
    actual_runs: int
    actual_wickets: int


class MatchReplay:
    """Replay an innings to produce prediction-ready live-state snapshots."""

    _NON_DISMISSAL_KINDS = frozenset({"retired hurt"})

    def frames(self, innings: Innings) -> Iterator[ReplayFrame]:
        """Yield one live-state frame before each non-empty over.

        ``LiveMatchState.over`` is one-based because it is supplied directly
        to the prediction model.  Cricsheet's over numbers are zero-based.
        """

        score = 0
        wickets = 0
        balls_faced: dict[str, int] = {}
        legal_balls = 0
        recent_overs: list[tuple[int, int]] = []
        recent_deliveries: list[Delivery] = []

        for over in innings.overs:
            if not over.deliveries:
                logger.debug(
                    "Skipping empty over %d for %s.", over.over_number, innings.team
                )
                continue

            first_delivery = over.deliveries[0]
            recent_window = self._recent_delivery_window(recent_deliveries, 12)
            recent_count = sum(
                "wides" not in delivery.extras
                and "noballs" not in delivery.extras
                for delivery in recent_window
            )
            recent_runs = sum(
                delivery.runs.get("total", 0) for delivery in recent_window
            )
            state = LiveMatchState(
                over=over.over_number + 1,
                score_before_over=score,
                wkts_down_before_over=wickets,
                balls_faced_before_over=balls_faced.get(first_delivery.batter, 0),
                striker=first_delivery.batter,
                non_striker=first_delivery.non_striker,
                bowler=first_delivery.bowler,
                runs_last_3_overs=sum(value[0] for value in recent_overs[-3:]),
                wickets_last_3_overs=sum(value[1] for value in recent_overs[-3:]),
                wickets_in_hand=max(0, 10 - wickets),
                legal_balls_bowled=legal_balls,
                balls_remaining=max(0, 120 - legal_balls),
                current_run_rate=score * 6 / legal_balls if legal_balls else 0.0,
                recent_legal_balls=recent_count,
                recent_runs_per_ball=(
                    recent_runs / recent_count if recent_count else 0.0
                ),
                recent_dot_rate=(
                    sum(
                        delivery.runs.get("total", 0) == 0
                        for delivery in recent_window
                    )
                    / recent_count
                    if recent_count
                    else 0.0
                ),
                recent_single_rate=(
                    sum(
                        delivery.runs.get("total", 0) == 1
                        for delivery in recent_window
                    )
                    / recent_count
                    if recent_count
                    else 0.0
                ),
                recent_boundary_rate=(
                    sum(
                        delivery.runs.get("batter", 0) in {4, 6}
                        for delivery in recent_window
                    )
                    / recent_count
                    if recent_count
                    else 0.0
                ),
                recent_wicket_rate=(
                    sum(self._dismissal_count(delivery) for delivery in recent_window)
                    / recent_count
                    if recent_count
                    else 0.0
                ),
                match_style="T20",
            )

            actual_runs = 0
            actual_wickets = 0
            for delivery in over.deliveries:
                actual_runs += delivery.runs.get("total", 0)
                if self._is_ball_faced(delivery):
                    balls_faced[delivery.batter] = (
                        balls_faced.get(delivery.batter, 0) + 1
                    )
                dismissals = self._dismissal_count(delivery)
                actual_wickets += dismissals

            logger.debug(
                "Replay frame: %s over %d at %d/%d.",
                innings.team,
                state.over,
                state.score_before_over,
                state.wkts_down_before_over,
            )
            yield ReplayFrame(
                state=state,
                actual_runs=actual_runs,
                actual_wickets=actual_wickets,
            )

            score += actual_runs
            wickets += actual_wickets
            recent_overs.append((actual_runs, actual_wickets))
            for delivery in over.deliveries:
                if "wides" not in delivery.extras and "noballs" not in delivery.extras:
                    legal_balls += 1
                recent_deliveries.append(delivery)

    @staticmethod
    def _recent_delivery_window(
        deliveries: list[Delivery], legal_ball_limit: int
    ) -> list[Delivery]:
        window: list[Delivery] = []
        legal_balls = 0
        for delivery in reversed(deliveries):
            window.append(delivery)
            legal_balls += int(
                "wides" not in delivery.extras
                and "noballs" not in delivery.extras
            )
            if legal_balls == legal_ball_limit:
                break
        window.reverse()
        return window

    @staticmethod
    def _is_ball_faced(delivery: Delivery) -> bool:
        """Return whether a delivery counts as a ball faced by its batter."""

        return "wides" not in delivery.extras

    def _dismissal_count(self, delivery: Delivery) -> int:
        """Count dismissals while excluding a batter retiring hurt."""

        return sum(
            wicket.get("kind") not in self._NON_DISMISSAL_KINDS
            for wicket in delivery.wickets
        )
