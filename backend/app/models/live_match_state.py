"""
Live Match State.

Represents the dynamic state of an ongoing cricket match.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LiveMatchState:
    """
    Dynamic information that changes during a live match.
    """

    over: int = 0

    score_before_over: int = 0

    wkts_down_before_over: int = 0

    balls_faced_before_over: int = 0

    striker: str = ""

    non_striker: str = ""

    bowler: str = ""
