"""
Live Match State - Evolved Version.
Tracks momentum and prevents state-jumping.
"""

from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class LiveMatchState:
    """
    Dynamic information that changes ball-by-ball.
    """
    # Current identifiers
    over: int = 0
    striker: str = ""
    non_striker: str = ""
    bowler: str = ""

    # Totals before the current over starts
    score_before_over: int = 0
    wkts_down_before_over: int = 0
    balls_faced_before_over: int = 0

    # --- MOMENTUM FEATURES (For 90% Accuracy) ---
    # These allow the model to 'see' the current match rhythm
    runs_last_3_overs: int = 0
    wickets_last_3_overs: int = 0

    # Candidate v3 pre-over state. These definitions mirror
    # app.ml.candidate_v3_dataset exactly.
    wickets_in_hand: int = 10
    legal_balls_bowled: int = 0
    balls_remaining: int = 0
    current_run_rate: float = 0.0
    is_chase: int = 0
    runs_required: int = 0
    required_run_rate: float = 0.0
    recent_legal_balls: int = 0
    recent_runs_per_ball: float = 0.0
    recent_dot_rate: float = 0.0
    recent_single_rate: float = 0.0
    recent_boundary_rate: float = 0.0
    recent_wicket_rate: float = 0.0
    
    # --- MATCH TYPE ---
    # Helps the engine stop confusing T20 and ODI patterns
    match_style: str = "IPL" # Default: IPL, T20I, or ODI

    # --- VALIDATION ---
    is_result_pending: bool = False # Prevents jumping to the next over
