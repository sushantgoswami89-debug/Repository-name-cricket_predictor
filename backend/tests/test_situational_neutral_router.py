import numpy as np
import pandas as pd

from backtest_ipl_cold_start_v6_situational_neutral import _state


def test_unknown_batter_transitions_after_twelve_balls():
    data = pd.DataFrame(
        {
            "cold_start_status": ["TEMP", "TEMP"],
            "t20_prior_balls": [0, 0],
            "striker_prior_balls": [0, 0],
            "cold_start_is_bowler": [0, 0],
            "striker_match_balls": [11, 12],
            "chase_pressure": ["high", "high"],
            "recent_wicket_rate": [0.0, 0.0],
            "state_regime": ["stable", "stable"],
        }
    )
    assert _state(data).tolist() == [
        "ENTRY_TENSE",
        "SURVIVED_HIGH_VARIANCE",
    ]


def test_specialist_bowler_always_stays_on_baseline():
    data = pd.DataFrame(
        {
            "cold_start_status": ["TEMP"],
            "t20_prior_balls": [0],
            "striker_prior_balls": [0],
            "cold_start_is_bowler": [1],
            "striker_match_balls": [0],
            "chase_pressure": ["high"],
            "recent_wicket_rate": [1.0],
            "state_regime": ["wicket_pressure"],
        }
    )
    assert np.array_equal(_state(data), np.array(["BASELINE"]))
