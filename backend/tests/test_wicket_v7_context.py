import pandas as pd

from train_ipl_wicket_v7_context import _features


def test_wicket_context_features_use_pre_over_state():
    row = pd.DataFrame(
        {
            "striker_match_balls": [12],
            "partner_match_balls": [3],
            "partnership_legal_ball_age": [15],
            "recent_wicket_rate": [0.2],
            "wickets_in_hand": [3],
            "required_run_rate": [12.0],
            "current_run_rate": [8.0],
            "phase": ["middle"],
            "chase_pressure": ["high"],
        }
    )
    result = _features(row).iloc[0]
    assert result["batter_ball_age_bucket"] == "12_23"
    assert result["recent_wicket_flag"] == "RECENT_WICKET"
    assert result["phase_wicket_state"] == "middle|LOW"
    assert result["run_rate_gap"] == 4.0
