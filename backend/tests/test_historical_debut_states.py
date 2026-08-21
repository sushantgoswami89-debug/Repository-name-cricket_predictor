import pandas as pd

from analyze_historical_debut_states import _ball_bucket, _pressure_trigger


def test_ball_age_buckets_have_twelve_ball_transition():
    balls = pd.Series([0, 5, 6, 11, 12, 23, 24])
    assert _ball_bucket(balls).tolist() == [
        "0_5",
        "0_5",
        "6_11",
        "6_11",
        "12_23",
        "12_23",
        "24_plus",
    ]


def test_recent_wicket_has_pressure_precedence():
    data = pd.DataFrame(
        {
            "recent_wicket_rate": [0.2],
            "chase_pressure": ["high"],
            "wickets_in_hand": [2],
        }
    )
    assert _pressure_trigger(data).tolist() == ["recent_wicket"]
