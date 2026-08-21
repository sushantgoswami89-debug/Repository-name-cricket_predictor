from app.ml.state_strike_share_dataset import expected_striker_balls


def test_expected_striker_balls_when_strike_never_rotates() -> None:
    assert expected_striker_balls(0.0, 0.0) == 6.0


def test_expected_striker_balls_is_balanced_for_certain_rotation() -> None:
    assert expected_striker_balls(1.0, 1.0) == 3.0


def test_expected_striker_balls_stays_bounded() -> None:
    value = expected_striker_balls(0.27, 0.31)
    assert 3.0 <= value <= 6.0
