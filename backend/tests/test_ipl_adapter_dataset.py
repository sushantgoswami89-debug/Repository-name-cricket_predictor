from app.ml.ipl_adapter_dataset import (
    _batting_strength,
    _bowling_strength,
)


def test_strengths_shrink_empty_player_to_shared_prior() -> None:
    batter = {"balls": 0, "runs": 0, "boundaries": 0, "dismissals": 0}
    batter_prior = {"balls": 100, "runs": 120, "boundaries": 15, "dismissals": 5}
    bowler = {"balls": 0, "runs": 0, "dots": 0, "wickets": 0}
    bowler_prior = {"balls": 100, "runs": 130, "dots": 40, "wickets": 5}
    assert _batting_strength(batter, batter_prior) > 0
    assert _bowling_strength(bowler, bowler_prior) < 0
