from app.ml.innings_phase_player_dataset import bowling_category
from app.ml.venue_track_dataset import _rate


def test_bowling_category_maps_style_and_arm() -> None:
    assert bowling_category("Left arm Fast medium") == ("fast", "left")
    assert bowling_category("Right arm Medium") == ("medium", "right")
    assert bowling_category("Right arm Legbreak") == ("leg_spin", "right")
    assert bowling_category("Slow Left arm Orthodox") == ("off_spin", "left")


def test_sparse_player_rate_shrinks_to_context_prior() -> None:
    player = {"balls": 0, "runs": 0, "boundaries": 0, "dots": 0, "wickets": 0}
    assert _rate(player, "runs", 1.25, 72.0) == 1.25


def test_large_player_sample_has_more_influence() -> None:
    player = {"balls": 720, "runs": 1080, "boundaries": 0, "dots": 0, "wickets": 0}
    assert _rate(player, "runs", 1.0, 72.0) > 1.4
