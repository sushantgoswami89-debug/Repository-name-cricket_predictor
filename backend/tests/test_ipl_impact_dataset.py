from app.ml.ipl_impact_dataset import role_category


def test_role_category_normalizes_player_roles() -> None:
    assert role_category("Opening Batter") == "batter"
    assert role_category("Bowling Allrounder") == "allrounder"
    assert role_category("Bowler") == "bowler"
    assert role_category("") == "unknown"
