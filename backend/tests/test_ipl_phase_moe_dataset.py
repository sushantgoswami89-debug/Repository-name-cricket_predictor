from pathlib import Path

from app.ml.ipl_phase_moe_dataset import (
    UNKNOWN_CATEGORY,
    build_ipl_phase_moe_features,
)


def test_phase_moe_features_are_unique_and_bowler_safe():
    root = Path(__file__).resolve().parents[2]
    frame = build_ipl_phase_moe_features(root)

    assert "1359475.json" in set(frame["source_file"])
    assert not frame.duplicated(
        ["source_file", "match_date", "innings", "over"]
    ).any()
    assert set(frame["known_bowler"]) == {UNKNOWN_CATEGORY}
    assert set(frame["active_batter_state"]) <= {
        "new_batter",
        "established_pair",
    }
    assert set(frame["state_regime"]) <= {
        "wicket_pressure",
        "accelerating",
        "stable",
    }


def test_player_history_is_frozen_for_first_ipl_match():
    root = Path(__file__).resolve().parents[2]
    frame = build_ipl_phase_moe_features(root)
    first_match = frame.sort_values(
        ["match_date", "source_file", "innings", "over"]
    ).iloc[0]["source_file"]
    first = frame[frame["source_file"] == first_match]

    assert (first["striker_prior_balls"] == 0).all()
    assert (first["partner_prior_balls"] == 0).all()


def test_canonical_view_uses_uuid_players_and_franchise_ids():
    root = Path(__file__).resolve().parents[2]
    frame = build_ipl_phase_moe_features(root, canonical_identities=True)

    assert frame["striker"].str.startswith("player:").all()
    assert frame["non_striker"].str.startswith("player:").all()
    assert frame["batting_team"].str.startswith("team:").all()
