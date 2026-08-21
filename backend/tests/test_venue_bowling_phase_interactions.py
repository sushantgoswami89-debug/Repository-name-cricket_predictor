from analyze_venue_bowling_phase_interactions import _bowling_group


def test_spin_and_pace_categories_are_collapsed():
    assert _bowling_group("leg_spin") == "spin"
    assert _bowling_group("off_spin") == "spin"
    assert _bowling_group("fast") == "pace"
    assert _bowling_group("medium") == "pace"
    assert _bowling_group("unknown") == "unknown"
