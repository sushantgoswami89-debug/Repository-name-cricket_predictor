from build_announced_bowler_features import _bowler_wickets, _legal


def test_bowler_wicket_excludes_run_out():
    delivery = {
        "wickets": [
            {"kind": "bowled"},
            {"kind": "run out"},
        ]
    }
    assert _bowler_wickets(delivery) == 1


def test_wide_is_not_a_legal_bowler_ball():
    assert not _legal({"extras": {"wides": 1}})
