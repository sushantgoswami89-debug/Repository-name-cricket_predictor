from run_commentary_study import TOP_TEAMS, involves_top_team


def test_both_sides_in_list_qualifies():
    assert involves_top_team("Sri Lanka vs England", TOP_TEAMS)
    assert involves_top_team("india vs PAKISTAN", TOP_TEAMS)


def test_one_side_outside_list_does_not_qualify():
    assert not involves_top_team("Nepal vs India", TOP_TEAMS)
    assert not involves_top_team("Zimbabwe vs England", TOP_TEAMS)


def test_womens_and_domestic_and_a_teams_do_not_qualify():
    assert not involves_top_team("South Africa Women vs England Women", TOP_TEAMS)
    assert not involves_top_team("India A vs Australia A", TOP_TEAMS)
    assert not involves_top_team("Western Province vs Dolphins", TOP_TEAMS)
    assert not involves_top_team("Kenya vs Uganda", TOP_TEAMS)
