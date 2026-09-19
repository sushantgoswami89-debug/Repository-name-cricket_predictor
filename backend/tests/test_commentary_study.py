from run_commentary_study import TOP_TEAMS, involves_top_team


def test_top_team_on_either_side_qualifies():
    assert involves_top_team("Sri Lanka vs England", TOP_TEAMS)
    assert involves_top_team("Nepal vs India", TOP_TEAMS)


def test_women_suffix_is_ignored():
    assert involves_top_team("South Africa Women vs Zimbabwe Women", TOP_TEAMS)


def test_domestic_and_associate_sides_do_not_qualify():
    assert not involves_top_team("Kenya vs Uganda", TOP_TEAMS)
    assert not involves_top_team("Western Province vs Dolphins", TOP_TEAMS)
    assert not involves_top_team("Amsterdam Flames vs Belfast Wolves", TOP_TEAMS)
