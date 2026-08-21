from build_current_spell_bowler_features import (
    _bowler_runs,
    _bowler_wickets,
    _legal,
    _phase,
)


def test_bowler_runs_exclude_byes_and_leg_byes():
    delivery = {
        "runs": {"total": 7, "batter": 0},
        "extras": {"byes": 3, "legbyes": 2},
    }
    assert _bowler_runs(delivery) == 2


def test_wides_and_no_balls_are_not_legal_balls():
    assert not _legal({"extras": {"wides": 1}})
    assert not _legal({"extras": {"noballs": 1}})
    assert _legal({"extras": {"legbyes": 1}})


def test_non_bowler_dismissals_are_excluded():
    delivery = {
        "wickets": [
            {"kind": "bowled"},
            {"kind": "run out"},
            {"kind": "retired out"},
        ]
    }
    assert _bowler_wickets(delivery) == 1


def test_ipl_phase_boundaries():
    assert _phase(6) == "powerplay"
    assert _phase(7) == "middle"
    assert _phase(15) == "middle"
    assert _phase(16) == "death"
