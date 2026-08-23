from check_stale_paths import main


def test_no_high_risk_stale_paths_reachable_from_live_serving():
    """Guards against exactly the risk raised 2026-08-23: a hardcoded
    path reference silently going stale after a future refactor, inside
    code that's actually part of live serving (reachable from
    prediction_engine.py/match_winner_engine.py/pipeline.py). Low-risk
    hits in standalone research scripts are expected and fine -- see
    check_stale_paths.py's own docstring for why those don't fail this
    test."""
    assert main() == 0
