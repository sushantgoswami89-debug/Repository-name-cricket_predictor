import numpy as np

from train_announced_bowler_current_spell_v2 import (
    MAX_RUN_DELTA,
    MAX_WICKET_LOGIT_DELTA,
    _blend_runs,
    _blend_wickets,
    _logit,
)


def test_run_adjustment_is_bounded_and_can_be_disabled():
    base = np.array([8.0])
    proposed = np.array([20.0])
    assert _blend_runs(base, proposed, 0.0)[0] == 8.0
    assert _blend_runs(base, proposed, 1.0)[0] == 8.0 + MAX_RUN_DELTA


def test_wicket_adjustment_is_logit_bounded_and_can_be_disabled():
    base = np.array([0.25])
    proposed = np.array([0.99])
    unchanged = _blend_wickets(base, proposed, 0.0)
    adjusted = _blend_wickets(base, proposed, 1.0)
    assert np.allclose(unchanged, base)
    assert np.allclose(_logit(adjusted) - _logit(base), MAX_WICKET_LOGIT_DELTA)
