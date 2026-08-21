"""Tests for bounded match-level prediction adaptation."""

from __future__ import annotations

import pytest

from app.services.match_intelligence import (
    IntervalProfile,
    MatchIntelligence,
    MatchIntelligenceError,
)


def _intelligence() -> MatchIntelligence:
    return MatchIntelligence(
        {
            "powerplay": IntervalProfile(-5, 8, 1000),
            "middle": IntervalProfile(-4, 7, 1000),
            "default": IntervalProfile(-5, 8, 3000),
        }
    )


def test_prediction_is_locked_until_actual_over_is_supplied() -> None:
    intelligence = _intelligence()
    intelligence.predict(7.0, 0.30, "powerplay")

    with pytest.raises(MatchIntelligenceError, match="feed_actual_data"):
        intelligence.predict(7.0, 0.30, "powerplay")


def test_high_scoring_over_raises_runs_and_no_wicket_cools_probability() -> None:
    intelligence = _intelligence()
    first = intelligence.predict(7.0, 0.47, "powerplay")
    intelligence.feed_actual_data(actual_runs=12, actual_wickets=0)
    second = intelligence.predict(7.0, 0.47, "powerplay")

    assert second.expected_runs > first.expected_runs
    assert second.wicket_probability < first.wicket_probability
    assert second.range_high > second.range_low


def test_updates_are_bounded_and_phase_change_partially_resets_bias() -> None:
    intelligence = _intelligence()
    for _ in range(8):
        intelligence.predict(5.0, 0.20, "powerplay")
        intelligence.feed_actual_data(actual_runs=30, actual_wickets=1)

    assert intelligence.state.run_bias == 3.0
    previous_bias = intelligence.state.run_bias
    intelligence.predict(5.0, 0.20, "middle")

    assert intelligence.state.run_bias == pytest.approx(previous_bias * 0.70)


def test_snapshot_exposes_auditable_learning_state() -> None:
    intelligence = _intelligence()
    intelligence.predict(7.0, 0.30, "powerplay")
    snapshot = intelligence.feed_actual_data(9, 0)["self_learning"]

    assert snapshot["balls_observed"] == 6
    assert snapshot["current_match_weight"] == 0.1
    assert snapshot["interval_target"] == 0.9
    assert snapshot["prediction_pending"] is False
