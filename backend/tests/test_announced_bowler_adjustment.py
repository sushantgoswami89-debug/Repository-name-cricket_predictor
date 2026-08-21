import pytest

from app.ml.announced_bowler_adjustment import (
    BoundedBowlerAdjustment,
    BowlerAnnouncement,
    CurrentSpellTracker,
)


def _announcement(over: int = 5, **changes) -> BowlerAnnouncement:
    values = {
        "bowler": "Safe Bowler",
        "source": "scoreboard_pre_over",
        "over": over,
        "expected_over": over,
        "row_is_empty": True,
        "captured_before_first_ball": True,
    }
    values.update(changes)
    return BowlerAnnouncement(**values)


@pytest.mark.parametrize(
    "changes",
    [
        {"source": "unannounced"},
        {"expected_over": 4},
        {"row_is_empty": False},
        {"captured_before_first_ball": False},
        {"bowler": ""},
    ],
)
def test_only_explicit_empty_pre_over_scoreboard_row_is_accepted(changes):
    assert _announcement(**changes).accepted_bowler() == ""


def test_unannounced_case_keeps_situation_prediction_exactly():
    tracker = CurrentSpellTracker()
    features = tracker.features(
        _announcement(source="unannounced", captured_before_first_ball=False)
    )
    result = BoundedBowlerAdjustment().apply(
        base_runs=8.25,
        base_wicket_probability=0.31,
        features=features,
        proposed_run_delta=-20,
        proposed_wicket_logit_delta=20,
    )
    assert result.runs == 8.25
    assert result.wicket_probability == 0.31
    assert result.run_delta == 0
    assert result.wicket_logit_delta == 0
    assert result.bowler_source == "unannounced"


def test_unknown_announced_bowler_has_neutral_prior():
    features = CurrentSpellTracker().features(_announcement())
    result = BoundedBowlerAdjustment().apply(
        base_runs=7.0,
        base_wicket_probability=0.25,
        features=features,
        proposed_run_delta=1.0,
        proposed_wicket_logit_delta=0.4,
    )
    assert result.runs == 7.0
    assert result.wicket_probability == 0.25


def test_completed_over_state_is_visible_only_after_recording():
    tracker = CurrentSpellTracker()
    before = tracker.features(_announcement(over=3))
    assert before.match_balls == 0
    tracker.record_completed_over(
        bowler="Safe Bowler",
        over=3,
        legal_balls=6,
        runs_conceded=8,
        wickets=1,
        dots=3,
        boundaries=1,
    )
    after = tracker.features(_announcement(over=5))
    assert after.match_balls == 6
    assert after.match_runs_conceded == 8
    assert after.match_wickets == 1
    assert after.match_dot_rate == 0.5
    assert after.match_boundary_concession_rate == pytest.approx(1 / 6)
    assert after.match_strike_rate == 6
    assert after.match_economy == 8
    assert after.overs_since_previous == 1
    assert after.spell_number == 1
    assert after.current_spell_balls == 6


def test_return_after_two_intervening_overs_starts_new_spell():
    tracker = CurrentSpellTracker()
    tracker.record_completed_over(
        bowler="Safe Bowler",
        over=2,
        legal_balls=6,
        runs_conceded=4,
        wickets=0,
        dots=3,
        boundaries=0,
    )
    returning = tracker.features(_announcement(over=5))
    assert returning.overs_since_previous == 2
    assert returning.spell_number == 2
    assert returning.current_spell_balls == 0


def test_unsupported_history_and_h2h_are_suppressed():
    features = CurrentSpellTracker().features(
        _announcement(),
        history_phase_balls=119,
        history_phase_runs=500,
        history_phase_wickets=20,
        h2h_balls=23,
        h2h_runs=100,
        h2h_wickets=5,
    )
    assert not features.history_supported
    assert features.history_phase_economy == 0
    assert not features.h2h_supported
    assert features.h2h_balls == 0


def test_adjustment_is_bounded_even_with_strong_evidence():
    tracker = CurrentSpellTracker()
    for over in (1, 3, 5, 7, 9, 11):
        tracker.record_completed_over(
            bowler="Safe Bowler",
            over=over,
            legal_balls=6,
            runs_conceded=4,
            wickets=1,
            dots=3,
            boundaries=0,
        )
    features = tracker.features(
        _announcement(over=13),
        history_phase_balls=600,
        history_phase_runs=700,
        history_phase_wickets=30,
        h2h_balls=240,
        h2h_runs=250,
        h2h_wickets=10,
    )
    result = BoundedBowlerAdjustment().apply(
        base_runs=8,
        base_wicket_probability=0.3,
        features=features,
        proposed_run_delta=-99,
        proposed_wicket_logit_delta=99,
    )
    assert result.run_delta == -1.5
    assert result.wicket_logit_delta == 0.45
    assert 0.3 < result.wicket_probability < 0.5
