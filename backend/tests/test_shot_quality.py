"""Unit tests for app/live/shot_quality.py -- the "read commentary like
a human, not a row of data" detectors, user-requested 2026-09-18 after a
concrete example: an edge that races for four is recorded everywhere
else in this codebase as "4 runs, no wicket," indistinguishable from a
well-timed boundary. Real-text cases here are the exact strings
confirmed live against real completed matches the same day.
"""

from __future__ import annotations

from app.live.shot_quality import classify_near_wicket, classify_shot_quality


def test_real_edge_that_went_for_a_wicket():
    text = (
        "Ayaka Stafford to Bharti Fulmali, out Caught by Shimako Kato!! "
        "She's thrown it away, well tossed up leg break outside off, "
        "Fulmali was looking to go inside out, but gets an outside edge "
        "in the air towards point, where Shimako takes a comfortable catch."
    )
    assert classify_shot_quality(text) == "edge"


def test_real_edge_that_went_for_extra_runs_not_a_wicket():
    # The exact shape the user's example described: an edge, positive
    # run outcome, no wicket recorded anywhere else in this codebase.
    text = (
        "Dushmantha Chameera to Will Jacks, no ball, round the wicket, "
        "slanted away outside off, Jacks tries slapping hard but chops "
        "it off the inside edge into the on-side. Oh wait, Chameera has "
        "overstepped, free hit coming up"
    )
    assert classify_shot_quality(text) == "edge"


def test_real_beaten_delivery_that_bowled_the_batter():
    text = (
        "Ravi Bishnoi to Rashid Khan, out Bowled!! well tossed up leg "
        "break on middle, Khan kneels down and goes for the slog sweep, "
        "connects only thin air and the ball goes past the outside edge "
        "to crash into the stumps"
    )
    assert classify_shot_quality(text) == "beaten"


def test_ordinary_delivery_has_no_shot_quality_signal():
    text = (
        "Cooper Connolly to Sikandar Raza, no run, very full, angled "
        "in, the batter backs away and digs it out into the deck"
    )
    assert classify_shot_quality(text) is None


def test_well_timed_shot_detected():
    assert classify_shot_quality("Timed to perfection, races to the boundary") == "well_timed"
    assert classify_shot_quality("Middled that, straight down the ground for four") == "well_timed"


def test_empty_commentary_returns_none():
    assert classify_shot_quality("") is None
    assert classify_near_wicket("") is None


def test_dropped_catch_detected():
    text = "Goes up in the air, and he drops it! A regulation catch put down at mid-off."
    assert classify_near_wicket(text) == "dropped_catch"


def test_survived_close_appeal_detected():
    text = "Huge appeal for LBW from the bowler, the umpire shakes his head, not out."
    assert classify_near_wicket(text) == "survived_close_call"


def test_umpires_call_detected():
    text = "They go upstairs, and it's umpire's call on impact, so the batter survives."
    assert classify_near_wicket(text) == "survived_close_call"


def test_ordinary_delivery_has_no_near_wicket_signal():
    text = "Full and straight, defended back to the bowler, no run."
    assert classify_near_wicket(text) is None
