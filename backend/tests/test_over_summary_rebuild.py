"""Overs are rebuilt from Cricbuzz's own over summary when the ball entries
disagree (review replaced an OUT, a ball was dropped). Token formats are the
ones verified against overRuns in archived pages, 2026-10-06/-07."""

import pytest

from app.live.cricbuzz_reader import CricbuzzLiveReader
from app.live.over_summary import parse_over_summary, tokens_agree_with_total
from app.live.verification import LiveDeliveryVerifier


@pytest.mark.parametrize(
    "summary,runs,legal,wickets",
    [
        ("1 6 4 N1 1 1 4", 19, 6, 0),
        ("4 2 N 1 Wd W 6 W", 15, 6, 2),
        ("1 0 4 Wd3 0 0 Wd 1", 10, 6, 0),
        ("4 1 4 L1 0 Wd 4", 15, 6, 0),
        ("0 0 4 0 0 W", 4, 6, 1),
    ],
)
def test_real_over_summaries_parse_to_their_over_runs(summary, runs, legal, wickets):
    tokens = parse_over_summary(summary)
    assert tokens is not None
    assert tokens_agree_with_total(tokens, runs)
    assert sum(t.is_legal for t in tokens) == legal
    assert sum(t.wicket for t in tokens) == wickets


def test_unverified_token_is_refused_not_guessed():
    assert parse_over_summary("1 B2 0") is None
    assert parse_over_summary("") is None


def _obj(ts, metric, text, event="none", separator=None):
    return {
        "matchId": 1, "commType": "commentary", "commText": text, "inningsId": 2,
        "event": [event, "all"], "ballMetric": metric, "teamName": "IND", "timestamp": ts,
        "overSeparator": separator,
        "batsmanDetails": {"playerName": "Iyer"}, "bowlerDetails": {"playerName": "Springer"},
    }


def _reader():
    """A reader already tracking this innings at 102/2 (watched live from the start),
    so the first-attach checkpoint seeding does not hide the over under test."""
    reader = CricbuzzLiveReader()
    reader._running_score[("1", 2)] = 102
    reader._running_wickets[("1", 2)] = 2
    reader._seed_cutoff_ms[("1", 2)] = -1
    return reader


def _separator(summary, runs, score):
    return {
        "overNumber": 11, "overSummary": summary, "overRuns": runs,
        "batTeamObj": {"teamScore": score}, "bowlerObj": {"playerName": "Springer", "playerScore": "1-0-12-0"},
    }


def test_reviewed_out_is_removed_and_over_matches_cricbuzz_summary():
    """Over 11 of the 2026-10-06 chase: an OUT was posted, the third umpire
    overturned it and the corrected entry reused the ball number."""
    reader = _reader()
    window = [
        _obj(1, 10.1, "Springer to Iyer, no run"),
        _obj(2, 10.2, "Springer to Iyer, FOUR", "four"),
        _obj(3, 10.3, "Springer to Iyer, 1 run"),
        _obj(4, 10.4, "Springer to Iyer, out Caught by Forde!!", "wicket"),
        _obj(5, 10.4, "Springer to Iyer, no run"),  # review: not out, same ball number
        _obj(6, 10.5, "Springer to Iyer, 1 run"),
        _obj(7, 10.6, "Springer to Iyer, SIX", "six",
             _separator("0 4 1 0 1 6", 12, "114-2")),
    ]
    deliveries = reader._build_deliveries(("1", 2), window)
    assert [d.total_runs for d in deliveries] == [0, 4, 1, 0, 1, 6]
    assert not any(d.wicket_kind for d in deliveries)
    assert sum(d.is_legal for d in deliveries) == 6
    # score chains to Cricbuzz's own checkpoint, wickets without the phantom
    assert (deliveries[-1].feed_total, deliveries[-1].feed_wickets) == (114, 2)
    assert reader._running_score[("1", 2)] == 114
    assert reader.drain_repairs()
    verifier = LiveDeliveryVerifier("1", 2)
    for delivery in deliveries:
        verifier.apply(delivery)
    assert verifier.completed_over == 11

    # re-polling the same window must not resurrect the dropped OUT
    again = reader._build_deliveries(("1", 2), window)
    assert [d.total_runs for d in again] == [0, 4, 1, 0, 1, 6]
    assert not any(d.wicket_kind for d in again)


def test_over_that_already_agrees_is_untouched():
    reader = _reader()
    window = [
        _obj(1, 10.1, "Springer to Iyer, no run"),
        _obj(2, 10.2, "Springer to Iyer, FOUR", "four"),
        _obj(3, 10.3, "Springer to Iyer, 1 run"),
        _obj(4, 10.4, "Springer to Iyer, no run"),
        _obj(5, 10.5, "Springer to Iyer, 1 run"),
        _obj(6, 10.6, "Springer to Iyer, SIX", "six", _separator("0 4 1 0 1 6", 12, "114-2")),
    ]
    deliveries = reader._build_deliveries(("1", 2), window)
    assert len(deliveries) == 6
    assert reader.drain_repairs() == []


def test_missing_ball_is_filled_from_the_summary():
    reader = _reader()
    window = [
        _obj(1, 10.1, "Springer to Iyer, no run"),
        _obj(2, 10.2, "Springer to Iyer, FOUR", "four"),
        _obj(4, 10.4, "Springer to Iyer, no run"),
        _obj(5, 10.5, "Springer to Iyer, 1 run"),
        _obj(6, 10.6, "Springer to Iyer, SIX", "six", _separator("0 4 1 0 1 6", 12, "114-2")),
    ]
    deliveries = reader._build_deliveries(("1", 2), window)
    assert [d.total_runs for d in deliveries] == [0, 4, 1, 0, 1, 6]
    assert deliveries[2].commentary.startswith("(rebuilt")
    assert (deliveries[-1].feed_total, deliveries[-1].feed_wickets) == (114, 2)


def test_revised_legal_ball_replaces_the_earlier_entry_without_waiting_for_the_summary():
    """A reviewed OUT is replaced in place, so the over cannot reach 6 legal
    balls one ball early and the phantom wicket never counts."""
    reader = _reader()
    first = [
        _obj(1, 10.1, "Springer to Iyer, no run"),
        _obj(2, 10.2, "Springer to Iyer, FOUR", "four"),
        _obj(3, 10.3, "Springer to Iyer, 1 run"),
        _obj(4, 10.4, "Springer to Iyer, out Caught by Forde!!", "wicket"),
    ]
    before = reader._build_deliveries(("1", 2), first)
    assert before[-1].feed_wickets == 3 and before[-1].feed_total == 107
    revised = first + [_obj(5, 10.4, "Springer to Iyer, no run"), _obj(6, 10.5, "Springer to Iyer, 1 run")]
    after = reader._build_deliveries(("1", 2), revised)
    assert [d.ball for d in after] == [1, 2, 3, 4, 5]
    assert [d.wicket_kind for d in after] == [None] * 5
    assert (after[-1].feed_total, after[-1].feed_wickets) == (108, 2)
    assert reader._running_score[("1", 2)] == 108
    assert reader._running_wickets[("1", 2)] == 2
    verifier = LiveDeliveryVerifier("1", 2)
    for delivery in after:
        verifier.apply(delivery)
    assert verifier.completed_over is None  # only 5 legal balls so far


def test_wide_then_rebowled_ball_is_still_two_deliveries():
    reader = _reader()
    window = [
        _obj(1, 10.1, "Springer to Iyer, wide"),
        _obj(2, 10.1, "Springer to Iyer, no run"),
    ]
    deliveries = reader._build_deliveries(("1", 2), window)
    assert [d.is_legal for d in deliveries] == [False, True]
