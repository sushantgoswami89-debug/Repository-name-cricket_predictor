"""Regression: Cricbuzz reuses a ballMetric for the re-bowled ball after a
wide/no-ball. Deduping on (innings, over, ball) dropped it, so the over never
reached 6 legal balls (seen live 2026-10-06, over 4 never predicted)."""

from app.live.cricbuzz_reader import CricbuzzLiveReader
from app.live.verification import LiveDeliveryVerifier


def _obj(ts, metric, text, event="none"):
    return {
        "matchId": 1, "commType": "commentary", "commText": text, "inningsId": 1,
        "event": [event, "all"], "ballMetric": metric, "teamName": "WI", "timestamp": ts,
        "overSeparator": None,
        "batsmanDetails": {"playerName": "Hope"}, "bowlerDetails": {"playerName": "Arshdeep"},
    }


def _over_with_no_ball():
    return [
        _obj(1, 2.1, "Arshdeep to Hope, no ball, FOUR", "four"),
        _obj(2, 2.1, "Arshdeep to Hope, no run"),  # re-bowled, same ballMetric
        _obj(3, 2.2, "Arshdeep to Hope, no run"),
        _obj(4, 2.3, "Arshdeep to Hope, 1 run"),
        _obj(5, 2.4, "Arshdeep to Hope, no run"),
        _obj(6, 2.5, "Arshdeep to Hope, no run"),
        _obj(7, 2.6, "Arshdeep to Hope, no run"),
    ]


def test_rebowled_ball_after_no_ball_is_kept_and_over_completes():
    reader = CricbuzzLiveReader()
    deliveries = reader._build_deliveries(("1", 1), _over_with_no_ball())
    assert len(deliveries) == 7
    assert [d.ball for d in deliveries] == [1, 2, 3, 4, 5, 6, 7]
    assert sum(d.is_legal for d in deliveries) == 6
    verifier = LiveDeliveryVerifier("1", 1)
    for delivery in deliveries:
        verifier.apply(delivery)
    assert verifier.completed_over == 3


def test_repolling_same_window_does_not_duplicate():
    reader = CricbuzzLiveReader()
    first = reader._build_deliveries(("1", 1), _over_with_no_ball())
    second = reader._build_deliveries(("1", 1), _over_with_no_ball())
    assert len(first) == len(second) == 7


def test_mid_innings_attach_keeps_score_baseline():
    """A verifier attached at 44/1 must report score-before-over from that
    baseline, not just the runs it saw (2026-10-06 'Score: 9/2' for 53/3)."""
    from app.live.live_snapshot import LiveDelivery

    verifier = LiveDeliveryVerifier("1", 1)
    for ball, (runs, total) in enumerate([(2, 46), (0, 46), (4, 50)], start=1):
        verifier.apply(
            LiveDelivery(
                match_id="1", innings=1, over=7, ball=ball, bowler="b", striker="s",
                total_runs=runs, batter_runs=runs, extras={}, wicket_kind=None,
                feed_total=total, feed_wickets=1, timestamp_ms=ball, commentary="",
            )
        )
    assert verifier.base_score == 44 and verifier.base_wickets == 1
    assert verifier.base_score + sum(verifier.over_runs.values()) == 50


def test_leg_byes_four_is_extras_not_batter_runs():
    from app.live.cricbuzz_reader import _runs_and_extras

    total, bat, extras = _runs_and_extras("Mayank Yadav to Shai Hope, leg byes, <b>FOUR</b>", ["four", "all"])
    assert (total, bat, extras) == (4, 0, {"legbyes": 4})
    total, bat, extras = _runs_and_extras("Mayank Yadav to Shai Hope, <b>FOUR</b>, to long-off", ["four", "all"])
    assert (total, bat, extras) == (4, 4, {})


def test_longest_commentary_text_is_kept_across_polls():
    reader = CricbuzzLiveReader()
    long_text = "Kuldeep to Hope, SIX, it's so easy when Kuldeep bowls like this. edged"
    reader._build_deliveries(("1", 1), [_obj(10, 8.6, "Kuldeep to Hope, SIX", "six")])
    after_long = reader._build_deliveries(("1", 1), [_obj(10, 8.6, long_text, "six")])
    assert after_long[-1].commentary == long_text
    after_short = reader._build_deliveries(("1", 1), [_obj(10, 8.6, "Kuldeep to Hope, SIX", "six")])
    assert after_short[-1].commentary == long_text


def test_mistimes_verb_form_is_recognised():
    from app.live.shot_quality import classify_shot_quality

    assert classify_shot_quality("Rutherford mistimes the sweep to deep backward square-leg") == "mistimed"


def test_drinks_break_and_bowling_change_are_surfaced_as_notes():
    reader = CricbuzzLiveReader()
    drinks = _obj(20, "$undefined", "Time for a drink. With Shreyas using Arshdeep for three overs...")
    change = _obj(21, "$undefined", "Naman Dhir, right-arm off break, comes into the attack")
    trivia = _obj(22, "$undefined", "Most maiden overs for India in T20Is13 - Jasprit Bumrah")
    reader._build_deliveries(("1", 1), [drinks, change, trivia])
    notes = reader.drain_notes()
    assert [category for _ts, category, _text in notes] == ["DRINKS/TIMEOUT", "BOWLING CHANGE"]
    reader._build_deliveries(("1", 1), [drinks, change, trivia])
    assert reader.drain_notes() == []


def test_prose_containing_wide_is_not_a_wide():
    from app.live.cricbuzz_reader import _runs_and_extras

    prose = (
        "Arshdeep Singh to Sherfane Rutherford, 1 run, good yorker length delivery on off, "
        "Sherfane Rutherford picks it and jams it down, wide of short fine for one"
    )
    assert _runs_and_extras(prose, ["none", "all"]) == (1, 1, {})
    # fielding words that look like extras must not change the score either
    assert _runs_and_extras("A to B, no run, beaten, goes through to the keeper, byes would have been", ["none"]) == (0, 0, {})


def test_real_extras_still_parse_from_the_opening_phrase():
    from app.live.cricbuzz_reader import _runs_and_extras

    assert _runs_and_extras("A to B, wide, outside off", ["none"]) == (1, 0, {"wides": 1})
    assert _runs_and_extras("A to B, wide, 3 runs, down leg", ["none"]) == (4, 0, {"wides": 4})
    assert _runs_and_extras("A to B, no ball, FOUR", ["four"]) == (5, 4, {"noballs": 1})
    assert _runs_and_extras("A to B, leg byes, 2 runs, off the pads", ["none"]) == (2, 0, {"legbyes": 2})
    assert _runs_and_extras("A to B, byes, 1 run", ["none"]) == (1, 0, {"byes": 1})
    assert _runs_and_extras("A to B, 5 wides, way down leg", ["none"]) == (5, 0, {"wides": 5})
    assert _runs_and_extras("A to B, SIX, over long-on", ["six"]) == (6, 6, {})


def test_over_summary_is_captured_once_and_audit_flags_a_missing_ball():
    from types import SimpleNamespace

    from app.live.cricbuzz_session import _audit_over

    reader = CricbuzzLiveReader()
    reader._running_score[("1", 1)] = 0  # skip mid-innings seeding: a fresh match
    reader._running_wickets[("1", 1)] = 0
    reader._seed_cutoff_ms[("1", 1)] = -1
    last = _obj(9, 0.6, "A to B, no run")
    last["overSeparator"] = {
        "overNumber": 1, "overSummary": "1 1 0 0 0 0 ", "overRuns": 2,
        "batTeamObj": {"teamScore": "2-0"}, "bowlerObj": {"playerName": "A", "playerScore": "1-0-2-0"},
    }
    window = [_obj(i, 0 + i / 10, "A to B, 1 run" if i < 3 else "A to B, no run") for i in range(1, 6)] + [last]
    deliveries = reader._build_deliveries(("1", 1), window)
    summaries = reader.drain_over_summaries()
    assert len(summaries) == 1 and summaries[0]["over"] == 1
    assert reader.drain_over_summaries() == []
    snap = SimpleNamespace(deliveries=deliveries)
    assert "OK" in _audit_over(snap, summaries[0])
    missing = SimpleNamespace(deliveries=deliveries[:-1])
    assert "MISMATCH" in _audit_over(missing, summaries[0])
    assert "not seen live" in _audit_over(SimpleNamespace(deliveries=[]), summaries[0])


def _mid_innings_window():
    """Ten balls of over 9 and 10 with no over-checkpoint in the window (attached mid-innings)."""
    return [
        _obj(1, 8.5, "Arshdeep to Hope, 1 run"),
        _obj(2, 8.6, "Arshdeep to Hope, FOUR", "four"),
        _obj(3, 9.1, "Arshdeep to Hope, no run"),
        _obj(4, 9.2, "Arshdeep to Hope, 2 runs"),
        _obj(5, 9.3, "Arshdeep to Hope, 1 run"),
    ]


def test_mid_innings_attach_without_checkpoint_takes_baseline_from_matching_board():
    import pytest as _pytest

    from app.live.cricbuzz_reader import CricbuzzFeedError

    window = _mid_innings_window()  # 1+4+0+2+1 = 8 runs, latest ball 9.3 -> board overs "9.3"
    reader = CricbuzzLiveReader()
    board = {"innings": 1, "score": 98, "wickets": 2, "overs": "9.3"}
    deliveries = reader._build_deliveries(("1", 1), window, board=board, require_baseline=True)
    assert deliveries[-1].feed_total == 98 and deliveries[-1].feed_wickets == 2

    # an older cached copy of the page must not add already-counted balls again
    older = [_obj(0, 8.1, "Arshdeep to Hope, 6 runs")] + window
    again = reader._build_deliveries(("1", 1), older, board=board, require_baseline=True)
    assert again[-1].feed_total == 98

    # no board, or a board at a different point, means wait -- never guess 0/0
    for bad_board in (None, {"innings": 1, "score": 90, "wickets": 2, "overs": "9.1"}):
        with _pytest.raises(CricbuzzFeedError):
            CricbuzzLiveReader()._build_deliveries(("1", 1), window, board=bad_board, require_baseline=True)


def test_window_that_reaches_the_first_ball_still_starts_from_zero():
    reader = CricbuzzLiveReader()
    window = [_obj(1, 0.1, "Arshdeep to Hope, 1 run"), _obj(2, 0.2, "Arshdeep to Hope, FOUR", "four")]
    deliveries = reader._build_deliveries(("1", 1), window, board=None, require_baseline=True)
    assert deliveries[-1].feed_total == 5


def test_long_commentary_sent_as_a_text_row_pointer_is_resolved():
    """Cricbuzz sends long texts as a pointer ("$26") plus a separate text row ("26:T<hexlen>," then the text).
    151565 ball 10.5 arrived as the bare pointer, so its runs/wicket words were invisible and the over came out wrong."""
    from app.live.cricbuzz_reader import _delivery_objects, _runs_and_extras

    body = "Springer to Kishan, 1 run, they are checking for a clean catch but the third umpire says not out."
    entry = ('{"matchId":1,"commType":"commentary","commText":"$26","inningsId":2,"event":["none","all"],'
             '"ballMetric":10.5,"teamName":"IND","timestamp":7,"overSeparator":null,'
             '"batsmanDetails":{"playerName":"Kishan"},"bowlerDetails":{"playerName":"Springer"}}')
    chunks = [entry, f"26:T{len(body.encode()):x},", body, "27:T3,", "abc"]
    [obj] = _delivery_objects(chunks, "1")
    assert obj["commText"] == body
    assert _runs_and_extras(obj["commText"], []) == (1, 1, {})
    # a pointer with no matching row is left alone rather than invented
    [orphan] = _delivery_objects([entry.replace("$26", "$99")], "1")
    assert orphan["commText"] == "$99"
    # multi-byte text: the length counts bytes
    smart = "Hope ’s six, 6 runs"
    chunks = [entry.replace("$26", "$30"), f"30:T{len(smart.encode()):x},", smart]
    assert _delivery_objects(chunks, "1")[0]["commText"] == smart
