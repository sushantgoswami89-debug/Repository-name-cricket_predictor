"""Unit tests for the Cricbuzz reader's toss/venue/squad extraction,
added 2026-09-18 to close the "team_players/toss enrichment not ported
here yet" gap flagged in app/live/cricbuzz_reader.py's own module
docstring. Fixtures are shaped exactly like the real flight-chunk JSON
confirmed live against a real match (Australia vs Zimbabwe, 2nd ODI,
2026-09-18) -- see that module's docstrings for the full field-by-field
confirmation.
"""

from __future__ import annotations

from app.live.cricbuzz_reader import CricbuzzLiveReader, _classify_wicket, _runs_and_extras


def test_toss_from_chunks_extracts_winner_and_decision():
    chunk = (
        '{"matchId":152742,"state":"In Progress",'
        '"tossResults":{"tossWinnerId":12,"tossWinnerName":"Zimbabwe",'
        '"decision":"Bowling"},"result":{}}'
    )
    assert CricbuzzLiveReader._toss_from_chunks([chunk]) == ("Zimbabwe", "Bowling")


def test_toss_from_chunks_before_toss_returns_empty():
    chunk = '{"matchId":171657,"state":"Preview","tossResults":{},"result":{}}'
    assert CricbuzzLiveReader._toss_from_chunks([chunk]) == ("", "")


def test_venue_from_chunks_extracts_ground_and_city():
    chunk = (
        '{"matchInfo":{"matchId":152742,"seriesId":11997,'
        '"venueInfo":{"id":69,"ground":"Harare Sports Club","city":"Harare",'
        '"timezone":"+02:00"}}}'
    )
    assert (
        CricbuzzLiveReader._venue_from_chunks([chunk], "152742")
        == "Harare Sports Club, Harare"
    )


def test_venue_from_chunks_wrong_match_id_is_ignored():
    chunk = (
        '{"matchInfo":{"matchId":999,'
        '"venueInfo":{"ground":"Some Other Ground","city":"Elsewhere"}}}'
    )
    assert CricbuzzLiveReader._venue_from_chunks([chunk], "152742") == ""


def _squad_chunk() -> str:
    # Shaped exactly like the real object confirmed live: 11 confirmed XI
    # (substitute=False), 4 bench (substitute=True), 4 coaching staff
    # (substitute=null) sharing the same list.
    players = (
        [
            f'{{"id":{100 + i},"name":"Player {i}","role":"Batsman",'
            f'"substitute":false,"teamId":12}}'
            for i in range(11)
        ]
        + [
            f'{{"id":{200 + i},"name":"Bench {i}","role":"Bowler",'
            f'"substitute":true,"teamId":12}}'
            for i in range(4)
        ]
        + [
            f'{{"id":{300 + i},"name":"Coach {i}","role":"Assistant coach",'
            f'"substitute":null,"teamId":12}}'
            for i in range(4)
        ]
    )
    return (
        '{"matchId":152742,'
        '"team1":{"id":12,"name":"Zimbabwe","playerDetails":['
        + ",".join(players)
        + ']},"team2":{"id":4,"name":"Australia","playerDetails":[]}}'
    )


def test_squads_from_chunks_excludes_coaching_staff():
    squads = CricbuzzLiveReader._squads_from_chunks([_squad_chunk()])

    players = squads["Zimbabwe"]
    assert len(players) == 15  # 11 confirmed XI + 4 bench, coaches excluded
    assert all(not name.startswith("Coach") for name in (p.name for p in players))


def test_squads_from_chunks_confirm_xi_matches_substitute_flag():
    squads = CricbuzzLiveReader._squads_from_chunks([_squad_chunk()])

    players = squads["Zimbabwe"]
    confirmed = [p for p in players if p.confirm_xi]
    bench = [p for p in players if not p.confirm_xi]
    assert len(confirmed) == 11
    assert len(bench) == 4
    assert all(name.startswith("Player") for name in (p.name for p in confirmed))
    assert all(name.startswith("Bench") for name in (p.name for p in bench))


def test_squads_from_chunks_empty_playerdetails_is_skipped():
    squads = CricbuzzLiveReader._squads_from_chunks([_squad_chunk()])

    assert "Australia" not in squads


# Real commentary text, confirmed live 2026-09-18 by scanning archived
# completed matches for event types docs/live_field_test_2026-09-15.md
# flagged as unconfirmed (scan_cricbuzz_event_types.py). Regression tests
# for a real bug found this way: _leading_run_count's regex didn't
# tolerate the leading comma Cricbuzz always puts before an explicit run
# count, silently defaulting to 1 run on every extra worth more than one.


def test_bowled_wicket_from_real_commentary():
    text = (
        "Ravi Bishnoi to Rashid Khan, <b>out</b> Bowled!! 2nd wicket for "
        "Bishnoi, well tossed up leg break on middle, Khan kneels down and "
        "goes for the slog sweep, connects only thin air and the ball goes "
        "past the outside edge to crash into the stumps, India are one "
        "wicket away from a clean sweep. <b>Rashid Khan b Ravi Bishnoi "
        "6(5) [4s-1]</b>"
    )
    assert _classify_wicket(text) == "bowled"


def test_caught_wicket_from_real_commentary():
    text = (
        "Ravi Bishnoi to Abdullah Ahmadzai, <b>out</b> Caught by Axar "
        "Patel!! It's all over, a clean sweep for India, well tossed up on "
        "off, Ahmadzai goes for a wild slog across the line and miscues it "
        "high in the air towards deep midwicket, where Axar takes a well "
        "judged catch moving to his left. <b>Abdullah Ahmadzai c Axar "
        "Patel b Ravi Bishnoi 6(8) [4s-1]</b>"
    )
    assert _classify_wicket(text) == "caught"


def test_real_leg_bye_with_explicit_multi_run_count():
    # The real bug: this used to parse as 1, not 2 -- the leading comma
    # before "2 runs" broke _leading_run_count's regex.
    text = "Josh Bohannon to S Northeast, leg byes, 2 runs"
    assert _runs_and_extras(text, []) == (2, 0, {"legbyes": 2})


def test_real_no_ball_with_no_explicit_run_count():
    text = (
        "Dushmantha Chameera to Will Jacks, <b>no ball</b>, round the "
        "wicket, slanted away outside off, Jacks tries slapping hard but "
        "chops it off the inside edge into the on-side. Oh wait, Chameera "
        "has overstepped, free hit coming up"
    )
    # No explicit run count in the opening phrase. Cricbuzz states runs
    # explicitly ("no ball, 1 run", "no ball, FOUR"), so a bare "no ball" is
    # the penalty run only; the old +1 guess read prose instead. A wrong total
    # is caught by the over-end checkpoint comparison.
    assert _runs_and_extras(text, []) == (1, 0, {"noballs": 1})


def test_wide_with_explicit_multi_run_count():
    text = "Bowler to Batter, wide, 3 runs, way down the leg side"
    assert _runs_and_extras(text, []) == (4, 0, {"wides": 4})


def test_no_ball_with_explicit_multi_run_count():
    text = "Bowler to Batter, no ball, 2 runs, edged past the keeper"
    assert _runs_and_extras(text, []) == (3, 2, {"noballs": 1})


def test_int_field_tolerates_next_js_undefined_marker():
    # Seen live 2026-09-19: inningsId arrived as the string "$undefined"
    # and crashed the whole driver with a ValueError.
    from app.live.cricbuzz_reader import _int_field

    assert _int_field({"inningsId": "$undefined"}, "inningsId", 1) == 1
    assert _int_field({}, "inningsId", 2) == 2
    assert _int_field({"inningsId": "2"}, "inningsId", 1) == 2
    assert _int_field({"inningsId": 2}, "inningsId", 1) == 2
    assert _int_field({"timestamp": None}, "timestamp", 0) == 0


def test_plain_wide_with_prose_is_one_run():
    # Real 2026-09-19 commentary: scored 2 before the fix, feed said 1.
    text = "Will Jacks to Kamil Mishara, wide, the slider and it's on a length but it's down leg. Wided"
    assert _runs_and_extras(text, []) == (1, 0, {"wides": 1})



def test_counted_wides_without_comma_before_number():
    # Real 2026-09-19 Zimbabwe Women text; scored 0 before the fix.
    text = "Kayla Reyneke to Mugeri, 5 wides, the partnership between Mugeri and Kelis Ndhlovu is now"
    assert _runs_and_extras(text, []) == (5, 0, {"wides": 5})
    # Stray boundary tag must not override it.
    assert _runs_and_extras("Will Jacks to Charith Asalanka, 4 wides, on a length down leg", ["four"]) == (4, 0, {"wides": 4})


def test_no_ball_boundary_counts_the_no_ball_run_too():
    text = "Seshnie Naidu to Mugeri, no ball, FOUR"
    assert _runs_and_extras(text, ["four"]) == (5, 4, {"noballs": 1})


def test_run_out_counts_the_runs_completed_before_the_dismissal():
    """2026-10-11 India v West Indies 16.4: "out Tilak Varma Run Out!! 1 run completed." was read as 0 runs, so our score ran one
    behind the page and every later prediction was held."""
    from app.live.cricbuzz_reader import _runs_and_extras

    text = ("Gudakesh Motie to Axar Patel, out Tilak Varma Run Out!! 1 run completed. Such poor cricket from India. "
            "This is what happened: A low full-toss on the pads, Axar flicks to the deep backward square-leg fence "
            "Varma run out (Kamil Pooran/Shai Hope/Gudakesh Motie) 28(16) [4s-4]")
    assert _runs_and_extras(text, ["wicket"]) == (1, 1, {})
    assert _runs_and_extras(text.replace("1 run completed", "2 runs completed"), ["wicket"]) == (2, 2, {})
    # a run out with nothing completed, or any other wicket, is unchanged
    assert _runs_and_extras("Jofra Archer to Buttler, out Run Out!! Direct hit at the keeper's end.", ["wicket"]) == (0, 0, {})
    assert _runs_and_extras("Bumrah to Hope, no run, beaten outside off", []) == (0, 0, {})
    assert _runs_and_extras("Bumrah to Hope, 2 runs, driven to deep cover. 2 runs completed for sure", []) == (2, 2, {})


def test_chase_gets_an_empty_pre_first_ball_snapshot_at_the_innings_break():
    """2026-10-11, India v West Indies and every earlier chase: the reader only handed out a pre-first-ball snapshot for innings 1, so
    the over-1 prediction of the chase never fired. At the innings break the reader now hands out innings 2 with the target."""
    from app.live.cricbuzz_reader import CricbuzzLiveReader

    assert CricbuzzLiveReader._at_innings_break({"state": "Innings Break", "status": "Innings Break"})
    assert CricbuzzLiveReader._at_innings_break({"state": "In Progress", "status": "Innings break - West Indies need 223"})
    assert not CricbuzzLiveReader._at_innings_break({"state": "In Progress", "status": "West Indies opt to bowl"})
    assert not CricbuzzLiveReader._at_innings_break(None)

    reader = CricbuzzLiveReader()
    snap = reader._pre_next_innings_snapshot(
        "151587", "<html></html>", [], {"state": "Innings Break"}, ["India", "West Indies"], "T20", "India", 20,
        first_innings_score=222,
    )
    assert (snap.innings, snap.batting_team, snap.bowling_team) == (2, "West Indies", "India")
    assert (snap.score, snap.wickets, snap.overs, snap.deliveries) == (0, 0, "0.0", ())
    assert snap.target == 223 and snap.is_live and not snap.innings_complete and snap.scheduled_overs == 20


def test_wicket_ball_posted_with_empty_player_fields_takes_the_names_from_its_text():
    """2026-10-13, 7.6: the entry first came with batsmanDetails/bowlerDetails playerName "" (filled ~16 s later); the console showed
    "WICKET --  (caught)" and an empty "Bowled:" although the text names both players."""
    from app.live.cricbuzz_reader import CricbuzzLiveReader, names_from_text

    text = ("Arafat Minhas to Sineth Jayawardena, <b>out</b> Caught by Razaullah at !! "
            "<b>Sineth Jayawardena c Razaullah b Arafat Minhas 38(26) [4s-5 6s-2]</b>")
    assert names_from_text(text) == ("Arafat Minhas", "Sineth Jayawardena")
    assert names_from_text("no names here") == ("", "")
    obj = {"matchId": 1, "commType": "commentary", "inningsId": 2, "ballMetric": 0.1, "timestamp": 5, "commText": text,
           "event": ["wicket", "all"], "batsmanDetails": {"playerId": 0, "playerName": ""},
           "bowlerDetails": {"playerId": 0, "playerName": ""}}
    reader = CricbuzzLiveReader()
    delivery = reader._build_deliveries(("1", 2), [obj])[-1]
    assert (delivery.bowler, delivery.striker) == ("Arafat Minhas", "Sineth Jayawardena")
    assert delivery.dismissed == "Sineth Jayawardena"


def test_batting_side_is_remembered_when_a_copy_of_the_page_names_no_team():
    """2026-10-14: a page copy whose window had no team name fell back to the first team in the list (India) for minutes and every poll
    was refused with "Batting team changed inside innings 1"."""
    from app.live.cricbuzz_reader import CricbuzzLiveReader

    reader = CricbuzzLiveReader()
    scope = ("1", 1)
    reader._batting_team_by_scope[scope] = "West Indies"
    named = [{"teamName": "WI", "timestamp": 1}]
    unnamed = [{"teamName": "", "timestamp": 2}]
    teams = ["India", "West Indies"]
    from app.live.cricbuzz_reader import _resolve_full_team_name

    assert _resolve_full_team_name("WI", teams) == "West Indies"
    women = ["India Women", "Zimbabwe Women"]
    assert _resolve_full_team_name("ZIMW", women) == "Zimbabwe Women" and _resolve_full_team_name("INDW", women) == "India Women"
    assert _resolve_full_team_name("SAW", ["England Women", "South Africa Women"]) == "South Africa Women"
    assert _resolve_full_team_name("", teams) == ""   # no code: the caller must use what it remembers, not teams[0]


def test_empty_twin_of_a_ball_is_ignored_and_separators_are_reattached_to_their_over():
    """2026-10-14: (1) a blank entry with the same ballMetric as a FOUR arrived 4 s later and was read as a revision to 0 runs;
    (2) over separators now arrive as separate {"commType":"over-separator"} objects and the code that reads them never saw them."""
    from app.live.cricbuzz_reader import _delivery_objects

    ball = ('{"matchId":9,"commType":"commentary","inningsId":1,"ballMetric":0.6,"timestamp":100,'
            '"commText":"A to B, <b>FOUR</b>","event":["four","all"],"teamName":"WI"}')
    blank = ('{"matchId":9,"commType":"commentary","inningsId":1,"ballMetric":0.6,"timestamp":104,'
             '"commText":"","event":["none","all"],"teamName":"WI"}')
    separator = ('{"commType":"over-separator","matchId":9,"inningsId":1,"timestamp":120,'
                 '"overSeparator":{"overNumber":1,"overSummary":"0 0 0 0 0 4 ","overRuns":4,"batTeamObj":{"teamName":"WI","teamScore":"4-0"}}}')
    objects = _delivery_objects([ball, blank, separator], "9")
    assert len(objects) == 1 and objects[0]["timestamp"] == 100            # the blank twin is gone
    assert objects[0]["overSeparator"]["batTeamObj"]["teamScore"] == "4-0"  # the separator sits on the last ball of over 1


def test_an_over_never_seen_ball_by_ball_is_built_from_cricbuzz_over_summary():
    """User 2026-10-20: the over summary is the official record. An earlier over that is only in the page as a summary (missed live) is
    rebuilt from it, and the running totals of everything after it stay consistent."""
    from app.live.cricbuzz_reader import CricbuzzLiveReader
    from app.live.live_snapshot import LiveDelivery
    from app.live.over_summary import parse_over_summary

    def ball(over, n, runs, total):
        return LiveDelivery("1", 1, over, n, "Bowler B", "Batter", runs, runs, {}, None, total, 0, over * 100 + n, "text")

    reader = CricbuzzLiveReader()
    reader._over_truth = {}
    reader._over_truth[("1", 1, 1)] = {"tokens": parse_over_summary("1 0 4 W 1 1"), "checkpoint": (7, 1), "bowler": "Bowler A",
                                       "summary": "1 0 4 W 1 1", "timestamp_ms": 1000}
    # over 2 is the only one we saw ball by ball: 6 balls of 1 run, the score after it 13/1
    known = {(1, 2, n): ball(2, n, 1, 7 + n) for n in range(1, 7)}
    # (over 1 has 7 runs and a wicket; we joined during over 2)
    changed = reader._reconcile_overs(("1", 1), known, {})
    assert {d.over for d in known.values()} == {1, 2}
    over_one = sorted((d for d in known.values() if d.over == 1), key=lambda d: d.ball)
    assert [d.total_runs for d in over_one] == [1, 0, 4, 0, 1, 1] and over_one[3].wicket_kind and over_one[0].bowler == "Bowler A"
    assert changed == (13, 1)                                            # the corrected running totals after the last ball
    last = max(known.values(), key=lambda d: d.key)
    assert (last.feed_total, last.feed_wickets) == (13, 1)


def test_at_the_crease_entry_is_not_a_delivery_and_nothing_is_kept_beyond_the_format_length():
    """2026-10-20: "<batters> are at the crease" carries ballMetric 0 and became "over 1 ball 0" ("Verification error: Impossible ball
    number 1.0" on every poll). A T20 innings is 20 overs and an ODI innings 50; a super over (innings 3+) is its own scope."""
    from app.live.cricbuzz_reader import _delivery_objects, _within_scheduled_overs

    intro = ('{"matchId":9,"commType":"commentary","inningsId":1,"ballMetric":0,"timestamp":90,'
             '"commText":"<b>A and B are at the crease. C will open the attack</b>","event":["none","all"]}')
    first = '{"matchId":9,"commType":"commentary","inningsId":1,"ballMetric":0.1,"timestamp":100,"commText":"C to A, no run","event":["none","all"]}'
    objects = _delivery_objects([intro, first], "9")
    assert [o["ballMetric"] for o in objects] == ["$undefined", 0.1]

    def obj(innings, metric):
        return {"inningsId": innings, "ballMetric": metric}

    mixed = [obj(1, 19.6), obj(1, 20.1), obj(2, 20.2), obj(3, 0.1), obj(2, "$undefined")]
    assert [(o["inningsId"], o["ballMetric"]) for o in _within_scheduled_overs(mixed, 20)] == [(1, 19.6), (3, 0.1), (2, "$undefined")]
    assert len(_within_scheduled_overs(mixed, 50)) == 5 and len(_within_scheduled_overs(mixed, 0)) == 5
    assert [o["ballMetric"] for o in _within_scheduled_overs([obj(2, 49.6), obj(2, 50.1)], 50)] == [49.6]


def test_short_names_from_commentary_text_become_the_full_name_cricbuzz_uses():
    """2026-10-20: "Mayers" and "Sharne Mayers" were two batters in our record, so per-batter figures never matched the scorecard."""
    from app.live.cricbuzz_reader import CricbuzzLiveReader

    reader = CricbuzzLiveReader()
    reader._learn_roster([
        {"batsmanDetails": {"playerName": "Sharne Mayers"}, "bowlerDetails": {"playerName": "Kranti Gaud"}},
        {"batsmanDetails": {"playerName": "Chipo Mugeri-Tiripano"}},
        {"overSeparator": {"batStrikerObj": {"playerName": "Beloved Biza"}, "batNonStrikerObj": {"playerName": "Nyasha Gwanzura"}}},
    ])
    assert reader._canonical_name("Mayers") == "Sharne Mayers"
    assert reader._canonical_name("Mugeri") == "Chipo Mugeri-Tiripano"
    assert reader._canonical_name("Sharne Mayers") == "Sharne Mayers" and reader._canonical_name("") == ""
    reader._roster.add("Sharne Sharma"); reader._roster.add("Deepti Sharma")
    assert reader._canonical_name("Sharma") == "Sharma"          # two players share it: do not guess
    assert reader._canonical_name("Unknown Person") == "Unknown Person"


def test_first_forecast_is_released_five_minutes_before_the_scheduled_start(monkeypatch):
    """User 2026-10-24: "it should be 5 mins before the match" (the over-1 forecast came out at the start time itself)."""
    import pytest

    from app.live import cricbuzz_reader
    from app.live.cricbuzz_reader import CricbuzzInningsNotStarted, CricbuzzLiveReader

    start_ms = 1_800_000_000_000
    reader = CricbuzzLiveReader()
    monkeypatch.setattr(reader, "_start_ms_from_html", lambda html: start_ms)
    monkeypatch.setattr(cricbuzz_reader.time, "time", lambda: (start_ms - 6 * 60_000) / 1000)
    with pytest.raises(CricbuzzInningsNotStarted, match="5 minutes before the start"):
        reader._pre_first_ball_snapshot("1", "<html></html>", [])
    monkeypatch.setattr(cricbuzz_reader.time, "time", lambda: (start_ms - 4 * 60_000) / 1000)
    with pytest.raises(CricbuzzInningsNotStarted, match="toss/batting side"):   # past the time gate: now it waits only for the toss
        reader._pre_first_ball_snapshot("1", "<html></html>", [])
