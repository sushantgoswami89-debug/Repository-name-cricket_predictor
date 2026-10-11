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
