"""Unit tests for the Cricbuzz reader's toss/venue/squad extraction,
added 2026-09-18 to close the "team_players/toss enrichment not ported
here yet" gap flagged in app/live/cricbuzz_reader.py's own module
docstring. Fixtures are shaped exactly like the real flight-chunk JSON
confirmed live against a real match (Australia vs Zimbabwe, 2nd ODI,
2026-09-18) -- see that module's docstrings for the full field-by-field
confirmation.
"""

from __future__ import annotations

from app.live.cricbuzz_reader import CricbuzzLiveReader


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
