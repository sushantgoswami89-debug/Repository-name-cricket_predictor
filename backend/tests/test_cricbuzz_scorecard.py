"""Scorecard page parsing, the scorecard cross-check, the match-header end signal and the dismissed field."""

import json
from types import SimpleNamespace

from app.live.cricbuzz_reader import CricbuzzLiveReader, dismissed_batter
from app.live.cricbuzz_scorecard import compare_with_deliveries, parse_scorecard
from app.live.cricbuzz_session import _header_result, _match_result_text, _match_winner
from app.live.live_snapshot import LiveDelivery


def _page(payload: dict) -> str:
    """Wrap JSON the way Cricbuzz's Next.js flight data carries it."""
    text = ",".join(f'"{k}":{json.dumps(v)}' for k, v in payload.items())
    return '<script>self.__next_f.push([1,' + json.dumps(text) + '])</script>'


SCORECARD = {
    "scoreCard": [
        {
            "inningsId": 1,
            "batTeamDetails": {"batTeamName": "India", "batsmenData": {
                "bat_1": {"batName": "A Sharma", "runs": 20, "balls": 12, "fours": 2, "sixes": 1, "outDesc": "c X b Y", "wicketCode": "CAUGHT"},
                "bat_2": {"batName": "S Samson", "runs": 21, "balls": 10, "fours": 3, "sixes": 0, "outDesc": "not out", "wicketCode": ""},
            }},
            "bowlTeamDetails": {"bowlTeamName": "West Indies", "bowlersData": {
                "bowl_1": {"bowlName": "A Hosein", "overs": 3, "maidens": 0, "runs": 46, "wickets": 1, "no_balls": 1, "wides": 0}}},
            "scoreDetails": {"runs": 41, "wickets": 1, "overs": 3},
            "extrasData": {"total": 0},
            "wicketsData": {"wkt_1": {"batName": "A Sharma", "wktOver": 1.2, "wktRuns": 20, "wktNbr": 1}},
        }
    ],
    "matchHeader": {"state": "Complete", "status": "West Indies won by 6 wkts", "complete": True,
                    "result": {"resultType": "win", "winningTeam": "West Indies"}},
}


def _delivery(over, ball, striker, runs, total_after, wickets_after, wicket=None, dismissed=""):
    return LiveDelivery(
        match_id="1", innings=1, over=over, ball=ball, bowler="Y", striker=striker, total_runs=runs, batter_runs=runs,
        extras={}, wicket_kind=wicket, feed_total=total_after, feed_wickets=wickets_after, timestamp_ms=over * 10 + ball,
        commentary="", dismissed=dismissed,
    )


def test_scorecard_page_is_parsed_into_cards_extras_fall_and_header():
    card = parse_scorecard(_page(SCORECARD))
    assert card["header"] == {"state": "Complete", "status": "West Indies won by 6 wkts", "complete": True,
                              "result_type": "win", "winner": "West Indies"}
    inn = card["innings"][0]
    assert (inn["team"], inn["runs"], inn["wickets"]) == ("India", 41, 1)
    assert inn["batting"][0] == {"name": "A Sharma", "runs": 20, "balls": 12, "fours": 2, "sixes": 1,
                                 "out": "c X b Y", "wicket_code": "CAUGHT"}
    assert inn["fall"] == [{"batter": "A Sharma", "over": 1.2, "runs": 20, "number": 1}]
    assert parse_scorecard("<html>no flight data</html>") is None


def test_cross_check_agrees_with_a_clean_innings_and_flags_differences():
    card = parse_scorecard(_page(SCORECARD))
    clean = [_delivery(1, 1, "A Sharma", 20, 20, 0), _delivery(1, 2, "A Sharma", 0, 20, 1, "caught", "A Sharma"),
             _delivery(1, 3, "S Samson", 21, 41, 1)]
    assert compare_with_deliveries(card, clean, 1) == []
    wrong_total = clean[:2] + [_delivery(1, 3, "S Samson", 15, 35, 1)]
    problems = compare_with_deliveries(card, wrong_total, 1)
    assert any("total ours 35/1 vs card 41/1" in p for p in problems) and any("S Samson ours 15 vs card 21" in p for p in problems)
    # attached mid-innings: only the total is comparable (per-batter figures would be partial)
    mid = [_delivery(3, 1, "S Samson", 21, 41, 1)]  # starts at over 3 ball 1, not the innings start
    assert compare_with_deliveries(card, mid, 1) == []


def test_header_decides_the_result_and_the_winner():
    done = SimpleNamespace(innings=2, target=250, score=100, wickets=3, innings_complete=False, batting_team="X", bowling_team="Y",
                           board={"state": "Complete", "complete": True, "status": "West Indies won by 6 wkts",
                                  "result": {"resultType": "win", "winningTeam": "West Indies"}})
    assert _header_result(done) == ("West Indies won by 6 wkts", "West Indies")
    assert _match_result_text(done) == "West Indies won by 6 wkts" and _match_winner(done) == "West Indies"
    no_result = SimpleNamespace(**{**done.__dict__, "board": {"state": "Complete", "complete": True, "status": "No result",
                                                              "result": {"resultType": "noresult"}}})
    assert _match_result_text(no_result) == "No result" and _match_winner(no_result) is None
    live = SimpleNamespace(**{**done.__dict__, "board": {"state": "In Progress", "complete": False}})
    assert _header_result(live) is None and _match_result_text(live) is None  # falls back to the score/target inference


def test_dismissed_batter_handles_run_outs_of_the_non_striker():
    assert dismissed_batter(None, "no run", "Hope") == ""
    assert dismissed_batter("caught", "out Caught by X!!", "Hope") == "Hope"
    assert dismissed_batter("run out", "out Shimron Hetmyer Run Out by Y", "Shai Hope") == "Shimron Hetmyer"


def test_reader_sets_dismissed_and_match_complete_from_the_board():
    reader = CricbuzzLiveReader()
    obj = {"matchId": 1, "commType": "commentary", "commText": "Y to Shai Hope, out Shimron Hetmyer Run Out", "inningsId": 1,
           "event": ["wicket", "all"], "ballMetric": 0.1, "teamName": "WI", "timestamp": 5, "overSeparator": None,
           "batsmanDetails": {"playerName": "Shai Hope"}, "bowlerDetails": {"playerName": "Y"}}
    [delivery] = reader._build_deliveries(("1", 1), [obj])
    assert delivery.wicket_kind == "run out" and delivery.dismissed == "Shimron Hetmyer"


def test_scorecard_total_is_compared_at_the_cards_own_over_not_with_our_newest_ball():
    """2026-10-20: "[card] 19.0 ov: MISMATCH: total ours 206/6 vs card 202/6" while our over was exact: the card was still at 18.5."""
    from types import SimpleNamespace

    from app.live.cricbuzz_scorecard import compare_with_deliveries, state_at_overs

    def ball(over, n, total):
        return SimpleNamespace(innings=1, over=over, ball=n, is_legal=True, feed_total=total, feed_wickets=6, key=(1, over, n),
                               striker="A", batter_runs=0, total_runs=0, wicket_kind=None, dismissed="")

    ours = [ball(19, n, 196 + n) for n in range(1, 5)] + [ball(19, 5, 202), ball(19, 6, 206)]
    card = {"innings": [{"innings": 1, "runs": 202, "wickets": 6, "overs": 18.5, "batting": [], "fall": []}]}
    assert state_at_overs(ours, 18.5) == (202, 6)
    assert compare_with_deliveries(card, ours, 1) == []                      # card at 18.5 with 202/6: it agrees with us at that ball
    card["innings"][0]["runs"] = 201
    assert compare_with_deliveries(card, ours, 1) != []                      # a real disagreement at its own ball is still reported


def test_batters_are_not_compared_when_an_over_was_rebuilt_from_the_summary_without_strikers():
    """2026-10-20: a restart in over 1 back-filled it (no striker names) and the card then 'disagreed' on every batter and wicket."""
    card = parse_scorecard(_page(SCORECARD))
    backfilled = [_delivery(1, 1, "", 20, 20, 0), _delivery(1, 2, "", 0, 20, 1, "caught"), _delivery(1, 3, "S Samson", 21, 41, 1)]
    assert compare_with_deliveries(card, backfilled, 1) == []
