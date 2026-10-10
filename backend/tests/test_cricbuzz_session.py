"""Unit tests for CricbuzzMatchSession, extracted 2026-09-18 from
run_cricbuzz_match.py so run_cricbuzz_auto.py's multi-match discovery
driver can reuse the same real-feed-inconsistency defenses per match.
These lock in behavior that was previously only exercised by hand
against a real live match.
"""

from __future__ import annotations

import json

import pytest

from app.live.cricbuzz_reader import CricbuzzFeedError
from app.live.cricbuzz_session import (
    RESEED_AFTER_CONSECUTIVE_ERRORS,
    CricbuzzMatchSession,
    briefing_text,
)
from app.live.live_snapshot import LiveSnapshot
from app.live.pipeline import VerifiedLivePredictionPipeline
from app.ml.prediction_result import PredictionResult


def _snapshot(**overrides) -> LiveSnapshot:
    values = dict(
        match_id="152742",
        match_format="T20",
        batting_team="India",
        bowling_team="Afghanistan",
        innings=1,
        score=50,
        wickets=1,
        overs="6.0",
        is_live=True,
        deliveries=(),
        scheduled_overs=20,
    )
    values.update(overrides)
    return LiveSnapshot(**values)


class FakeEngine:
    def predict(self, context):
        return PredictionResult(7.0, 0.2, 0.9, expected_range="5-9")

    def update_actuals(self, *actuals):
        pass


class FakeReader:
    def __init__(self, snapshots_or_errors):
        self._queue = list(snapshots_or_errors)

    def fetch(self, url):
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _session(reader) -> CricbuzzMatchSession:
    return CricbuzzMatchSession(
        match_id="152742",
        match_url="https://www.cricbuzz.com/live-cricket-scores/152742/x",
        pipeline=VerifiedLivePredictionPipeline(engine=FakeEngine(), publisher=None),
        reader=reader,
    )


def test_poll_once_prints_scorecard_and_does_not_finish_mid_match():
    reader = FakeReader([_snapshot()] * 3)
    session = _session(reader)

    session.poll_once()

    assert session.finished is False


def test_stale_snapshot_is_skipped_not_processed():
    ahead = _snapshot(overs="10.0")
    behind = _snapshot(overs="4.0")
    reader = FakeReader([ahead] * 3 + [behind] * 3)
    session = _session(reader)

    session.poll_once()  # establishes best_progress at over 10
    session.poll_once()  # the "behind" snapshot must be skipped

    assert session._best_progress == (1, 60)


def test_reseeds_reader_after_consecutive_feed_errors():
    reader = FakeReader([CricbuzzFeedError("boom")] * (RESEED_AFTER_CONSECUTIVE_ERRORS * 3))
    session = _session(reader)
    original_reader = session.reader

    for _ in range(RESEED_AFTER_CONSECUTIVE_ERRORS):
        session.poll_once()

    assert session.reader is not original_reader
    assert session._consecutive_feed_errors == 0


def test_match_completion_via_target_reached_marks_finished():
    chase_won = _snapshot(
        innings=2,
        batting_team="India",
        bowling_team="Afghanistan",
        score=160,
        wickets=4,
        target=155,
        overs="18.2",
    )
    reader = FakeReader([chase_won] * 3)
    session = _session(reader)

    session.poll_once()

    assert session.finished is True


def test_match_still_live_when_target_not_reached_and_not_complete():
    chase_in_progress = _snapshot(
        innings=2, target=155, score=80, wickets=2, overs="10.0"
    )
    reader = FakeReader([chase_in_progress] * 3)
    session = _session(reader)

    session.poll_once()

    assert session.finished is False


class _FakeBriefingResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        pass


class _FakeBriefingSession:
    def __init__(self, html: str) -> None:
        self._html = html
        self.headers: dict[str, str] = {}

    def get(self, url, timeout):
        return _FakeBriefingResponse(self._html)


def _briefing_html() -> str:
    match_info = json.dumps(
        {
            "matchInfo": {
                "matchId": 152742,
                "venueInfo": {"ground": "Harare Sports Club", "city": "Harare"},
            }
        },
        separators=(",", ":"),
    )
    squads = json.dumps(
        {
            "matchId": 152742,
            "team1": {
                "id": 12,
                "name": "Zimbabwe",
                "playerDetails": [
                    {"id": 1, "name": "Player A", "role": "Batsman", "substitute": False}
                ],
            },
            "team2": {"id": 4, "name": "Australia", "playerDetails": []},
            "tossResults": {
                "tossWinnerId": 12,
                "tossWinnerName": "Zimbabwe",
                "decision": "Bowling",
            },
        },
        separators=(",", ":"),
    )
    ldjson = (
        '<script type="application/ld+json">'
        '{"@type":"SportsEvent","competitor":[{"name":"Zimbabwe"},{"name":"Australia"}]}'
        "</script>"
    )
    return (
        f'<script>self.__next_f.push([1,{json.dumps(match_info)}])</script>'
        f'<script>self.__next_f.push([2,{json.dumps(squads)}])</script>'
        f"{ldjson}"
    )


def test_briefing_text_uses_page_teams_as_label(monkeypatch):
    monkeypatch.setattr(
        "app.live.cricbuzz_session.requests.Session",
        lambda: _FakeBriefingSession(_briefing_html()),
    )

    text = briefing_text("152742", "https://www.cricbuzz.com/live-cricket-scores/152742/x", "152742")

    assert text is not None
    assert text.startswith("🏏 Zimbabwe vs Australia")
    assert "Toss: Zimbabwe won, chose to bowl -> Australia bat first" in text
    assert "-- Harare | " in text.splitlines()[0]
    # the header must really be there: briefing_text swallows header errors, so a bug would be silent
    assert "Harare | " in text.splitlines()[0] and "| T20 - " in text.splitlines()[0]
    assert text.splitlines()[1].startswith("Weather:")
    lines = text.splitlines()
    assert lines[-1].startswith("XIs:") or lines[-3].startswith("XIs:")  # line-up block is last
    assert not any(l.startswith("XIs:") for l in lines[:6])
    assert any(", Pitch: " in line and line.startswith("Dew:") for line in text.splitlines())


@pytest.fixture(autouse=True)
def _no_weather_network(monkeypatch, tmp_path):
    monkeypatch.setattr("app.live.match_weather.match_day_weather", lambda city, day=None: None)
    monkeypatch.setattr("app.live.prematch_analysis.toi_pitch_note", lambda a, b: "")
    from app.live import prematch_analysis, prematch_log

    prematch_analysis._TTL.clear()
    monkeypatch.setattr(prematch_log, "LOG_PATH", tmp_path / "pred.jsonl")
    monkeypatch.setattr(prematch_log, "RESULTS_PATH", tmp_path / "res.jsonl")


def test_briefing_text_falls_back_to_given_label_without_ldjson(monkeypatch):
    html = _briefing_html().replace("SportsEvent", "SomethingElse")
    monkeypatch.setattr(
        "app.live.cricbuzz_session.requests.Session", lambda: _FakeBriefingSession(html)
    )

    text = briefing_text("152742", "https://www.cricbuzz.com/live-cricket-scores/152742/x", "my-label")

    assert text is not None
    assert text.startswith("🏏 my-label")


def test_briefing_text_returns_none_on_fetch_failure(monkeypatch):
    class _FailingSession:
        headers: dict[str, str] = {}

        def get(self, url, timeout):
            raise ConnectionError("boom")

    monkeypatch.setattr(
        "app.live.cricbuzz_session.requests.Session", lambda: _FailingSession()
    )

    assert briefing_text("152742", "https://example.com", "label") is None


class _RecordingPublisher:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def publish(self, key, text):
        self.sent.append((key, text))
        return True


def test_stuck_alert_fires_once_after_threshold_and_resets_on_recovery(monkeypatch):
    publisher = _RecordingPublisher()
    reader = FakeReader([CricbuzzFeedError("boom")] * 100)
    session = CricbuzzMatchSession(
        match_id="152742",
        match_url="https://www.cricbuzz.com/live-cricket-scores/152742/x",
        pipeline=VerifiedLivePredictionPipeline(engine=FakeEngine(), publisher=None),
        reader=reader,
        publisher=publisher,
        stuck_alert_seconds=60.0,
    )

    clock = [1000.0]
    monkeypatch.setattr("app.live.cricbuzz_session.time.time", lambda: clock[0])

    session.poll_once()  # starts the error streak at t=1000
    assert publisher.sent == []

    clock[0] = 1000.0 + 61.0  # past the 60s threshold
    session.poll_once()
    assert len(publisher.sent) == 1
    key, text = publisher.sent[0]
    assert key.startswith("stuck_alert:152742:")
    assert "erroring" in text
    assert "resume automatically" in text

    clock[0] += 5.0
    session.poll_once()  # still erroring -- must not repeat
    assert len(publisher.sent) == 1

    # A real snapshot arrives: the streak resets. (The 3rd consecutive
    # failure above also triggered a reseed, replacing session.reader with
    # a brand-new real CricbuzzLiveReader() -- reinstall the fake so this
    # test keeps controlling what fetch() returns, same as production
    # would keep using whatever fresh reader it just reseeded to.)
    session.reader = reader
    reader._queue = [_snapshot()] * 3
    session.poll_once()
    assert session._error_streak_started_at is None
    assert session._stuck_alert_sent is False

    # Erroring again later must be able to alert a second time.
    session.reader = reader
    reader._queue = [CricbuzzFeedError("boom again")] * 3
    clock[0] += 1.0
    session.poll_once()
    clock[0] += 61.0
    session.reader = reader
    reader._queue = [CricbuzzFeedError("boom again")] * 3
    session.poll_once()
    assert len(publisher.sent) == 2


def test_run_out_event_names_the_dismissed_non_striker():
    from types import SimpleNamespace

    from app.live.cricbuzz_session import _dismissed_batter

    run_out = SimpleNamespace(
        wicket_kind="run out",
        striker="Shai Hope",
        commentary="Axar Patel to Shai Hope, out Shimron Hetmyer Run Out!! Shimron Hetmyer run out (Tilak Varma/Sanju Samson) 5(2)",
    )
    assert _dismissed_batter(run_out) == "Shimron Hetmyer"
    bowled = SimpleNamespace(wicket_kind="bowled", striker="Rovman Powell", commentary="out Bowled!!")
    assert _dismissed_batter(bowled) == "Rovman Powell"


def test_newest_board_ignores_stale_cached_copies():
    from app.live.cricbuzz_session import _newest_board

    fresh = {"innings": 2, "last_updated_s": 2000, "overs": 12.0, "striker": "Iyer"}
    stale = {"innings": 2, "last_updated_s": 1000, "overs": 8.1, "striker": "Kishan"}
    assert _newest_board(fresh, stale) is fresh
    assert _newest_board(stale, fresh) is fresh
    assert _newest_board(None, stale) is stale
    assert _newest_board(fresh, None) is fresh


def test_next_over_line_uses_announced_bowler_and_last_ball_parity():
    from types import SimpleNamespace

    from app.live.cricbuzz_session import _announced_bowler_name, _next_over_line

    assert _announced_bowler_name("Matthew Forde [1.0-0-15-0] is back into the attack") == "Matthew Forde"
    assert _announced_bowler_name("Naman Dhir, right-arm off break, comes into the attack") == "Naman Dhir"

    def snap(last_runs, batter_runs, wicket=None):
        last = SimpleNamespace(
            timestamp_ms=1000, over=13, innings=2, dismissed="", striker="Iyer", total_runs=last_runs, batter_runs=batter_runs, wicket_kind=wicket
        )
        return SimpleNamespace(
            overs="13.0", innings_complete=False, deliveries=[last],
            board={"striker": "Iyer", "non_striker": "Kishan"},
        )

    # SIX on the last ball (real case at 13.6): no crossing, ends change -> Kishan faces
    line = _next_over_line(snap(6, 6), (2000, "Akeal Hosein"))
    assert "Akeal Hosein bowling" in line and "Kishan* expected to face" in line
    # single on the last ball: batters crossed, then ends changed -> Iyer faces again
    assert "Iyer* expected to face" in _next_over_line(snap(1, 1), None)
    assert "not announced yet" in _next_over_line(snap(1, 1), (500, "Old Bowler"))
    # result is the same whether or not the board already swapped
    swapped = snap(6, 6); swapped.board = {"striker": "Kishan", "non_striker": "Iyer"}
    assert "Kishan* expected to face" in _next_over_line(swapped, None)
    assert "see board" in _next_over_line(snap(0, 0, "bowled"), None)
    mid = snap(1, 1); mid.overs = "13.2"
    assert _next_over_line(mid, None) == ""


def test_announced_bowler_goes_on_snapshot_only_when_posted_after_last_ball():
    from dataclasses import replace as _replace
    from app.live.cricbuzz_session import _with_announced_bowler
    from app.live.live_snapshot import LiveDelivery, LiveSnapshot

    last = LiveDelivery(
        match_id="1", innings=2, over=13, ball=6, bowler="A", striker="S", total_runs=1,
        batter_runs=1, extras={}, wicket_kind=None, feed_total=140, feed_wickets=2,
        timestamp_ms=1000, commentary="",
    )
    snap = LiveSnapshot(
        match_id="1", match_format="T20I", batting_team="X", bowling_team="Y", innings=2,
        score=140, wickets=2, overs="13.0", is_live=True, deliveries=(last,),
    )
    assert _with_announced_bowler(snap, (999, "B")).announced_bowler == ""
    assert _with_announced_bowler(snap, None).announced_bowler == ""
    got = _with_announced_bowler(snap, (1500, "B"))
    assert (got.announced_bowler, got.announced_bowler_over) == ("B", 14)
    mid = _replace(snap, overs="13.3")
    assert _with_announced_bowler(mid, (1500, "B")).announced_bowler == ""


def test_prediction_held_only_when_board_disagrees_at_the_same_point():
    from types import SimpleNamespace

    from app.live.cricbuzz_session import _prediction_hold

    def snap(board, overs="13.0", complete=False):
        return SimpleNamespace(
            innings=2, overs=overs, score=140, wickets=2, innings_complete=complete, board=board
        )

    same = {"innings": 2, "overs": "13.0", "score": 140, "wickets": 2}
    assert _prediction_hold(snap(same)) is None
    assert "board says 140/3" in _prediction_hold(snap({**same, "wickets": 3}))
    # an older cached copy (fewer overs) or a newer one never holds
    assert _prediction_hold(snap({**same, "overs": "12.5", "score": 137})) is None
    assert _prediction_hold(snap({**same, "overs": "13.2", "score": 146})) is None
    # innings over: record the outcome regardless
    assert _prediction_hold(snap({**same, "wickets": 3}, complete=True)) is None
    assert _prediction_hold(snap(None)) is None


def test_next_over_expectation_is_checked_against_the_first_real_ball():
    from types import SimpleNamespace

    from app.live.cricbuzz_session import _check_next_over

    def ball(over, ball_no, striker, bowler):
        return SimpleNamespace(over=over, key=(2, over, ball_no), striker=striker, bowler=bowler)

    expected = {"over": 14, "bowler": "Forde", "striker": "Iyer", "other": "Kishan"}
    waiting = SimpleNamespace(deliveries=(ball(13, 6, "Kishan", "Hosein"),))
    assert _check_next_over(expected, waiting) is None
    hit = SimpleNamespace(deliveries=(ball(13, 6, "Kishan", "Hosein"), ball(14, 1, "Iyer", "Forde")))
    line = _check_next_over(expected, hit)
    assert "striker expected Iyer, actual Iyer: HIT" in line and "bowler announced Forde, actual Forde: HIT" in line
    miss = SimpleNamespace(deliveries=(ball(14, 1, "Kishan", "Springer"),))
    assert _check_next_over(expected, miss).count("MISS") == 2
    assert _check_next_over(None, hit) is None


def _over_break_snapshot(striker="Kishan", last_runs=1, wicket=None, dismissed="", board_pair=("Kishan", "Iyer")):
    from types import SimpleNamespace

    last = SimpleNamespace(
        timestamp_ms=1000, over=13, innings=2, striker=striker, bowler="Hosein", total_runs=last_runs,
        batter_runs=last_runs, wicket_kind=wicket, dismissed=dismissed, commentary="",
    )
    return SimpleNamespace(
        overs="13.0", innings_complete=False, deliveries=[last], batting_team="India", score=140, wickets=2,
        board={"striker": board_pair[0], "non_striker": board_pair[1], "bowler": "Hosein"},
    )


def test_over_break_line_shows_no_stale_bowler_or_batter_names():
    from app.live.cricbuzz_session import _scorecard_line

    line = _scorecard_line(_over_break_snapshot())
    assert "over complete (last ball: Hosein to Kishan)" in line
    assert "NOW" not in line and "on strike" not in line   # the board's names are the finished over's
    mid = _over_break_snapshot()
    mid.overs = "13.2"
    assert "[NOW: Hosein* bowling, Kishan* on strike" in _scorecard_line(mid)  # mid-over the board is current


def test_next_striker_uses_the_card_pair_and_drops_dismissed_batters():
    from types import SimpleNamespace

    from app.live.cricbuzz_session import _expected_next_over

    # last ball a single by Kishan: batters crossed, then the ends change for the new over -> Kishan faces again
    snap = _over_break_snapshot(striker="Kishan", last_runs=1)
    got = _expected_next_over(snap, None, crease=["Kishan", "Iyer"])
    assert (got["striker"], got["other"], got["over"]) == ("Kishan", "Iyer", 14)
    # even run: the other batter faces
    assert _expected_next_over(_over_break_snapshot(striker="Kishan", last_runs=2), None, crease=["Kishan", "Iyer"])["striker"] == "Iyer"
    # a stale board still listing a batter who is out must not be trusted (the 10-09 miss: expected Kishan, Dube faced)
    stale = _over_break_snapshot(striker="Dube", last_runs=0, board_pair=("Kishan", "Dube"))
    stale.deliveries.insert(0, SimpleNamespace(timestamp_ms=900, over=11, innings=2, striker="Kishan", bowler="Chase",
                                               total_runs=0, batter_runs=0, wicket_kind="bowled", dismissed="Kishan", commentary=""))
    assert _expected_next_over(stale, None)["striker"] == ""   # only one batter left in the pair -> says so instead of guessing
    # a wicket on the last ball: nothing derived
    assert _expected_next_over(_over_break_snapshot(wicket="caught", dismissed="Kishan"), None, crease=["Iyer", "New"])["striker"] == ""


def test_live_line_shows_exact_batter_and_bowler_figures_from_the_board():
    from types import SimpleNamespace

    from app.live.cricbuzz_session import _board_players

    board = {
        "striker": "Shai Hope", "non_striker": "Sherfane Rutherford", "bowler": "Mayank Yadav",
        "striker_stats": {"runs": 38, "balls": 19, "fours": 4, "sixes": 2},
        "non_striker_stats": {"runs": 19, "balls": 7},
        "bowler_stats": {"overs": 3, "maidens": 0, "runs": 36, "wickets": 3},
    }
    line = _board_players(SimpleNamespace(board=board))
    assert "[NOW: Mayank Yadav* 3-0-36-3 bowling, Shai Hope* 38(19) on strike, Sherfane Rutherford 19(7) at the other end]" in line
    board.pop("striker_stats")
    assert "Shai Hope* on strike" in _board_players(SimpleNamespace(board=board))  # no figures -> names only, as before


def test_over_break_figures_come_from_the_fresh_card():
    from app.live.cricbuzz_session import _over_break_figures

    snap = _over_break_snapshot(striker="Hope")
    text = _over_break_figures(
        snap, ["Hope", "Shepherd"], {"Hope": " 102*(45)", "Shepherd": " 34*(13)"},
        {"Hosein": " 4-0-37-1", "Forde": " 2-0-39-0"}, announced="Forde",
    )
    assert text == "   [JUST BOWLED: Hosein 4-0-37-1 | AT THE CREASE: Hope 102*(45), Shepherd 34*(13) | NEXT BOWLER: Forde 2-0-39-0]"
    no_card = _over_break_figures(snap, None, {}, {})
    assert no_card == "   [JUST BOWLED: Hosein]"   # without a fresh card nothing is invented
    assert "(not bowled yet)" in _over_break_figures(snap, None, {}, {}, announced="Springer")


def test_board_parsing_keeps_the_exact_figures():
    import json as _json

    from app.live.cricbuzz_reader import CricbuzzLiveReader

    mini = {"inningsId": 2, "batTeam": {"teamScore": 86, "teamWkts": 3}, "overs": 7,
            "batsmanStriker": {"name": "Shai Hope", "runs": 38, "balls": 19, "fours": 4, "sixes": 2, "strikeRate": "200.00", "id": 1},
            "batsmanNonStriker": {"name": "S Rutherford", "runs": 19, "balls": 7},
            "bowlerStriker": {"name": "Mayank Yadav", "overs": 3, "maidens": 0, "runs": 36, "wickets": 3, "economy": 12, "id": 2},
            "bowlerNonStriker": {"name": "Shivam Dube", "overs": 1, "maidens": 0, "runs": 14, "wickets": 0}}
    board = CricbuzzLiveReader._board_from_chunks(['{"miniscore":' + _json.dumps(mini) + "}"])
    assert board["striker_stats"] == {"runs": 38, "balls": 19, "fours": 4, "sixes": 2, "strikeRate": "200.00"}
    assert board["bowler_stats"] == {"overs": 3, "maidens": 0, "runs": 36, "wickets": 3, "economy": 12}
    assert board["other_bowler_stats"]["wickets"] == 0 and board["non_striker_stats"]["balls"] == 7
