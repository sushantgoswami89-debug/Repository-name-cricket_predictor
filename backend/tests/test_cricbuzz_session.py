"""Unit tests for CricbuzzMatchSession, extracted 2026-09-18 from
run_cricbuzz_match.py so run_cricbuzz_auto.py's multi-match discovery
driver can reuse the same real-feed-inconsistency defenses per match.
These lock in behavior that was previously only exercised by hand
against a real live match.
"""

from __future__ import annotations

import json

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
    assert "Zimbabwe vs Australia -- Match Briefing" in text
    assert "Toss: Zimbabwe won, chose to bowling" in text
    assert "Venue: Harare Sports Club, Harare" in text


def test_briefing_text_falls_back_to_given_label_without_ldjson(monkeypatch):
    html = _briefing_html().replace("SportsEvent", "SomethingElse")
    monkeypatch.setattr(
        "app.live.cricbuzz_session.requests.Session", lambda: _FakeBriefingSession(html)
    )

    text = briefing_text("152742", "https://www.cricbuzz.com/live-cricket-scores/152742/x", "my-label")

    assert text is not None
    assert "my-label -- Match Briefing" in text


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
