import json

import pytest
import requests

from app.live.cricketdata_reader import (
    CricketDataCapabilityError,
    CricketDataError,
    CricketDataReader,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append((url, params, timeout))
        return self.response


@pytest.fixture
def credentials(tmp_path):
    path = tmp_path / ".cricketdata.json"
    path.write_text(json.dumps({"api_key": "secret-test-key"}), encoding="utf-8")
    return path


def test_parses_current_match_without_exposing_key(credentials):
    payload = {
        "status": "success",
        "info": {"hitsToday": 3, "hitsUsed": 1, "hitsLimit": 100},
        "data": [
            {
                "id": "match-1",
                "name": "Zimbabwe vs India",
                "matchType": "t20",
                "status": "India won",
                "matchStarted": True,
                "matchEnded": True,
                "score": [
                    {"r": 125, "w": 7, "o": 20, "inning": "Zimbabwe Inning 1"},
                    {"r": 126, "w": 3, "o": 13.2, "inning": "India Inning 1"},
                ],
            }
        ],
    }
    session = FakeSession(FakeResponse(payload))
    reader = CricketDataReader(credentials, session=session)

    match = reader.current_matches()[0]

    assert match.match_id == "match-1"
    assert match.innings[1].runs == 126
    assert match.usage.hits_limit == 100
    assert "secret-test-key" not in repr(match)
    assert session.calls[0][2] == 10.0


def test_refuses_candidate_v3_without_deliveries(credentials):
    payload = {
        "status": "success",
        "info": {},
        "data": [{"id": "match-1", "name": "Zimbabwe vs India"}],
    }
    match = CricketDataReader(
        credentials, session=FakeSession(FakeResponse(payload))
    ).current_matches()[0]

    with pytest.raises(CricketDataCapabilityError, match="cannot feed Candidate v3"):
        match.require_ball_by_ball()


def test_provider_failure_is_explicit(credentials):
    reader = CricketDataReader(
        credentials,
        session=FakeSession(FakeResponse({"status": "failure", "reason": "hits limit"})),
    )

    with pytest.raises(CricketDataError, match="hits limit"):
        reader.current_matches()
