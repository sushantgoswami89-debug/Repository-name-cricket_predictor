"""Unit tests for CricbuzzMatchDiscovery, built 2026-09-18 to close the
"Cricbuzz match discovery still hardcoded to one URL" punch-list gap.
Fixtures are shaped exactly like the real matchInfo objects confirmed
live against the actual index page (https://www.cricbuzz.com/cricket-match/live-scores)
on 2026-09-18.
"""

from __future__ import annotations

import json

from app.live.cricbuzz_discovery import CricbuzzMatchDiscovery, is_t20_international


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        pass


class FakeSession:
    def __init__(self, html: str) -> None:
        self._html = html

    def get(self, url, timeout):
        return FakeResponse(self._html)


def _match_info_html(**overrides) -> str:
    info = {
        "matchId": 152742,
        "seriesId": 11997,
        "seriesName": "Australia tour of Zimbabwe 2026",
        "matchDesc": "2nd ODI",
        "matchFormat": "ODI",
        "state": "In Progress",
        "status": "Zimbabwe opt to bowl",
        "team1": {"teamId": 4, "teamName": "Australia"},
        "team2": {"teamId": 12, "teamName": "Zimbabwe"},
        "venueInfo": {"ground": "Harare Sports Club", "city": "Harare"},
        "matchType": "International",
    }
    info.update(overrides)
    payload = json.dumps({"matchInfo": info}, separators=(",", ":"))
    # Same self.__next_f.push([id, "<json-escaped string>"]) shape the real
    # page uses -- json.dumps(payload) re-escapes it as a JSON string
    # literal, matching what _extract_flight_chunks expects to json.loads.
    return f'<script>self.__next_f.push([1,{json.dumps(payload)}])</script>'


def _discovery(html: str) -> CricbuzzMatchDiscovery:
    discovery = CricbuzzMatchDiscovery()
    discovery._session = FakeSession(html)
    return discovery


def test_parses_a_real_shaped_match_info_object():
    matches = _discovery(_match_info_html()).matches()

    assert len(matches) == 1
    match = matches[0]
    assert match.match_id == "152742"
    assert match.label == "Australia vs Zimbabwe"
    assert match.venue == "Harare Sports Club, Harare"
    assert match.url == "https://www.cricbuzz.com/live-cricket-scores/152742/x"
    assert match.is_live is True
    assert match.is_upcoming is False


def test_state_preview_is_upcoming_not_live():
    matches = _discovery(_match_info_html(state="Preview")).matches()
    assert matches[0].is_live is False
    assert matches[0].is_upcoming is True


def test_state_toss_counts_as_live():
    matches = _discovery(_match_info_html(state="Toss")).matches()
    assert matches[0].is_live is True


def test_state_complete_is_neither_live_nor_upcoming():
    matches = _discovery(_match_info_html(state="Complete")).matches()
    assert matches[0].is_live is False
    assert matches[0].is_upcoming is False


def test_live_matches_filters_out_completed():
    html = _match_info_html(state="Complete") + _match_info_html(
        matchId=152743, state="In Progress"
    )
    live = _discovery(html).live_matches()
    assert [m.match_id for m in live] == ["152743"]


def test_duplicate_match_ids_across_chunks_are_deduplicated():
    html = _match_info_html() + _match_info_html()
    matches = _discovery(html).matches()
    assert len(matches) == 1


def test_is_t20_international_true_when_match_type_confirms_it():
    matches = _discovery(
        _match_info_html(matchFormat="T20", matchDesc="2nd T20I", matchType="International")
    ).matches()
    assert is_t20_international(matches[0]) is True


def test_is_t20_international_false_for_a_league_match():
    matches = _discovery(
        _match_info_html(
            matchFormat="T20",
            matchDesc="Qualifier 1",
            matchType="league",
            seriesName="Caribbean Premier League 2026",
        )
    ).matches()
    assert is_t20_international(matches[0]) is False


def test_is_t20_international_falls_back_to_matchdesc_when_matchtype_missing():
    # Confirmed live 2026-09-18: Cricbuzz's index page can leave matchType
    # null even for a genuine T20I (e.g. an associate-nations fixture) --
    # matchDesc still says "T20I" in that case.
    matches = _discovery(
        _match_info_html(matchFormat="T20", matchDesc="1st T20I", matchType=None)
    ).matches()
    assert is_t20_international(matches[0]) is True


def test_is_t20_international_false_for_odi():
    matches = _discovery(
        _match_info_html(matchFormat="ODI", matchDesc="2nd ODI")
    ).matches()
    assert is_t20_international(matches[0]) is False
