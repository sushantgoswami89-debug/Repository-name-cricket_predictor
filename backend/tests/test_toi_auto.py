"""Automatic TOI shadow-monitor filtering and recording tests."""

from __future__ import annotations

import json

from app.live.toi_discovery import ToiMatchSummary
from run_toi_auto import _is_t20_international, _write_shadow_prediction


def _summary(
    match_format: str, series: str, team_type: str = ""
) -> ToiMatchSummary:
    return ToiMatchSummary(
        match_id="match",
        url="https://example.test/match",
        label="A vs B",
        match_format=match_format,
        series=series,
        venue="Venue",
        start_time="",
        status="Live",
        is_live=True,
        is_upcoming=False,
        weather="",
        temperature="",
        humidity="",
        wind="",
        rain="",
        pitch="",
        likely_track="",
        dew_outlook="",
        team_type=team_type,
    )


def test_shadow_filter_accepts_t20i_and_rejects_ipl() -> None:
    assert _is_t20_international(_summary("T20I", "International series"))
    assert _is_t20_international(
        _summary("T20", "Bilateral series", "International")
    )
    assert not _is_t20_international(
        _summary("T20", "Indian Premier League", "Domestic")
    )


def test_shadow_prediction_is_immutable_and_match_scoped(tmp_path) -> None:
    output = {"innings": 1, "over": 2, "prediction": {"expected_runs": 7}}

    path = _write_shadow_prediction(tmp_path, "match-a", output)
    _write_shadow_prediction(
        tmp_path,
        "match-a",
        {"innings": 1, "over": 2, "prediction": {"expected_runs": 99}},
    )

    assert path == (
        tmp_path / "match-a" / "predictions" / "innings-1-over-2.json"
    )
    assert json.loads(path.read_text(encoding="utf-8")) == output
