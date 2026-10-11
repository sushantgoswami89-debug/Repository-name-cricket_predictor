"""TOI T20I filtering tests."""

from __future__ import annotations

from app.live.toi_discovery import ToiMatchSummary, is_t20_international


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
    assert is_t20_international(_summary("T20I", "International series"))
    assert is_t20_international(
        _summary("T20", "Bilateral series", "International")
    )
    assert not is_t20_international(
        _summary("T20", "Indian Premier League", "Domestic")
    )
