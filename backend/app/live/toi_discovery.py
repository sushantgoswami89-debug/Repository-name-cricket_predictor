"""Discover currently live matches from TOI's cricket match center."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests

from app.live.toi_reader import ToiLiveReader


@dataclass(frozen=True, slots=True)
class ToiLiveMatch:
    match_id: str
    url: str
    label: str


@dataclass(frozen=True, slots=True)
class ToiMatchSummary:
    match_id: str
    url: str
    label: str
    match_format: str
    series: str
    venue: str
    start_time: str
    status: str
    is_live: bool
    is_upcoming: bool
    weather: str
    temperature: str
    humidity: str
    wind: str
    rain: str
    pitch: str
    likely_track: str
    dew_outlook: str
    team_type: str = ""


class ToiMatchDiscovery:
    LIST_URL = "https://timesofindia.indiatimes.com/sports/cricket/live-cricket-score"
    SCHEDULE_URL = (
        "https://timesofindia.indiatimes.com/sports/cricket/match-center/schedule"
    )
    _LINK_RE = re.compile(
        r'(?:href=|"(?:scoreCardUrl|defaultUrl)":)'
        r'["\'](?P<url>(?:https://timesofindia\.indiatimes\.com)?'
        r'/sports/cricket/match-center-scorecard/[^"\']+)["\']',
        re.IGNORECASE,
    )

    def __init__(
        self, timeout_seconds: float = 12.0, session: requests.Session | None = None
    ) -> None:
        self._timeout = timeout_seconds
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0 CricketBaba/3.0"})

    def live_matches(self) -> tuple[ToiLiveMatch, ...]:
        return tuple(
            ToiLiveMatch(match.match_id, match.url, match.label)
            for match in self.matches()
            if match.is_live
        )

    def live_summaries(self) -> tuple[ToiMatchSummary, ...]:
        """Return complete metadata for matches TOI currently marks live."""
        return tuple(match for match in self.matches() if match.is_live)

    def matches(self) -> tuple[ToiMatchSummary, ...]:
        """Return TOI match-center fixtures with pre-match metadata."""
        urls = self._match_urls(self.LIST_URL)
        urls.update(self._schedule_urls())
        matches: list[ToiMatchSummary] = []
        for url in sorted(urls):
            match_id = ToiLiveReader.match_id_from_url(url)
            raw = self._json(ToiLiveReader.RAW_URL.format(match_id=match_id))
            detail = raw.get("Matchdetail", {})
            match = detail.get("Match", {})
            teams = raw.get("Teams", [])
            team_rows = teams.values() if isinstance(teams, dict) else teams
            names = [
                str(team.get("Name_Full") or team.get("Name") or "")
                for team in team_rows
                if isinstance(team, dict)
            ]
            label = " vs ".join(name for name in names if name) or match_id
            series = detail.get("Series", {})
            venue = detail.get("Venue", {})
            venue_weather = venue.get("Venue_Weather", {}) or {}
            pitch_detail = venue.get("Pitch_Detail", {}) or {}
            humidity = str(venue_weather.get("Humidity", "Unavailable"))
            surface = str(pitch_detail.get("Pitch_Surface") or "Unknown")
            suited_for = str(pitch_detail.get("Pitch_Suited_For") or "Unknown")
            matches.append(
                ToiMatchSummary(
                    match_id=match_id,
                    url=url,
                    label=label,
                    match_format=str(match.get("Type", "Unknown")),
                    series=str(series.get("Name", "Unknown series")),
                    venue=str(venue.get("Name", "Unknown venue")),
                    start_time=str(match.get("StartDate", "Unknown")),
                    status=str(detail.get("Status", "Scheduled")),
                    is_live=self._is_live(match),
                    is_upcoming=self._is_future(match),
                    weather=str(
                        venue_weather.get("Description")
                        or venue_weather.get("Weather")
                        or detail.get("Weather")
                        or "Unavailable"
                    ),
                    temperature=str(venue_weather.get("Temperature", "Unavailable")),
                    humidity=humidity,
                    wind=str(venue_weather.get("Wind_Speed", "Unavailable")),
                    rain=str(venue_weather.get("Rain") or "None reported"),
                    pitch=f"{suited_for}; {surface} surface",
                    likely_track=self._track_outlook(suited_for, surface),
                    dew_outlook=self._dew_outlook(
                        humidity, str(match.get("Daynight", ""))
                    ),
                    team_type=str(
                        match.get("TeamType")
                        or match.get("teamType")
                        or detail.get("TeamType")
                        or ""
                    ),
                )
            )
        return tuple(matches)

    def _match_urls(self, url: str) -> set[str]:
        response = self._session.get(url, timeout=self._timeout)
        response.raise_for_status()
        return {
            self._absolute(html.unescape(match.group("url")).replace("\\u002F", "/"))
            for match in self._LINK_RE.finditer(response.text)
        }

    def _schedule_urls(self) -> set[str]:
        response = self._session.get(self.SCHEDULE_URL, timeout=self._timeout)
        response.raise_for_status()
        marker = "window.App="
        start = response.text.find(marker)
        if start < 0:
            return set()
        try:
            app, _ = json.JSONDecoder().raw_decode(response.text[start + len(marker) :])
            sections = app["state"]["schedule"]["data"]["sectionitems"]
            schedule = next(
                section
                for section in sections
                if section.get("tn") == "globalmatchSchedule"
            )
            return {
                str(item["cardurl"])
                for item in schedule.get("items", [])
                if item.get("cardurl")
            }
        except (KeyError, TypeError, StopIteration, json.JSONDecodeError):
            return set()

    def _json(self, url: str) -> dict[str, Any]:
        response = self._session.get(url, timeout=self._timeout)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _absolute(url: str) -> str:
        if url.startswith("http"):
            return url
        return f"https://timesofindia.indiatimes.com{url}"

    @staticmethod
    def _is_live(match: dict[str, Any]) -> bool:
        return bool(match.get("Live")) and not bool(match.get("Upcoming"))

    @staticmethod
    def _is_future(match: dict[str, Any]) -> bool:
        if not match.get("Upcoming"):
            return False
        try:
            value = str(match["StartDate"])
            try:
                start = datetime.fromisoformat(value)
            except ValueError:
                start = datetime.strptime(value, "%m/%d/%YT%H:%M%z")
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            return start >= datetime.now(timezone.utc)
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _track_outlook(suited_for: str, surface: str) -> str:
        lower = f"{suited_for} {surface}".lower()
        if "green" in lower or "grass" in lower:
            return "Likely early seam movement and carry; batting may ease later."
        if "dry" in lower or "cracked" in lower:
            return "Likely good grip for slower balls/spin as the surface wears."
        if "hard" in lower:
            return "Likely true bounce and carry, generally helpful for strokeplay."
        if "damp" in lower or "wet" in lower:
            return "Likely seam-friendly early with variable pace off the surface."
        if "batting" in lower:
            return "Likely a comparatively true batting track."
        if "bowling" in lower:
            return "Likely offers more assistance to bowlers than a neutral track."
        return "No reliable track inference is available from TOI yet."

    @staticmethod
    def _dew_outlook(humidity: str, daynight: str) -> str:
        try:
            value = float(humidity.replace("%", "").strip())
        except ValueError:
            return "Unknown — TOI humidity is unavailable."
        if daynight.lower() != "yes":
            return "Low during the scheduled daytime conditions."
        if value >= 75:
            return "High potential; gripping the ball may become difficult later."
        if value >= 60:
            return "Moderate potential in the second innings."
        return "Low potential based on current TOI humidity."
