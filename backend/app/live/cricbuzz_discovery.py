"""Discover live/upcoming matches from Cricbuzz's own match-center index.

Built 2026-09-18 to close the "Cricbuzz match discovery still hardcoded
to one URL" punch-list gap (run_cricbuzz_match.py has always required the
match URL as a command-line argument -- see its own module docstring).

Unlike TOI's ToiMatchDiscovery (scrape anchor links off the index page,
then fetch each candidate's own per-match JSON just to classify it),
Cricbuzz's live-scores index page already embeds the exact same
self.__next_f.push([id, "<json>"]) React Server Component flight-data
chunks app/live/cricbuzz_reader.py parses for ball-by-ball commentary --
this reuses that same extraction to read structured match-summary objects
instead (matchId/matchFormat/state/matchType/team1/team2/venueInfo), all
in one request. No per-candidate fetch is needed just to decide whether a
match is a live T20I worth attaching to.

Confirmed live 2026-09-18 against the real index page
(https://www.cricbuzz.com/cricket-match/live-scores): `matchType` is a
direct "International"/"Domestic"/"League"/"Women" classifier Cricbuzz
computes itself. Contrast run_toi_auto.py's own _is_t20_international(),
which has to infer this from series-name keyword substrings because TOI's
feed has no equivalent field -- Cricbuzz's discovery filter can be exact
where TOI's has to guess.

`state` values confirmed live: "Preview" (upcoming), "Toss" (toss done,
play not yet started), "In Progress" (live), "Stumps" (a Test day's play
paused overnight -- the match itself is still live, but Test cricket is
already out of scope everywhere else in this project, see
app/ml/engine_router.py's EngineRouter.resolve), "Complete".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests

from app.live.cricbuzz_reader import _extract_balanced, _extract_flight_chunks


@dataclass(frozen=True, slots=True)
class CricbuzzMatchSummary:
    match_id: str
    url: str
    label: str
    series: str
    match_desc: str
    match_format: str
    match_type: str
    state: str
    status: str
    venue: str
    is_live: bool
    is_upcoming: bool


class CricbuzzMatchDiscovery:
    LIST_URL = "https://www.cricbuzz.com/cricket-match/live-scores"
    _MATCH_INFO_RE = re.compile(r'\{"matchInfo":\{"matchId":\d+')
    _LIVE_STATES = frozenset({"in progress", "toss"})
    _UPCOMING_STATES = frozenset({"preview"})

    def __init__(
        self, timeout_seconds: float = 12.0, session: requests.Session | None = None
    ) -> None:
        self._timeout = timeout_seconds
        self._session = session or requests.Session()
        self._session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; CricketBaba/3.0)"}
        )

    def matches(self) -> tuple[CricbuzzMatchSummary, ...]:
        """Every match Cricbuzz's index currently lists (any state)."""
        response = self._session.get(self.LIST_URL, timeout=self._timeout)
        response.raise_for_status()
        chunks = _extract_flight_chunks(response.text)
        seen: dict[int, CricbuzzMatchSummary] = {}
        for chunk in chunks:
            for found in self._MATCH_INFO_RE.finditer(chunk):
                raw = _extract_balanced(chunk, found.start())
                if raw is None:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                info = obj.get("matchInfo")
                if not isinstance(info, dict):
                    continue
                match_id = info.get("matchId")
                if not isinstance(match_id, int) or match_id in seen:
                    continue
                seen[match_id] = self._summary(match_id, info)
        return tuple(seen.values())

    def live_matches(self) -> tuple[CricbuzzMatchSummary, ...]:
        return tuple(match for match in self.matches() if match.is_live)

    @classmethod
    def _summary(cls, match_id: int, info: dict[str, Any]) -> CricbuzzMatchSummary:
        team1 = info.get("team1") if isinstance(info.get("team1"), dict) else {}
        team2 = info.get("team2") if isinstance(info.get("team2"), dict) else {}
        team1_name = str(team1.get("teamName") or "")
        team2_name = str(team2.get("teamName") or "")
        venue = info.get("venueInfo") if isinstance(info.get("venueInfo"), dict) else {}
        state = str(info.get("state") or "").strip()
        return CricbuzzMatchSummary(
            match_id=str(match_id),
            url=f"https://www.cricbuzz.com/live-cricket-scores/{match_id}/x",
            label=(
                " vs ".join(name for name in (team1_name, team2_name) if name)
                or str(match_id)
            ),
            series=str(info.get("seriesName") or ""),
            match_desc=str(info.get("matchDesc") or ""),
            match_format=str(info.get("matchFormat") or ""),
            match_type=str(info.get("matchType") or "").strip(),
            state=state,
            status=str(info.get("status") or ""),
            venue=", ".join(
                part
                for part in (str(venue.get("ground") or ""), str(venue.get("city") or ""))
                if part
            ),
            is_live=state.casefold() in cls._LIVE_STATES,
            is_upcoming=state.casefold() in cls._UPCOMING_STATES,
        )


def is_t20_international(match: CricbuzzMatchSummary) -> bool:
    """T20/T20I fixtures that are genuine international play, not a
    franchise/domestic league.

    Cricbuzz's own `matchType` field is the strongest signal when present
    ("International" vs "League"/"Domestic") -- confirmed live
    2026-09-18: the Australia-vs-Zimbabwe ODI showed "International" the
    same day CPL/WCPL/ETPL matches all showed "league". But it is NOT
    reliably populated on the index page for every match -- confirmed the
    same day: a live Africa Continental Cup fixture (an associate-nations
    T20 series) showed matchType null in all 4 of its occurrences on the
    real index page, despite the per-match page populating it cleanly for
    higher-profile series (AUS vs ZIM). Cricbuzz's own `matchDesc` almost
    always spells out "T20I" in the description text for genuine
    internationals ("2nd T20I", "1st T20I", ...) but never for league
    matches ("Qualifier 1", "29th Match", ...) -- checking both signals
    together catches more real T20Is than either alone, without
    (re-)introducing run_toi_auto.py's series-name-keyword guessing.

    Known limitation, same as run_toi_auto.py's own
    _is_t20_international(): a T20I-status associate tournament whose
    matchDesc doesn't literally say "T20I" (e.g. a numbered "Nth Match"
    round) and whose matchType is also missing will be excluded here --
    an unsolved general problem (ICC T20I status isn't always evident
    from branding alone), not a regression introduced by this filter.
    """
    if match.match_format.strip().upper() not in {"T20", "T20I", "IT20"}:
        return False
    match_type = match.match_type.strip().casefold()
    if match_type in {"league", "domestic"}:
        return False
    return match_type == "international" or "t20i" in match.match_desc.casefold()
