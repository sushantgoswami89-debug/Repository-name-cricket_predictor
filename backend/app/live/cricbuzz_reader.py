"""Read and normalize Cricbuzz live ball-by-ball cricket data.

Built 2026-09-15 as a fallback discovery/data path after TOI did not
publish the Afghanistan vs India 2nd T20I at all, despite the match being
real and live (confirmed independently via ESPNcricinfo). See
docs/live_field_test_2026-09-15.md for the investigation.

Cricbuzz's match-center page is a Next.js app. The rendered HTML does not
carry a plain top-level JSON blob the way TOI's ``window.App`` does --
instead, per-ball commentary objects are embedded inside React Server
Component "flight" data chunks: ``<script>self.__next_f.push([id,
"<json-escaped string>"])</script>``. Each pushed string, once JSON-decoded
as a string literal, contains plain embedded JSON objects of the shape
``{"matchId":...,"commType":"commentary","commText":...,"event":[...],
"ballMetric":over.ball,"batsmanDetails":{...},"bowlerDetails":{...},
"overSeparator":{...over summary + cumulative score, only on the last
ball of each over...} }``. This is scraped by locating those objects
directly (self-contained, balanced-brace) rather than parsing the full
React element tree, which also contains non-cricket noise.

IMPORTANT, UNVALIDATED against real data yet (spot-checked only against
the first ~3 overs of one real match's first innings):
- Only "four", "none", "wicket", "over-break", "maiden_over" event tags
  have been observed. "six", "wide", "noball", "bye", "legbye" equivalents
  are assumed by naming-convention analogy, not confirmed -- commText
  phrase parsing is the primary signal for run count and extras, with the
  event tag used only as a secondary cross-check where recognized.
- No second-innings (chase) example has been observed, so target
  extraction is unvalidated.
- team_players / toss / venue (present on LiveSnapshot, used by
  MatchContext) were ported 2026-09-18, confirmed live against a real
  match (Australia vs Zimbabwe, 2nd ODI) -- see _toss_from_chunks/
  _venue_from_chunks/_squads_from_chunks. pitch/weather enrichment has
  no equivalent found on Cricbuzz's pages at all (checked directly, not
  assumed missing) -- those two fields stay at LiveSnapshot's defaults on
  this path, same as before.

Verification design differs from TOI on purpose, not by oversight: TOI's
feed reports an independent cumulative score/wickets on every single
ball, which LiveDeliveryVerifier cross-checks each ball's own run
increment against. Cricbuzz's per-ball objects carry no such cumulative
figure -- only `overSeparator` (present once every 6 balls, on the over's
last delivery) gives a real independently-reported checkpoint
(`batTeamObj.teamScore`). This reader self-accumulates a running score
across balls and cross-checks it against that checkpoint whenever one
arrives, raising CricbuzzFeedError on a mismatch -- catching drift once
per over rather than once per ball. That is a real, deliberate reduction
in verification granularity versus TOI, not a bug to silently paper over.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from app.live.live_snapshot import LiveDelivery, LivePlayer, LiveSnapshot


class CricbuzzFeedError(RuntimeError):
    """Raised when a Cricbuzz response cannot be trusted or normalized."""


_DISMISSAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("run out", re.compile(r"\brun\s*out\b", re.IGNORECASE)),
    ("stumped", re.compile(r"\bst\s+[A-Z][\w.'-]*(?:\s+[\w.'-]+)*\s+b\s+", re.IGNORECASE)),
    ("lbw", re.compile(r"\blbw\s+b\s+", re.IGNORECASE)),
    ("hit wicket", re.compile(r"\bhit\s*wicket\b", re.IGNORECASE)),
    ("caught", re.compile(r"\bc\s+(?:&\s*)?[A-Z][\w.'-]*(?:\s+[\w.'-]+)*\s+b\s+", re.IGNORECASE)),
    ("caught and bowled", re.compile(r"\bc\s*&\s*b\s+", re.IGNORECASE)),
    ("bowled", re.compile(r"(?<!l)\bb\s+[A-Z][\w.'-]+", re.IGNORECASE)),
)

_RUN_WORDS = {"no": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


# Real bug found live 2026-09-15 (docs/live_field_test_2026-09-15.md item
# 12): per-ball objects give a short team code ("AFG") via `teamName`,
# while the page's JSON-LD gives full names (["Afghanistan","India"]).
# Comparing them directly as strings never matches, so `bowling_team`
# resolved to the SAME team as `batting_team` (just spelled differently)
# -- a real, silent, high-impact corruption of the win-probability
# model's team-identity features, not merely a display nuisance. A few
# common international short codes don't follow the general "first N
# letters" pattern Cricbuzz otherwise uses, so those are listed
# explicitly; everything else is derived algorithmically.
_TEAM_CODE_OVERRIDES: dict[str, str] = {
    "USA": "United States",
    "UAE": "United Arab Emirates",
    "HK": "Hong Kong",
    "PNG": "Papua New Guinea",
}


def _resolve_full_team_name(code: str, full_names: list[str]) -> str:
    """Resolve a Cricbuzz short team code to its full name from `teams`
    (both must already be in `teams` -- this only picks which one)."""
    code = code.strip().upper()
    if not code:
        return ""
    for full_name in full_names:
        if full_name.strip().upper() == code:
            return full_name
        words = full_name.split()
        candidate = (
            "".join(word[0] for word in words).upper()
            if len(words) > 1
            else full_name[:3].upper()
        )
        if candidate == code:
            return full_name
    for override_code, override_name in _TEAM_CODE_OVERRIDES.items():
        if override_code == code and override_name in full_names:
            return override_name
    return ""


def _extract_flight_chunks(html: str) -> list[str]:
    """Every self.__next_f.push([id, "<escaped json string>"]) payload,
    decoded back to plain text. Each push call is itself a valid 2-element
    JSON array (the outer HTML-escaping is standard JSON string escaping),
    so json.loads on the array text recovers the real payload cleanly."""
    chunks: list[str] = []
    for match in re.finditer(r"self\.__next_f\.push\((\[.*?\])\)(?=</script>)", html, re.S):
        try:
            _chunk_id, payload = json.loads(match.group(1))
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, str):
            chunks.append(payload)
    return chunks


def _extract_balanced(text: str, start: int) -> str | None:
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _delivery_objects(chunks: list[str], match_id: str) -> list[dict[str, Any]]:
    marker = f'{{"matchId":{match_id},"commType":"commentary"'
    objects: list[dict[str, Any]] = []
    for chunk in chunks:
        for match in re.finditer(re.escape(marker), chunk):
            raw = _extract_balanced(chunk, match.start())
            if raw is None:
                continue
            try:
                objects.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return objects


def _classify_wicket(comm_text: str) -> str | None:
    if "event" not in comm_text and "out" not in comm_text.lower():
        return None
    for label, pattern in _DISMISSAL_PATTERNS:
        if pattern.search(comm_text):
            return label
    return "unknown" if re.search(r"\bOUT\b", comm_text, re.IGNORECASE) else None


def _runs_and_extras(comm_text: str, events: list[str]) -> tuple[int, int, dict[str, int]]:
    """Returns (total_runs, batter_runs, extras) for one delivery, parsed
    from Cricbuzz's natural-language commentary text (the reliable signal
    here -- event tags are used only as a cross-check, since only a
    handful of tag values have been observed against real data)."""
    text = re.sub(r"<[^>]+>", "", comm_text)
    lower = text.lower()
    extras: dict[str, int] = {}

    def _leading_run_count(after: str) -> int:
        # Real bug found 2026-09-18 scanning archived matches
        # (scan_cricbuzz_event_types.py): Cricbuzz's real phrasing always
        # puts a comma before the explicit run count ("leg byes, 2 runs",
        # "wide, 3 runs"), which the original `\s*` prefix here never
        # matched -- silently falling through to the hardcoded default of
        # 1 below on EVERY extra with more than one run, confirmed
        # directly: a real "leg byes, 2 runs" delivery parsed as 1, not 2.
        match = re.match(r"\s*,?\s*(\d+)\s*runs?\b", after)
        if match:
            return int(match.group(1))
        match = re.match(r"\s*,?\s*no\s*runs?\b", after)
        if match:
            return 0
        return 1

    wide_match = re.search(r",\s*wide(?:s)?\b(.*)", lower)
    noball_match = re.search(r",\s*no\s*ball\b(.*)", lower)
    bye_match = re.search(r",\s*(leg\s*bye|bye)s?\b(.*)", lower)

    if "four" in events or re.search(r"<b>four</b>", comm_text, re.IGNORECASE):
        return 4, 4, extras
    if "six" in events or re.search(r"<b>six</b>", comm_text, re.IGNORECASE):
        return 6, 6, extras

    if wide_match:
        runs = 1 + _leading_run_count(wide_match.group(1))
        extras["wides"] = runs
        return runs, 0, extras
    if noball_match:
        rest = noball_match.group(1)
        additional = _leading_run_count(rest) if rest.strip() else 0
        runs = 1 + additional
        extras["noballs"] = 1
        return runs, additional, extras
    if bye_match:
        kind = "legbyes" if "leg" in bye_match.group(1) else "byes"
        runs = _leading_run_count(bye_match.group(2))
        runs = runs if runs else 1
        extras[kind] = runs
        return runs, 0, extras

    match = re.search(r",\s*(no\s*run|\d+\s*runs?)\b", lower)
    if match:
        token = match.group(1)
        if token.startswith("no"):
            return 0, 0, extras
        return int(re.match(r"\d+", token).group()), int(re.match(r"\d+", token).group()), extras

    return 0, 0, extras


@dataclass
class CricbuzzLiveReader:
    """Poll Cricbuzz's match-center page and return a LiveSnapshot-shaped
    result so it can plug into VerifiedLivePredictionPipeline without
    changes there. Fallback-only: see module docstring for what is and
    isn't covered/validated yet."""

    timeout_seconds: float = 12.0
    _session: requests.Session | None = None
    # Keyed by (match_id, innings), not match_id alone -- a real bug found
    # live 2026-09-15 right at the innings break (docs/
    # live_field_test_2026-09-15.md item 14): keying by match_id only
    # meant the 2nd innings would have silently kept accumulating on top
    # of the 1st innings' final score/wickets/cutoff instead of resetting
    # to a fresh 0/0 baseline, the moment the first real ball of the chase
    # arrived. Fixed before that ball landed, not after.
    _running_score: dict[tuple[str, int], int] | None = None
    _running_wickets: dict[tuple[str, int], int] | None = None
    # Accumulated delivery history per (match_id, innings), growing across
    # polls -- the pipeline's LiveDeliveryVerifier expects the FULL known
    # history on every call (it dedupes by .key itself, matching how
    # TOI's own feed returns the complete scorecard every time), not just
    # this poll's new balls.
    _all_deliveries: dict[tuple[str, int], dict[tuple[int, int, int], LiveDelivery]] | None = None
    _seed_cutoff_ms: dict[tuple[str, int], int] | None = None
    _last_delivery: dict[tuple[str, int], LiveDelivery] | None = None

    def __post_init__(self) -> None:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "User-Agent": "Mozilla/5.0 (compatible; CricketBaba/3.0)",
                    "Cache-Control": "no-cache, no-store",
                    "Pragma": "no-cache",
                }
            )
        if self._running_score is None:
            self._running_score = {}
        if self._running_wickets is None:
            self._running_wickets = {}
        if self._all_deliveries is None:
            self._all_deliveries = {}
        if self._seed_cutoff_ms is None:
            self._seed_cutoff_ms = {}
        if self._last_delivery is None:
            self._last_delivery = {}

    @staticmethod
    def match_id_from_url(url_or_id: str) -> str:
        match = re.search(r"/live-cricket-scores/(\d+)/", url_or_id)
        if match:
            return match.group(1)
        if url_or_id.isdigit():
            return url_or_id
        raise CricbuzzFeedError(f"Could not extract a Cricbuzz match id from {url_or_id!r}")

    def fetch(self, url_or_id: str) -> LiveSnapshot:
        match_id = self.match_id_from_url(url_or_id)
        url = (
            url_or_id
            if url_or_id.startswith("http")
            else f"https://www.cricbuzz.com/live-cricket-scores/{match_id}/x"
        )
        try:
            # Real bug found live 2026-09-17
            # (docs/live_field_test_2026-09-17.md): reusing one Session's
            # kept-alive connection for every poll of a match can pin
            # requests to a single CDN edge that serves a stale cached
            # snapshot of this page for minutes at a stretch -- seen
            # directly, both as a whole innings' worth of scoreboard
            # frozen mid-chase and as over-checkpoint totals that
            # disagreed with a simultaneous fresh request. A cache-busting
            # query param (on top of the no-cache headers set above)
            # forces each poll to be treated as a distinct URL, which is
            # the one thing that reliably defeats an edge cache.
            cache_bust = "&" if "?" in url else "?"
            response = self._session.get(
                f"{url}{cache_bust}_cb={int(time.time() * 1000)}",
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise CricbuzzFeedError(f"Could not fetch Cricbuzz match {match_id}: {exc}") from exc

        html = response.text
        chunks = _extract_flight_chunks(html)
        raw_objects = _delivery_objects(chunks, match_id)
        if not raw_objects:
            raise CricbuzzFeedError(
                f"No delivery commentary found for Cricbuzz match {match_id} -- "
                "page structure may have changed or match has no ball-by-ball yet."
            )

        # The window can carry the tail of a just-finished innings AND the
        # very first (pre-ball) commentary of the next one at the same
        # time -- seen live 2026-09-15 right at this match's innings
        # break. The highest inningsId present is authoritative for
        # "what's current now"; older-innings objects are only kept
        # around to seed that innings' own already-persisted state, never
        # mixed into the new one's.
        current_innings = max(
            (int(obj.get("inningsId", 1)) for obj in raw_objects), default=1
        )
        scope = (match_id, current_innings)
        current_innings_objects = [
            obj for obj in raw_objects if int(obj.get("inningsId", 1)) == current_innings
        ]

        deliveries = self._build_deliveries(scope, current_innings_objects)
        # Empty is a legitimate outcome here, not a failure: it means this
        # poll landed exactly at an over-checkpoint boundary with nothing
        # newer yet (or, on a first-ever poll, the window's newest item
        # was itself the seed checkpoint) -- the same "nothing new this
        # cycle" case TOI's own feed produces regularly. Fall back to the
        # last known ball for THIS SAME innings scope rather than raising
        # -- never fall back across an innings boundary (that would
        # silently re-report the previous innings' final state as if it
        # were current).
        latest = deliveries[-1] if deliveries else self._last_delivery.get(scope)
        if latest is None:
            raise CricbuzzFeedError(
                f"No ball-by-ball state yet for Cricbuzz match {match_id} innings "
                f"{current_innings} -- innings likely hasn't started."
            )
        if deliveries:
            self._last_delivery[scope] = latest

        teams = self._teams_from_ldjson(html)
        match_format = self._format_from_html(html)
        latest_raw = max(current_innings_objects, key=lambda obj: obj.get("timestamp", 0))
        batting_code = str(latest_raw.get("teamName", ""))
        batting_team = _resolve_full_team_name(batting_code, teams) or (
            teams[0] if teams else batting_code
        )
        bowling_team = next((team for team in teams if team != batting_team), "")

        wickets = self._running_wickets.get(scope, 0)
        scheduled_overs = {"T20": 20, "T20I": 20, "ODI": 50}.get(match_format, 0)
        legal_balls_total = (latest.over - 1) * 6 + latest.ball
        # Real bug found live 2026-09-15, over 21 predicted for a finished
        # 20-over innings (docs/live_field_test_2026-09-15.md item 12):
        # LiveSnapshot.innings_complete/match_complete default to False and
        # were never set here, and pipeline._is_complete()'s own ball-count
        # fallback counts only verifier.accepted deliveries -- which, for
        # Cricbuzz's rolling window, covers just the balls seen since
        # attach, never the full innings. Must detect completion here
        # directly rather than relying on either.
        innings_complete = wickets >= 10 or (
            scheduled_overs > 0 and legal_balls_total >= scheduled_overs * 6
        )
        # An even-later innings appearing in the raw window (not just the
        # current scope) is the clearest unambiguous signal this innings
        # ended, independent of the over/wicket-count heuristics above.
        if any(int(obj.get("inningsId", 1)) > current_innings for obj in raw_objects):
            innings_complete = True

        toss_won_by, toss_decision = self._toss_from_chunks(chunks)

        return LiveSnapshot(
            match_id=match_id,
            match_format=match_format,
            batting_team=batting_team,
            bowling_team=bowling_team,
            innings=current_innings,
            score=self._running_score.get(scope, 0),
            wickets=wickets,
            overs=f"{latest.over - 1}.{latest.ball}",
            is_live=True,
            target=self._target_from_chunks(chunks),
            innings_complete=innings_complete,
            scheduled_overs=scheduled_overs,
            deliveries=tuple(deliveries),
            team_players=self._squads_from_chunks(chunks),
            venue_name=self._venue_from_chunks(chunks, match_id),
            toss_won_by=toss_won_by,
            toss_decision=toss_decision,
        )

    @staticmethod
    def _target_from_chunks(chunks: list[str]) -> int:
        # Real gap found live 2026-09-15 (docs/live_field_test_2026-09-15.md
        # item 15): without this, `target` stayed 0 for the entire chase,
        # so `context.is_chase` (target > 0) was False and every
        # required-run-rate/chase feature would have been silently wrong
        # for all of India's innings. Cricbuzz embeds a direct
        # `"target":<int>` field (0/absent during the 1st innings, which is
        # the correct value there too) once the chase is set -- but, like
        # the delivery objects, it lives inside an escaped RSC flight-chunk
        # string, not plain HTML. A first attempt regexed raw `html`
        # directly and got the RIGHT-LOOKING but WRONG field: an unrelated
        # schema.org `"target":{"@type":"EntryPoint"...}` block matched
        # first because the real cricket field is escaped (`\"target\":160`)
        # in the raw bytes and never matches an unescaped `"target":(\d+)`
        # pattern at all. Searching the already-unescaped chunks (same
        # source `_delivery_objects` uses) avoids both problems.
        for chunk in chunks:
            match = re.search(r'"target":(\d+)', chunk)
            if match:
                return int(match.group(1))
        return 0

    @staticmethod
    def _checkpoint(obj: dict[str, Any]) -> tuple[int, int] | None:
        over_separator = obj.get("overSeparator")
        if not isinstance(over_separator, dict):
            return None
        checkpoint_score = over_separator.get("batTeamObj", {}).get("teamScore", "")
        match = re.match(r"(\d+)-(\d+)", str(checkpoint_score))
        return (int(match.group(1)), int(match.group(2))) if match else None

    def _build_deliveries(
        self, scope: tuple[str, int], raw_objects: list[dict[str, Any]]
    ) -> list[LiveDelivery]:
        match_id, innings = scope
        # Cricbuzz lists newest-first; process oldest-first so running
        # totals accumulate in real chronological order.
        ordered = sorted(raw_objects, key=lambda obj: obj.get("timestamp", 0))
        known = self._all_deliveries.setdefault(scope, {})

        if scope not in self._running_score:
            # First attach for this match: Cricbuzz's page only carries a
            # rolling window of recent individual balls (unlike TOI, whose
            # feed is the complete scorecard from ball 1) -- attaching
            # mid-match means the early overs' own ball entries are
            # already gone from the window even though their aggregate
            # checkpoint total is still present. Seed from the LATEST
            # available checkpoint rather than assuming 0/0, and only
            # trust/emit balls strictly after it -- replaying balls we
            # can see on top of a wrong zero baseline double-undercounts
            # exactly like tonight's real test run showed (9/0 vs the
            # feed's real 24/1 at the same point).
            best_checkpoint_obj: dict[str, Any] | None = None
            best_checkpoint: tuple[int, int, int] | None = None  # (timestamp, runs, wkts)
            for obj in ordered:
                checkpoint = self._checkpoint(obj)
                if checkpoint is not None:
                    timestamp = int(obj.get("timestamp", 0))
                    if best_checkpoint is None or timestamp > best_checkpoint[0]:
                        best_checkpoint = (timestamp, checkpoint[0], checkpoint[1])
                        best_checkpoint_obj = obj
            if best_checkpoint is not None and best_checkpoint_obj is not None:
                seed_ts, seed_runs, seed_wkts = best_checkpoint
                self._running_score[scope] = seed_runs
                self._running_wickets[scope] = seed_wkts
                self._seed_cutoff_ms[scope] = seed_ts
                # Record the checkpoint itself as the last-known state so
                # fetch() can still report a real snapshot (e.g. "innings
                # complete, 159/8") even if this poll finds zero balls
                # strictly after it -- a real, persistent case during an
                # innings break, not just a rare single-poll race.
                seed_ball_metric = best_checkpoint_obj.get("ballMetric")
                if isinstance(seed_ball_metric, (int, float)):
                    seed_over_index = int(seed_ball_metric)
                    seed_ball_number = round((seed_ball_metric - seed_over_index) * 10)
                    self._last_delivery[scope] = LiveDelivery(
                        match_id=match_id,
                        innings=int(best_checkpoint_obj.get("inningsId", innings)),
                        over=seed_over_index + 1,
                        ball=seed_ball_number,
                        bowler=best_checkpoint_obj.get("bowlerDetails", {}).get("playerName", ""),
                        striker=best_checkpoint_obj.get("batsmanDetails", {}).get("playerName", ""),
                        total_runs=0,
                        batter_runs=0,
                        extras={},
                        wicket_kind=None,
                        feed_total=seed_runs,
                        feed_wickets=seed_wkts,
                        timestamp_ms=seed_ts,
                        commentary="(seeded from over-checkpoint, not a live delivery)",
                    )
            else:
                # No over has completed yet in this window -- genuinely
                # early in the innings, 0/0 is the correct baseline.
                self._running_score[scope] = 0
                self._running_wickets[scope] = 0
                self._seed_cutoff_ms[scope] = -1

        running_score = self._running_score[scope]
        running_wickets = self._running_wickets[scope]
        cutoff_ms = self._seed_cutoff_ms[scope]

        for obj in ordered:
            ball_metric = obj.get("ballMetric")
            if not isinstance(ball_metric, (int, float)):
                continue  # non-ball commentary (bowling change, milestone, etc.)
            timestamp = int(obj.get("timestamp", 0))
            if timestamp <= cutoff_ms:
                continue  # already folded into the seed checkpoint above
            over_index = int(ball_metric)
            ball_number = round((ball_metric - over_index) * 10)
            key = (int(obj.get("inningsId", 1)), over_index + 1, ball_number)
            if key in known:
                continue  # already accumulated on an earlier poll

            comm_text = obj.get("commText", "")
            events = [str(e) for e in obj.get("event", [])]

            total_runs, batter_runs, extras = _runs_and_extras(comm_text, events)
            wicket_kind = _classify_wicket(comm_text) if "wicket" in events else None

            running_score += total_runs
            if wicket_kind is not None:
                running_wickets += 1

            checkpoint = self._checkpoint(obj)
            if checkpoint is not None:
                checkpoint_runs, checkpoint_wkts = checkpoint
                if checkpoint_runs != running_score or checkpoint_wkts != running_wickets:
                    raise CricbuzzFeedError(
                        f"Cricbuzz over-checkpoint mismatch for {match_id} innings "
                        f"{innings}: accumulated {running_score}/{running_wickets}, "
                        f"feed reports {checkpoint_runs}/{checkpoint_wkts} "
                        f"after over {obj.get('overSeparator', {}).get('overNumber')}."
                    )

            known[key] = LiveDelivery(
                match_id=match_id,
                innings=key[0],
                over=over_index + 1,
                ball=ball_number,
                bowler=obj.get("bowlerDetails", {}).get("playerName", ""),
                striker=obj.get("batsmanDetails", {}).get("playerName", ""),
                total_runs=total_runs,
                batter_runs=batter_runs,
                extras=extras,
                wicket_kind=wicket_kind,
                feed_total=running_score,
                feed_wickets=running_wickets,
                timestamp_ms=int(obj.get("timestamp", 0)),
                commentary=re.sub(r"<[^>]+>", "", comm_text),
            )

        self._running_score[scope] = running_score
        self._running_wickets[scope] = running_wickets
        return sorted(known.values(), key=lambda delivery: delivery.key)

    @staticmethod
    def _teams_from_ldjson(html: str) -> list[str]:
        for match in re.finditer(
            r'"@type":"SportsEvent".*?"competitor":\[(.*?)\]', html, re.S
        ):
            names = re.findall(r'"name":"([^"]+)"', match.group(1))
            if names:
                return names
        return []

    @staticmethod
    def _format_from_html(html: str) -> str:
        # The page-level "matchFormat" JSON field belongs to whichever
        # match a sidebar/ticker widget happens to list first -- not
        # reliably this match's own format. The <title> tag always names
        # this specific match ("... 2nd T20I ..."), so parse it from there.
        title = re.search(r"<title>(.*?)</title>", html)
        if not title:
            return ""
        match = re.search(r"\b(T20I|ODI|Test|T20)\b", title.group(1))
        return match.group(1) if match else ""

    @staticmethod
    def _toss_from_chunks(chunks: list[str]) -> tuple[str, str]:
        """(toss_winner_full_name, decision), or ("", "") pre-toss.

        Closes a real gap flagged in this module's own docstring
        (team_players/toss/pitch/weather enrichment "NOT ported here
        yet"). Confirmed live 2026-09-18 against a real match (Australia
        vs Zimbabwe, 2nd ODI): Cricbuzz embeds a standalone
        "tossResults":{"tossWinnerId":...,"tossWinnerName":"Zimbabwe",
        "decision":"Bowling"} object with the full team name already
        resolved (no short-code translation needed, unlike batting_team/
        bowling_team elsewhere in this file).
        """
        for chunk in chunks:
            match = re.search(
                r'"tossResults":\{"tossWinnerId":\d+,'
                r'"tossWinnerName":"([^"]*)","decision":"([^"]*)"',
                chunk,
            )
            if match and match.group(1):
                return match.group(1), match.group(2)
        return "", ""

    @staticmethod
    def _venue_from_chunks(chunks: list[str], match_id: str) -> str:
        """"Ground, City", or "" if not found.

        Same {"matchInfo":{"matchId":...,"venueInfo":{"ground":...,
        "city":...}}} object CricbuzzMatchDiscovery reads for the match
        list -- confirmed present on the individual match page too.
        """
        marker = f'{{"matchInfo":{{"matchId":{match_id},'
        for chunk in chunks:
            for found in re.finditer(re.escape(marker), chunk):
                raw = _extract_balanced(chunk, found.start())
                if raw is None:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                venue = obj.get("matchInfo", {}).get("venueInfo")
                if isinstance(venue, dict):
                    ground = str(venue.get("ground") or "")
                    city = str(venue.get("city") or "")
                    name = ", ".join(part for part in (ground, city) if part)
                    if name:
                        return name
        return ""

    @staticmethod
    def _squads_from_chunks(chunks: list[str]) -> dict[str, tuple[LivePlayer, ...]]:
        """Best-effort team_players, matching LiveSnapshot.team_players'
        shape exactly so it plugs into MatchContext the same way TOI's
        does.

        Real gotcha confirmed live 2026-09-18 against a real squad list
        (Australia vs Zimbabwe, 2nd ODI, 20 entries per team): this
        playerDetails list mixes real players with coaching staff (head
        coach, assistant/batting/bowling coaches), and the coaches are
        NOT distinguishable by name or role text alone (a coach's `role`
        can be absent same as an unlisted player's). The one reliable
        signal is `substitute`: real players always carry an actual
        boolean (False = confirmed playing XI, True = nominated bench/
        impact substitute); coaching staff carry a literal null. Skipping
        anything where `substitute` isn't a real bool keeps this from
        silently feeding a coach into the wicket model's team-composition
        features as if they were an eligible bowler/batter.
        """
        squads: dict[str, tuple[LivePlayer, ...]] = {}
        for chunk in chunks:
            for team_key in ("team1", "team2"):
                match = re.search(
                    rf'"{team_key}":(\{{"id":\d+,"name":"([^"]*)",'
                    r'"playerDetails":\[\{)',
                    chunk,
                )
                if not match:
                    continue
                team_name = match.group(2)
                if not team_name or team_name in squads:
                    continue
                raw = _extract_balanced(chunk, match.start(1))
                if raw is None:
                    continue
                try:
                    team = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                real_players = [
                    entry
                    for entry in team.get("playerDetails", [])
                    if isinstance(entry.get("substitute"), bool)
                ]
                players = tuple(
                    LivePlayer(
                        position=index,
                        name=str(entry.get("name") or ""),
                        role=str(entry.get("role") or ""),
                        confirm_xi=not entry["substitute"],
                    )
                    for index, entry in enumerate(real_players, start=1)
                    if entry.get("name")
                )
                if players:
                    squads[team_name] = players
        return squads
