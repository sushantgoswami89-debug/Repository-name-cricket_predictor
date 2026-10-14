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
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

import requests

from app.live.live_snapshot import LiveDelivery, LivePlayer, LiveSnapshot
from app.live.over_summary import parse_over_summary, tokens_agree_with_total



def _int_field(obj: dict[str, Any], key: str, default: int) -> int:
    """Read an integer field from a Cricbuzz flight object.

    Next.js serializes a missing value as the literal string "$undefined"
    (seen live 2026-09-19 in inningsId, which crashed the whole driver),
    so anything non-numeric falls back to the default instead of raising.
    """
    value = obj.get(key, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return default

class CricbuzzFeedError(RuntimeError):
    """Raised when a Cricbuzz response cannot be trusted or normalized."""


class CricbuzzInningsNotStarted(CricbuzzFeedError):
    """Before the scheduled start (or before the batting side is known) --
    a normal waiting state, not a feed fault. Callers must not count it
    toward reseeds or stuck-feed alerts (2026-10-06: it did, and printed
    continuous "errors" plus a false "feed erroring for 5 min" alert for a
    match that simply hadn't started yet)."""


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


_NOTE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("DRINKS/TIMEOUT", re.compile(r"\btime for a drink\b|\bdrinks?\s+(?:break|time|interval)\b|\bstrategic\s+time-?out\b|\btime-?out\b", re.I)),
    ("WEATHER/DELAY", re.compile(r"\brain\b|\bdrizzle\b|\bcovers\b|\bbad light\b|\bplay (?:has been )?(?:suspended|stopped|halted)\b|\bdelay\b|\bflood-?lights?\b|\bdew\b", re.I)),
    ("INJURY/SUB", re.compile(r"\binjur|\bphysio\b|\bretired (?:hurt|out)\b|\bconcussion\b|\bleaves the field\b|\bimpact player\b|\bsubstitute\b", re.I)),
    ("REVIEW", re.compile(r"\bDRS\b|\bumpire'?s call\b|\bthird umpire\b|\bhas reviewed\b|\btakes? the review\b", re.I)),
    ("BALL/EQUIPMENT", re.compile(r"\bball (?:has been |is )?changed\b|\bchange of ball\b|\bnew ball\b|\bball inspection\b", re.I)),
    ("BOWLING CHANGE", re.compile(r"\bcomes? into the attack\b|\bis back into the attack\b|\bback into the attack\b", re.I)),
    # "Mayank Yadav, right handed bat, comes to the crease": the clearest statement of who walks in after a wicket
    # (2026-10-11: the scorecard page lagged after dismissals while this note was already there).
    ("NEW BATTER", re.compile(r"\b(?:right|left)[- ]?handed bat,?\s+comes? to the crease\b", re.I)),
)


def _classify_note(text: str) -> str | None:
    """Category for non-ball commentary worth surfacing (drinks, delays,
    injuries, reviews, bowling changes); None for trivia/stats. Added
    2026-10-06 after a drinks break sat in the feed unseen -- the reader
    used to drop every entry without a ball number."""
    for label, pattern in _NOTE_PATTERNS:
        if pattern.search(text):
            return label
    return None


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


_TEXT_ROW_HEADER = re.compile(r"^([0-9a-f]+):T([0-9a-f]+),$")
_TEXT_REF = re.compile(r"^\$([0-9a-f]+)$")


def _text_rows(chunks: list[str]) -> dict[str, str]:
    """React Server Component "text rows": a long string is not embedded where it is used; the entry holds
    a pointer ("$26") and the page sends the text separately as a header chunk "26:T501," (hex length) followed
    by the text itself. Found 2026-10-10: a real ball entry (151565, 10.5) arrived as the bare pointer "$26", so
    its runs/wicket words were invisible to the parser and the over came out wrong."""
    rows: dict[str, str] = {}
    for index, chunk in enumerate(chunks):
        header = _TEXT_ROW_HEADER.match(chunk)
        if not header:
            continue
        wanted = int(header.group(2), 16)  # length in bytes
        text, at = "", index + 1
        while at < len(chunks) and len(text.encode("utf-8")) < wanted:
            text += chunks[at]
            at += 1
        rows[header.group(1)] = text
    return rows


def _delivery_objects(chunks: list[str], match_id: str) -> list[dict[str, Any]]:
    marker = f'{{"matchId":{match_id},"commType":"commentary"'
    objects: list[dict[str, Any]] = []
    rows: dict[str, str] | None = None
    for chunk in chunks:
        for match in re.finditer(re.escape(marker), chunk):
            raw = _extract_balanced(chunk, match.start())
            if raw is None:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            reference = _TEXT_REF.match(str(obj.get("commText", "")).strip())
            if reference:
                rows = _text_rows(chunks) if rows is None else rows
                if reference.group(1) in rows:
                    obj["commText"] = rows[reference.group(1)]
            if not str(obj.get("commText", "")).strip():
                # Cricbuzz posts an EMPTY twin of a ball a few seconds after it (151598 ball 16.1: the FOUR, then a blank entry with the
                # same ballMetric and a new timestamp). It looked like a revision "4 -> 0 run(s)" and left the total 4 low.
                continue
            objects.append(obj)
    _attach_over_separators(chunks, objects)
    return objects


_SEPARATOR_MARKER = '{"commType":"over-separator"'


def _attach_over_separators(chunks: list[str], objects: list[dict[str, Any]]) -> None:
    """Put each over separator back on the last ball of its over.

    Until about 2026-10-10 Cricbuzz sent the separator (overSummary tokens, overRuns, the team score after the over) INSIDE the last
    ball's object; the repair, re-anchoring, audit and confirmation code all read it there. Since then it comes as a separate object
    ({"commType":"over-separator", "inningsId":..., "overSeparator":{"overNumber":N,...}}), so none of that code saw a separator
    and every total depended on the starting baseline alone (found 2026-10-14: 1-run offsets that lasted to the end of an innings).
    Re-attaching it makes the existing code work again."""
    for chunk in chunks:
        for hit in re.finditer(re.escape(_SEPARATOR_MARKER), chunk):
            raw = _extract_balanced(chunk, hit.start())
            if raw is None:
                continue
            try:
                holder = json.loads(raw)
            except json.JSONDecodeError:
                continue
            separator = holder.get("overSeparator")
            if not isinstance(separator, dict) or separator.get("overNumber") is None:
                continue
            innings = holder.get("inningsId")
            over_number = _int_field(separator, "overNumber", 0)
            balls = [
                o for o in objects
                if o.get("inningsId") == innings and isinstance(o.get("ballMetric"), (int, float)) and int(o["ballMetric"]) == over_number - 1
            ]
            if not balls:
                continue
            last = max(balls, key=lambda o: (o["ballMetric"], o.get("timestamp", 0)))
            if not isinstance(last.get("overSeparator"), dict):
                last["overSeparator"] = separator


def _classify_wicket(comm_text: str) -> str | None:
    if "event" not in comm_text and "out" not in comm_text.lower():
        return None
    for label, pattern in _DISMISSAL_PATTERNS:
        if pattern.search(comm_text):
            return label
    return "unknown" if re.search(r"\bOUT\b", comm_text, re.IGNORECASE) else None


_RUN_OUT_NAME = re.compile(r"([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*)*)\s+[Rr]un\s+[Oo]ut")


def dismissed_batter(wicket_kind: str | None, commentary: str, striker: str) -> str:
    """Who went out. For a run-out it can be the non-striker (seen live 2026-10-06: Shimron Hetmyer
    run out while Shai Hope faced the ball); every other dismissal is the striker. "" without a wicket."""
    if not wicket_kind:
        return ""
    if wicket_kind == "run out":
        match = _RUN_OUT_NAME.search(commentary)
        if match:
            return match.group(1)
    return striker


_HEAD_SEGMENT = re.compile(
    r"^(?:no\s*runs?|\d+\s*runs?|four|six|wides?|\d+\s+wides?|no\s*ball|(?:leg\s*)?byes?|out\b.*)$",
    re.IGNORECASE,
)


def _opening_phrase(text: str) -> str:
    """The scoring words right after "Bowler to Batter, " (", "-separated),
    stopping at the first prose segment. Extras and runs must be read from
    this only: the prose that follows can contain ", wide of short fine" (a
    fielding position) or ", byes" etc. Seen live 2026-10-06: 17.2 "1 run,
    good yorker length ..., jams it down, wide of short fine for one" was
    parsed as a WIDE, so over 18 showed 5 legal balls and never completed
    -- no over-19 prediction. Returns a lower-case string with a leading
    ", " so the existing patterns apply unchanged."""
    match = re.match(r"^.*? to .*?, (.*)$", text, re.DOTALL)
    rest = match.group(1) if match else text
    kept: list[str] = []
    for segment in rest.split(", "):
        if _HEAD_SEGMENT.match(segment.strip()):
            kept.append(segment.strip())
        else:
            break
    return ", " + ", ".join(kept).lower() if kept else ", " + rest.lower()


def _runs_and_extras(comm_text: str, events: list[str]) -> tuple[int, int, dict[str, int]]:
    """Returns (total_runs, batter_runs, extras) for one delivery, parsed
    from Cricbuzz's natural-language commentary text (the reliable signal
    here -- event tags are used only as a cross-check, since only a
    handful of tag values have been observed against real data)."""
    text = re.sub(r"<[^>]+>", "", comm_text)
    lower = _opening_phrase(text)
    extras: dict[str, int] = {}

    def _leading_run_count(after: str, default: int = 1) -> int:
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
        return default

    wide_match = re.search(r",\s*wide(?:s)?\b(.*)", lower)
    noball_match = re.search(r",\s*no\s*ball\b(.*)", lower)
    bye_match = re.search(r",\s*(leg\s*bye|bye)s?\b(.*)", lower)

    # "5 wides" / "4 wides" (count BEFORE the word, no comma) is the total
    # for the delivery. Seen live 2026-09-19: "5 wides" parsed as 0 runs
    # (the ",\s*wide" pattern never matched it), and "4 wides" was only
    # right by accident via a stray boundary tag.
    counted_wides = re.search(r",\s*(\d+)\s+wides?\b", lower)
    if counted_wides:
        runs = int(counted_wides.group(1))
        extras["wides"] = runs
        return runs, 0, extras

    boundary = 4 if ("four" in events or re.search(r",\s*four\b", lower)) else (
        6 if ("six" in events or re.search(r",\s*six\b", lower)) else 0
    )
    if boundary:
        if bye_match:
            # "leg byes, FOUR" is four EXTRAS, not four off the bat (seen live
            # 2026-10-06 at 8.6: it was credited to Shai Hope as batter runs).
            kind = "legbyes" if "leg" in bye_match.group(1) else "byes"
            extras[kind] = boundary
            return boundary, 0, extras
        if noball_match:
            # "no ball, FOUR": the bat's boundary PLUS the no-ball's own run.
            extras["noballs"] = 1
            return boundary + 1, boundary, extras
        return boundary, boundary, extras

    if wide_match:
        # No explicit count means just the wide itself (seen live 2026-09-19:
        # descriptive prose after "wide," was scoring an extra run).
        runs = 1 + _leading_run_count(wide_match.group(1), default=0)
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

    # A run out after runs were completed: "...out Tilak Varma Run Out!! 1 run completed. Such poor cricket ..." has no
    # ", 1 run" head, so the runs fell through as 0 and our score ran one behind the page for the rest of the innings
    # (India v West Indies, 2026-10-11, 16.4: every later prediction was held).
    if re.search(r"\brun\s*out\b", text, re.IGNORECASE):
        completed = re.search(r"\b(\d+)\s+runs?\s+completed\b", text, re.IGNORECASE)
        if completed:
            runs = int(completed.group(1))
            return runs, runs, extras

    match = re.search(r",\s*(no\s*run|\d+\s*runs?)\b", lower)
    if match:
        token = match.group(1)
        if token.startswith("no"):
            return 0, 0, extras
        return int(re.match(r"\d+", token).group()), int(re.match(r"\d+", token).group()), extras

    return 0, 0, extras



def _align(wanted: list, have: list) -> dict[int, int]:
    """Longest-common-subsequence match of wanted[i] to have[j] (equal values,
    order kept). Returns {i: j}; unmatched wanted items are absent."""
    n, m = len(wanted), len(have)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            table[i][j] = (
                table[i + 1][j + 1] + 1
                if wanted[i] == have[j]
                else max(table[i + 1][j], table[i][j + 1])
            )
    pairs, i, j = {}, 0, 0
    while i < n and j < m:
        if wanted[i] == have[j]:
            pairs[i] = j
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


_TO_NAMES = re.compile(r"^\s*([^,<>]+?)\s+to\s+([^,<>]+?)\s*,")


def names_from_text(comm_text: str) -> tuple[str, str]:
    """(bowler, striker) from "Arafat Minhas to Sineth Jayawardena, out Caught by ...". A wicket ball is first posted with EMPTY
    batsmanDetails / bowlerDetails (playerName "", seen 2026-10-13, filled about 16 s later) although its text names both."""
    match = _TO_NAMES.match(re.sub(r"<[^>]+>", "", comm_text or ""))
    return (match.group(1).strip(), match.group(2).strip()) if match else ("", "")


def board_match_complete(board: dict | None) -> bool:
    """True only when the page's match header says the MATCH is over. At the innings break the header also has
    complete=True ("Innings Break"), which on 2026-10-11 made both drivers treat the match as finished at the break: the
    result was logged as "Innings Break" (so the pre-match log could not be scored) and the session was dropped, so the
    chase was re-attached mid-over and its first-over prediction was lost."""
    if not board:
        return False
    if any("break" in str(board.get(key) or "").lower() for key in ("state", "status")):
        return False
    return board.get("complete") is True or str(board.get("state") or "").lower() == "complete"


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
    _batting_team_by_scope: dict = field(default_factory=dict)
    _seed_cutoff_ms: dict[tuple[str, int], int] | None = None
    _last_delivery: dict[tuple[str, int], LiveDelivery] | None = None
    _seen_timestamps: dict[tuple[str, int], set[int]] | None = None
    _timestamp_keys: dict[tuple[str, int], dict[int, tuple[int, int, int]]] | None = None
    _seen_note_timestamps: set[int] | None = None
    _seen_over_summaries: set[tuple[int, int]] | None = None
    _pending_summaries: list[dict] | None = None
    _pending_notes: list[tuple[int, str, str]] | None = None
    # matches whose finished first innings has been handed out once; later polls at the innings break then get the
    # next innings' empty pre-first-ball snapshot (so the over-1 prediction of a chase can fire)
    _first_innings_final_sent: set[str] | None = None
    # Cricbuzz's own per-over result (overSummary tokens + checkpoint score),
    # keyed (match_id, innings, over); the ground truth an over is rebuilt from.
    _over_truth: dict[tuple[str, int, int], dict] | None = None
    # Arrival-ordered delivery keys per (scope, ballMetric slot): lets a legal
    # ball that is posted again in the same slot (a review revision) replace
    # the earlier entry instead of counting as an extra delivery.
    _slots: dict[tuple[str, int], dict[float, list[tuple[int, int, int]]]] | None = None
    _pending_repairs: list[str] | None = None

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
        if self._seen_timestamps is None:
            self._seen_timestamps = {}
        if self._timestamp_keys is None:
            self._timestamp_keys = {}
        if self._seen_note_timestamps is None:
            self._seen_note_timestamps = set()
        if self._seen_over_summaries is None:
            self._seen_over_summaries = set()
        if self._pending_summaries is None:
            self._pending_summaries = []
        if self._pending_notes is None:
            self._pending_notes = []
        if self._over_truth is None:
            self._over_truth = {}
        if self._slots is None:
            self._slots = {}
        if self._pending_repairs is None:
            self._pending_repairs = []

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
            if self._last_delivery.get((match_id, 1)) is None:
                return self._pre_first_ball_snapshot(match_id, html, chunks)
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
            (_int_field(obj, "inningsId", 1) for obj in raw_objects), default=1
        )
        scope = (match_id, current_innings)
        current_innings_objects = [
            obj for obj in raw_objects if _int_field(obj, "inningsId", 1) == current_innings
        ]

        board = self._board_from_chunks(chunks)
        deliveries = self._build_deliveries(
            scope, current_innings_objects, board=board, require_baseline=True
        )
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
            if current_innings == 1:
                return self._pre_first_ball_snapshot(match_id, html, chunks)
            # A later innings that has not bowled yet (the innings break) is a
            # normal wait, not a feed fault -- same false alarm as before the toss.
            raise CricbuzzInningsNotStarted(
                f"No ball-by-ball state yet for Cricbuzz match {match_id} innings "
                f"{current_innings} -- innings likely hasn't started."
            )
        if deliveries:
            self._last_delivery[scope] = latest

        teams = self._teams_from_ldjson(html)
        match_format = self._format_from_html(html)
        # Non-ball commentary (team news, trivia, "X and Y are at the crease")
        # often has an empty teamName; taking the newest object blindly gave
        # "" there, which fell back to teams[0] -- India -- for a West Indies
        # innings (seen live 2026-10-06, one poll's scorecard flipped to
        # "India 0/0"). Use the newest object that actually names a team.
        named = [o for o in current_innings_objects if str(o.get("teamName", "")).strip()]
        latest_raw = max(named or current_innings_objects, key=lambda obj: obj.get("timestamp", 0))
        batting_code = str(latest_raw.get("teamName", ""))
        resolved_team = _resolve_full_team_name(batting_code, teams)
        # The side batting cannot change inside an innings. 2026-10-14 (West Indies v India): a copy of the page whose window
        # named no team fell back to teams[0] and flipped the batting side to India for minutes, so every poll was refused
        # ("Batting team changed inside innings 1"). Remember the side the first time a ball object names it and keep it.
        remembered = self._batting_team_by_scope.setdefault(scope, "")
        if resolved_team:
            self._batting_team_by_scope[scope] = resolved_team
        batting_team = resolved_team or remembered or (teams[0] if teams else batting_code)
        bowling_team = next((team for team in teams if team != batting_team), "")

        wickets = self._running_wickets.get(scope, 0)
        scheduled_overs = {"T20": 20, "T20I": 20, "ODI": 50}.get(match_format, 0)
        # latest.ball counts every delivery in the over (wides/no-balls shift
        # it past 6); the legal-ball position is that minus the illegal
        # deliveries already seen in the over. Using latest.ball raw would
        # call an innings complete one ball early after a wide in over 20.
        illegal_in_over = sum(
            1 for d in deliveries if d.over == latest.over and not d.is_legal
        )
        legal_in_over = min(6, max(0, latest.ball - illegal_in_over))
        legal_balls_total = (latest.over - 1) * 6 + legal_in_over
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
        if any(_int_field(obj, "inningsId", 1) > current_innings for obj in raw_objects):
            innings_complete = True

        toss_won_by, toss_decision = self._toss_from_chunks(chunks)
        # The page's own match header says when the match is over ("Complete" + a result), which is more
        # reliable than inferring it from the score and the target.
        match_complete = board_match_complete(board)
        if match_complete:
            innings_complete = True

        if (
            current_innings == 1 and innings_complete and not match_complete and scheduled_overs
            and self._at_innings_break(board)
        ):
            if self._first_innings_final_sent is None:
                self._first_innings_final_sent = set()
            if match_id in self._first_innings_final_sent and len(teams) == 2:
                # The finished innings went out once already; until the chase bowls its first ball, hand out the chase's
                # empty snapshot so the pipeline can make the over-1 prediction (user 2026-10-11: it was missed every time).
                return self._pre_next_innings_snapshot(
                    match_id, html, chunks, board, teams, match_format, batting_team, scheduled_overs,
                    first_innings_score=self._running_score.get(scope, 0),
                )
            self._first_innings_final_sent.add(match_id)

        return LiveSnapshot(
            match_id=match_id,
            match_format=match_format,
            batting_team=batting_team,
            bowling_team=bowling_team,
            innings=current_innings,
            score=self._running_score.get(scope, 0),
            wickets=wickets,
            # Standard cricket notation: a finished over reads 7.0, not 6.6
            # (Cricbuzz's own metric), so it agrees with "Over 8 Prediction".
            overs=(
                f"{latest.over}.0" if legal_in_over >= 6 else f"{latest.over - 1}.{legal_in_over}"
            ),
            is_live=True,
            target=self._target_from_chunks(chunks),
            innings_complete=innings_complete,
            match_complete=match_complete,
            scheduled_overs=scheduled_overs,
            deliveries=tuple(deliveries),
            team_players=self._squads_from_chunks(chunks),
            venue_name=self._venue_from_chunks(chunks, match_id),
            toss_won_by=toss_won_by,
            toss_decision=toss_decision,
            board=board,
            raw_html=html,
        )

    @staticmethod
    def _start_ms_from_html(html: str) -> int | None:
        # schema.org SportsEvent carries a series-level and a match-level
        # startDate; the match's is never earlier than the series', so the
        # latest one is the match's own scheduled start.
        stamps = re.findall(r'"startDate":"(\d{4}-\d\d-\d\dT[\d:.]+Z)"', html)
        parsed = []
        for stamp in stamps:
            try:
                parsed.append(int(datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() * 1000))
            except ValueError:
                continue
        return max(parsed) if parsed else None

    @staticmethod
    def _at_innings_break(board: dict | None) -> bool:
        """The page's own match header says "Innings Break" (so the first innings really is over, not just our count)."""
        return bool(board) and any("break" in str(board.get(key) or "").lower() for key in ("state", "status"))

    def _pre_next_innings_snapshot(
        self, match_id: str, html: str, chunks: list[str], board: dict | None, teams: list[str], match_format: str,
        first_batting_team: str, scheduled_overs: int, first_innings_score: int,
    ) -> LiveSnapshot:
        """Empty second-innings snapshot at the innings break: the chasing side bats, the target is the first innings total + 1
        (the page's own target when it already shows one)."""
        chasing = next((team for team in teams if team != first_batting_team), "")
        toss_won_by, toss_decision = self._toss_from_chunks(chunks)
        target = self._target_from_chunks(chunks) or (first_innings_score + 1 if first_innings_score else 0)
        return LiveSnapshot(
            match_id=match_id,
            match_format=match_format,
            batting_team=chasing,
            bowling_team=first_batting_team,
            innings=2,
            score=0,
            wickets=0,
            overs="0.0",
            is_live=True,
            target=target,
            scheduled_overs=scheduled_overs,
            deliveries=(),
            team_players=self._squads_from_chunks(chunks),
            venue_name=self._venue_from_chunks(chunks, match_id),
            toss_won_by=toss_won_by,
            toss_decision=toss_decision,
            board=board,
            raw_html=html,
        )

    def _pre_first_ball_snapshot(self, match_id: str, html: str, chunks: list[str]) -> LiveSnapshot:
        """Empty first-innings snapshot, handed out once the scheduled start
        has passed, so the pipeline's pre-innings ("over 1") prediction can
        fire exactly as it does on the TOI path. Before the scheduled start,
        or while the batting side can't be determined (no toss yet), this
        raises CricbuzzInningsNotStarted instead -- a quiet wait."""
        start_ms = self._start_ms_from_html(html)
        if start_ms is not None and time.time() * 1000 < start_ms:
            raise CricbuzzInningsNotStarted(
                f"match {match_id} starts at {datetime.fromtimestamp(start_ms / 1000).strftime('%H:%M')} local; "
                "waiting for the scheduled start."
            )
        teams = self._teams_from_ldjson(html)
        toss_won_by, toss_decision = self._toss_from_chunks(chunks)
        winner = toss_won_by if toss_won_by in teams else _resolve_full_team_name(toss_won_by, teams)
        if len(teams) != 2 or not winner:
            raise CricbuzzInningsNotStarted(
                f"match {match_id}: toss/batting side not known yet; waiting."
            )
        batting_team = winner if toss_decision.lower().startswith("bat") else next(
            (team for team in teams if team != winner), ""
        )
        bowling_team = next((team for team in teams if team != batting_team), "")
        match_format = self._format_from_html(html)
        return LiveSnapshot(
            match_id=match_id,
            match_format=match_format,
            batting_team=batting_team,
            bowling_team=bowling_team,
            innings=1,
            score=0,
            wickets=0,
            overs="0.0",
            is_live=True,
            target=0,
            scheduled_overs={"T20": 20, "T20I": 20, "ODI": 50}.get(match_format, 0),
            deliveries=(),
            team_players=self._squads_from_chunks(chunks),
            venue_name=self._venue_from_chunks(chunks, match_id),
            toss_won_by=toss_won_by,
            toss_decision=toss_decision,
            board=self._board_from_chunks(chunks),
            raw_html=html,
        )

    @staticmethod
    def _balanced_after(chunks: list[str], marker: str) -> dict | None:
        for chunk in chunks:
            start = chunk.find(marker)
            if start == -1:
                continue
            raw = _extract_balanced(chunk, start + len(marker))
            if raw is None:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    @staticmethod
    def _board_from_chunks(chunks: list[str]) -> dict | None:
        """Everything the page itself says about the match state, independent
        of the ball-by-ball commentary we rebuild from: the "miniscore"
        (starred striker / bowler, score, overs, partnership), the
        "matchHeader" (state, status text, complete flag, result) and the
        site's own "winProbability" -- an external benchmark for our model
        that was on the page all along and never recorded (2026-10-06)."""
        board: dict = {}
        mini = CricbuzzLiveReader._balanced_after(chunks, '"miniscore":')
        if mini:
            def _name(key: str) -> str:
                value = mini.get(key)
                return str(value.get("name", "")) if isinstance(value, dict) else ""

            def _stats(value: Any) -> dict | None:
                if not isinstance(value, dict):
                    return None
                keep = ("runs", "balls", "fours", "sixes", "strikeRate", "overs", "maidens", "wickets", "economy")
                return {k: value[k] for k in keep if k in value}

            bat_team = mini.get("batTeam") if isinstance(mini.get("batTeam"), dict) else {}
            board.update(
                {
                    "innings": mini.get("inningsId"),
                    "score": bat_team.get("teamScore"),
                    "wickets": bat_team.get("teamWkts"),
                    "overs": mini.get("overs"),
                    "striker": _name("batsmanStriker"),
                    "non_striker": _name("batsmanNonStriker"),
                    "bowler": _name("bowlerStriker"),
                    "other_bowler": _name("bowlerNonStriker"),
                    # exact figures, not just names: runs/balls/4s/6s for batters, overs/maidens/runs/wickets for bowlers
                    "striker_stats": _stats(mini.get("batsmanStriker")),
                    "non_striker_stats": _stats(mini.get("batsmanNonStriker")),
                    "bowler_stats": _stats(mini.get("bowlerStriker")),
                    "other_bowler_stats": _stats(mini.get("bowlerNonStriker")),
                    "partnership": mini.get("partnerShip"),
                    "last_wicket": mini.get("lastWicket"),
                    "last_updated_s": mini.get("responseLastUpdated"),
                }
            )
        header = CricbuzzLiveReader._balanced_after(chunks, '"matchHeader":')
        if header:
            board.update(
                {
                    "state": header.get("state"),
                    "status": header.get("status"),
                    "complete": header.get("complete"),
                    "result": header.get("result") or None,
                }
            )
        win = CricbuzzLiveReader._balanced_after(chunks, '"winProbability":')
        if win:
            team1, team2 = win.get("team1") or {}, win.get("team2") or {}
            board["cb_win"] = {
                team1.get("name", "team1"): team1.get("percent"),
                team2.get("name", "team2"): team2.get("percent"),
                "as_of_over": win.get("lastUpdatedOver"),
            }
        return board or None

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

    def drain_over_summaries(self) -> list[dict]:
        """New per-over summaries (Cricbuzz's own result string and runs for each
        finished over) since the last call, oldest first."""
        summaries, self._pending_summaries = self._pending_summaries, []
        return sorted(summaries, key=lambda item: (item["innings"], item["over"]))

    def drain_repairs(self) -> list[str]:
        """Human-readable lines for every over rebuilt from Cricbuzz's own over
        summary (or score re-anchored to its checkpoint) since the last call."""
        repairs, self._pending_repairs = self._pending_repairs, []
        return repairs

    def drain_notes(self) -> list[tuple[int, str, str]]:
        """New (timestamp_ms, category, text) notes since the last call."""
        notes, self._pending_notes = self._pending_notes, []
        return sorted(notes)

    @staticmethod
    def _checkpoint(obj: dict[str, Any]) -> tuple[int, int] | None:
        over_separator = obj.get("overSeparator")
        if not isinstance(over_separator, dict):
            return None
        checkpoint_score = over_separator.get("batTeamObj", {}).get("teamScore", "")
        match = re.match(r"(\d+)-(\d+)", str(checkpoint_score))
        return (int(match.group(1)), int(match.group(2))) if match else None

    def _baseline_from_board(
        self, ordered: list[dict[str, Any]], board: dict | None, innings: int
    ) -> tuple[int, int] | None:
        """Score/wickets BEFORE the first ball of this window, taken from the page's
        live board when we attach mid-innings with no over-checkpoint in the window.

        Only used when the board is demonstrably at the same point as the window:
        same innings and the same over.ball as the newest delivery (Cricbuzz serves
        cached copies of different ages, so an older board must not be trusted).
        Field test 2026-10-09: two matches attached mid-innings with no checkpoint
        started from 0 and ran ~100 runs behind the board for the whole innings."""
        if not board or board.get("score") is None or board.get("overs") is None:
            return None
        if board.get("innings") not in (None, innings):
            return None
        balls = []
        seen_stamps: set = set()
        for o in ordered:
            if not isinstance(o.get("ballMetric"), (int, float)):
                continue
            # The last ball of an over is posted TWICE with the same timestamp (one copy tagged over-break). Counting both put the
            # baseline 1 run too low for the rest of the innings (2026-10-14, 16.6: ours 206 vs board 207 from 17.0 on).
            stamp = o.get("timestamp")
            if stamp is not None and stamp in seen_stamps:
                continue
            seen_stamps.add(stamp)
            balls.append(o)
        if not balls:
            return None
        try:
            board_overs = float(board["overs"])
        except (TypeError, ValueError):
            return None

        def position(prefix: list[dict[str, Any]]) -> float:
            """over.ball after the last ball of `prefix`, counting only legal balls of that over."""
            latest = max(prefix, key=lambda o: o.get("timestamp", 0))
            over_index = int(latest["ballMetric"])
            legal_in_over = 0
            for o in prefix:
                if int(o["ballMetric"]) != over_index:
                    continue
                _total, _bat, extras = _runs_and_extras(o.get("commText", ""), [str(e) for e in o.get("event", [])])
                if "wides" not in extras and "noballs" not in extras:
                    legal_in_over += 1
            return over_index + 1.0 if legal_in_over >= 6 else over_index + min(legal_in_over, 5) / 10.0

        # Cricbuzz serves the board and the commentary from caches of different ages (2026-10-13: for minutes the board was
        # a ball behind the commentary and no start was ever accepted). The board is the true score at ITS ball, so find that
        # ball in the window (longest prefix first) and take the baseline there; the balls after it are then added on top.
        first_over = int(min(balls, key=lambda o: o.get("timestamp", 0))["ballMetric"])
        first_over_complete_from_ball_one = min(
            (o["ballMetric"] - int(o["ballMetric"]) for o in balls if int(o["ballMetric"]) == first_over), default=1.0
        ) < 0.15
        for size in range(len(balls), 0, -1):
            prefix = balls[:size]
            last_over = int(max(prefix, key=lambda o: o.get("timestamp", 0))["ballMetric"])
            over_finished = size < len(balls) and int(balls[size]["ballMetric"]) > last_over  # a later over has started
            if over_finished:
                where = last_over + 1.0  # overs only move on once the previous one is complete
            elif last_over == first_over and not first_over_complete_from_ball_one:
                continue  # the window starts mid-over: the balls we hold of that over are not all of it, so its position is unknown
            else:
                where = position(prefix)
            if abs(board_overs - where) > 0.001:
                continue
            window_runs = window_wickets = 0
            for o in prefix:
                events = [str(e) for e in o.get("event", [])]
                total, _bat, _extras = _runs_and_extras(o.get("commText", ""), events)
                window_runs += total
                if "wicket" in events and _classify_wicket(o.get("commText", "")) is not None:
                    window_wickets += 1
            baseline = (int(board["score"]) - window_runs, int(board.get("wickets") or 0) - window_wickets)
            return baseline if baseline[0] >= 0 and baseline[1] >= 0 else None
        return None

    def _build_deliveries(
        self,
        scope: tuple[str, int],
        raw_objects: list[dict[str, Any]],
        board: dict | None = None,
        require_baseline: bool = False,
    ) -> list[LiveDelivery]:
        match_id, innings = scope
        # Cricbuzz lists newest-first; process oldest-first so running
        # totals accumulate in real chronological order.
        ordered = sorted(raw_objects, key=lambda obj: obj.get("timestamp", 0))
        known = self._all_deliveries.setdefault(scope, {})
        seen_timestamps = self._seen_timestamps.setdefault(scope, set())
        timestamp_keys = self._timestamp_keys.setdefault(scope, {})

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
                    timestamp = _int_field(obj, "timestamp", 0)
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
                        innings=_int_field(best_checkpoint_obj, "inningsId", innings),
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
                # No over-checkpoint in this window. That means "early in the
                # innings, 0/0" only if the window reaches the innings' first
                # balls; otherwise we attached mid-innings and 0/0 would be
                # wrong for the rest of the innings. Take the baseline from the
                # page's board (when it matches the window), else wait.
                starts = [
                    float(o["ballMetric"]) for o in ordered
                    if isinstance(o.get("ballMetric"), (int, float))
                ]
                # ball 0.1 must be IN the window: 2026-10-13 a window starting at 0.2 counted as "from the start", missed the 1 run
                # of ball 0.1 and sat one run behind the board until the next re-sync
                from_start = (not starts) or min(starts) < 0.15
                baseline = (0, 0)
                if require_baseline and not from_start:
                    baseline = self._baseline_from_board(ordered, board, innings)
                    if baseline is None:
                        raise CricbuzzFeedError(
                            f"Attached to Cricbuzz match {match_id} innings {innings} mid-innings with no "
                            "over-checkpoint in the window and no board that matches it yet -- "
                            "waiting for a consistent starting score."
                        )
                    self._pending_repairs.append(
                        f"innings {innings}: starting score {baseline[0]}/{baseline[1]} taken from the page's "
                        "board (attached mid-innings, no over-checkpoint in the window)"
                    )
                self._running_score[scope] = baseline[0]
                self._running_wickets[scope] = baseline[1]
                # With a board baseline, anything older than the window's first ball is
                # already inside that baseline: an older cached copy of the page must
                # not add those balls again (seen replaying 2026-10-09 pages: +37 runs).
                first_ts = min(
                    (_int_field(o, "timestamp", 0) for o in ordered
                     if isinstance(o.get("ballMetric"), (int, float))),
                    default=0,
                )
                self._seed_cutoff_ms[scope] = (first_ts - 1) if (require_baseline and not from_start) else -1

        running_score = self._running_score[scope]
        running_wickets = self._running_wickets[scope]
        cutoff_ms = self._seed_cutoff_ms[scope]

        for obj in ordered:
            ball_metric = obj.get("ballMetric")
            if not isinstance(ball_metric, (int, float)):
                # Non-ball commentary: not a delivery, but drinks breaks,
                # delays, injuries, reviews and bowling changes are worth
                # showing (see _classify_note) -- queue them for the session.
                note_ts = _int_field(obj, "timestamp", 0)
                note_text = re.sub(r"<[^>]+>", "", str(obj.get("commText", ""))).strip()
                category = _classify_note(note_text)
                if category and note_ts and note_ts not in self._seen_note_timestamps:
                    self._seen_note_timestamps.add(note_ts)
                    self._pending_notes.append((note_ts, category, note_text))
                continue
            separator = obj.get("overSeparator")
            if isinstance(separator, dict) and separator.get("overNumber") is not None:
                summary_key = (innings, _int_field(separator, "overNumber", 0))
                tokens = parse_over_summary(separator.get("overSummary", ""))
                if tokens is not None and not tokens_agree_with_total(tokens, separator.get("overRuns")):
                    tokens = None
                bowler_name = (separator.get("bowlerObj") or {}).get("playerName", "") if isinstance(separator.get("bowlerObj"), dict) else ""
                self._over_truth[(match_id, innings, summary_key[1])] = {
                    "tokens": tokens,
                    "checkpoint": self._checkpoint(obj),
                    "bowler": bowler_name,
                    "summary": str(separator.get("overSummary", "")).strip(),
                    "timestamp_ms": _int_field(obj, "timestamp", 0),
                }
                if summary_key not in self._seen_over_summaries:
                    self._seen_over_summaries.add(summary_key)
                    bowler_obj = separator.get("bowlerObj") if isinstance(separator.get("bowlerObj"), dict) else {}
                    self._pending_summaries.append(
                        {
                            "innings": innings,
                            "over": summary_key[1],
                            "summary": str(separator.get("overSummary", "")).strip(),
                            "over_runs": separator.get("overRuns"),
                            "score": (separator.get("batTeamObj") or {}).get("teamScore"),
                            "bowler": bowler_obj.get("playerName"),
                            "bowler_figures": bowler_obj.get("playerScore"),
                            "striker": (separator.get("batStrikerObj") or {}).get("playerName"),
                            "non_striker": (separator.get("batNonStrikerObj") or {}).get("playerName"),
                            "timestamp_ms": _int_field(obj, "timestamp", 0),
                        }
                    )
            timestamp = _int_field(obj, "timestamp", 0)
            if timestamp <= cutoff_ms:
                continue  # already folded into the seed checkpoint above
            if timestamp in seen_timestamps:
                # Same entry, re-read: Cricbuzz rewrites an entry's prose after
                # posting it (seen 2026-10-06: the same SIX had long prose in
                # one fetch and only "SIX, to deep mid-wicket" in the next),
                # so keep the fullest text ever seen rather than the first.
                known_key = timestamp_keys.get(timestamp)
                fuller = re.sub(r"<[^>]+>", "", obj.get("commText", ""))
                if known_key in known and len(fuller) > len(known[known_key].commentary):
                    known[known_key] = replace(known[known_key], commentary=fuller)
                continue  # already accumulated on an earlier poll
            over_index = int(ball_metric)
            comm_text = obj.get("commText", "")
            events = [str(e) for e in obj.get("event", [])]
            total_runs, batter_runs, extras = _runs_and_extras(comm_text, events)
            wicket_kind = _classify_wicket(comm_text) if "wicket" in events else None

            # Cricbuzz gives a re-bowled ball the SAME ballMetric as the wide
            # or no-ball it replaces (2.1 no-ball, then 2.1 again). Deduping
            # on (innings, over, ball) silently dropped the re-bowled ball, so
            # any over with an extra never reached 6 legal balls and was never
            # seen as complete -- the next over's prediction and the previous
            # over's review were skipped (2026-10-06 over 4; every saved match
            # with a wide/no-ball shows the same shortfall, incl. 09-15's
            # "over 14 skip"). Number deliveries by position in the over
            # instead: metric ball + illegal deliveries already seen in it.
            over_number = over_index + 1
            innings_id = _int_field(obj, "inningsId", 1)
            slot_keys = self._slots.setdefault(scope, {}).setdefault(round(float(ball_metric), 1), [])
            prior = next(
                (known[k] for k in reversed(slot_keys) if k in known and known[k].is_legal), None
            )
            if prior is not None:
                # The same slot was posted again while its earlier entry was a
                # LEGAL ball: that is a revision (a review replaced an OUT with
                # a not-out, or the prose/score was corrected), not a new ball.
                # Latest wins; later deliveries and the running totals shift by
                # the difference.
                delta_runs = total_runs - prior.total_runs
                delta_wkts = int(wicket_kind is not None) - int(prior.wicket_kind is not None)
                del known[prior.key]
                timestamp_keys[prior.timestamp_ms] = None
                for later_key, later in list(known.items()):
                    if later_key > prior.key:
                        known[later_key] = replace(
                            later,
                            feed_total=later.feed_total + delta_runs,
                            feed_wickets=later.feed_wickets + delta_wkts,
                        )
                running_score += delta_runs
                running_wickets += delta_wkts
                seen_timestamps.add(timestamp)
                timestamp_keys[timestamp] = prior.key
                known[prior.key] = LiveDelivery(
                    match_id=match_id,
                    innings=prior.innings,
                    over=prior.over,
                    ball=prior.ball,
                    bowler=obj.get("bowlerDetails", {}).get("playerName", "") or names_from_text(comm_text)[0] or prior.bowler,
                    striker=obj.get("batsmanDetails", {}).get("playerName", "") or names_from_text(comm_text)[1] or prior.striker,
                    total_runs=total_runs,
                    batter_runs=batter_runs,
                    extras=extras,
                    wicket_kind=wicket_kind,
                    feed_total=prior.feed_total + delta_runs,
                    feed_wickets=prior.feed_wickets + delta_wkts,
                    timestamp_ms=_int_field(obj, "timestamp", 0),
                    commentary=re.sub(r"<[^>]+>", "", comm_text),
                    dismissed=dismissed_batter(
                        wicket_kind,
                        re.sub(r"<[^>]+>", "", comm_text),
                        obj.get("batsmanDetails", {}).get("playerName", "") or names_from_text(comm_text)[1] or prior.striker,
                    ),
                )
                self._pending_repairs.append(
                    f"innings {innings} ball {prior.over - 1}.{prior.ball}: entry revised by Cricbuzz "
                    f"({prior.total_runs} run(s){', wicket' if prior.wicket_kind else ''} -> "
                    f"{total_runs} run(s){', wicket' if wicket_kind else ''}); latest entry kept"
                )
                slot_keys.append(prior.key)
                continue
            illegal_before = sum(
                1
                for (_i, existing_over, _b), existing in known.items()
                if existing_over == over_number and not existing.is_legal
            )
            ball_number = round((ball_metric - over_index) * 10) + illegal_before
            key = (innings_id, over_number, ball_number)
            while key in known:
                key = (innings_id, over_number, key[2] + 1)
            ball_number = key[2]
            seen_timestamps.add(timestamp)
            timestamp_keys[timestamp] = key
            slot_keys.append(key)

            running_score += total_runs
            if wicket_kind is not None:
                running_wickets += 1

            checkpoint = self._checkpoint(obj)
            if checkpoint is not None:
                checkpoint_runs, checkpoint_wkts = checkpoint
                truth = self._over_truth.get(
                    (match_id, innings, _int_field(obj.get("overSeparator") or {}, "overNumber", 0))
                )
                repairable = truth is not None and truth["tokens"] is not None
                if (checkpoint_runs != running_score or checkpoint_wkts != running_wickets) and not repairable:
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
                bowler=obj.get("bowlerDetails", {}).get("playerName", "") or names_from_text(comm_text)[0],
                striker=obj.get("batsmanDetails", {}).get("playerName", "") or names_from_text(comm_text)[1],
                total_runs=total_runs,
                batter_runs=batter_runs,
                extras=extras,
                wicket_kind=wicket_kind,
                feed_total=running_score,
                feed_wickets=running_wickets,
                timestamp_ms=_int_field(obj, "timestamp", 0),
                commentary=re.sub(r"<[^>]+>", "", comm_text),
                dismissed=dismissed_batter(
                    wicket_kind,
                    re.sub(r"<[^>]+>", "", comm_text),
                    obj.get("batsmanDetails", {}).get("playerName", "") or names_from_text(comm_text)[1],
                ),
            )

        if known:
            reconciled = self._reconcile_overs(scope, known, timestamp_keys)
            if reconciled is not None:
                running_score, running_wickets = reconciled
        self._running_score[scope] = running_score
        self._running_wickets[scope] = running_wickets
        return sorted(known.values(), key=lambda delivery: delivery.key)

    def _reconcile_overs(
        self,
        scope: tuple[str, int],
        known: dict[tuple[int, int, int], LiveDelivery],
        timestamp_keys: dict[int, tuple[int, int, int] | None],
    ) -> tuple[int, int] | None:
        """Make every finished over agree with Cricbuzz's own over summary, then
        re-chain the running score/wickets from the latest checkpoint.

        Ball entries are not append-only (prose rewritten, re-bowled balls reuse
        a ball number, a review replaces an earlier OUT), so an over built only
        from the stream can hold a phantom wicket or wrong runs. The over
        summary is posted once and carries one token per delivery. Returns the
        corrected (score, wickets), or None when nothing needed repair.
        """
        match_id, innings = scope
        ordered_before = sorted(known.values(), key=lambda d: d.key)
        base_score = ordered_before[0].feed_total - ordered_before[0].total_runs
        base_wickets = ordered_before[0].feed_wickets - int(ordered_before[0].wicket_kind is not None)
        changed = False
        for (m_id, inn, over), truth in sorted(self._over_truth.items()):
            if m_id != match_id or inn != innings or truth["tokens"] is None:
                continue
            existing = sorted((d for d in known.values() if d.over == over), key=lambda d: d.key)
            if not existing:
                continue  # an over we never saw ball by ball (attached after it)
            tokens = truth["tokens"]
            have = [(d.total_runs, d.wicket_kind is not None, d.is_legal) for d in existing]
            want = [(t.total_runs, t.wicket, t.is_legal) for t in tokens]
            if have == want:
                continue
            changed = True
            slot_map = self._slots.get(scope, {})
            for slot in [m for m in slot_map if int(m) == over - 1]:
                del slot_map[slot]
            entries = sorted(existing, key=lambda d: d.timestamp_ms)
            for old in existing:
                del known[old.key]
                timestamp_keys[old.timestamp_ms] = None  # stale entries must not return
            entry_sig = [(e.total_runs, e.wicket_kind is not None, e.is_legal) for e in entries]
            matched = _align(want, entry_sig)  # token index -> entry index
            for index, token in enumerate(tokens, start=1):
                match_at = matched.get(index - 1)
                if match_at is not None:
                    delivery = replace(entries[match_at], ball=index)
                else:
                    anchor = entries[min(index - 1, len(entries) - 1)]
                    delivery = LiveDelivery(
                        match_id=match_id,
                        innings=innings,
                        over=over,
                        ball=index,
                        bowler=truth["bowler"] or anchor.bowler,
                        striker="",
                        total_runs=token.total_runs,
                        batter_runs=token.batter_runs,
                        extras=dict(token.extras),
                        wicket_kind="wicket" if token.wicket else None,
                        feed_total=0,
                        feed_wickets=0,
                        timestamp_ms=truth["timestamp_ms"] or anchor.timestamp_ms,
                        commentary="(rebuilt from Cricbuzz over summary; no matching commentary entry)",
                    )
                known[delivery.key] = delivery
                timestamp_keys[delivery.timestamp_ms] = delivery.key
            self._pending_repairs.append(
                f"innings {innings} over {over}: rebuilt from Cricbuzz over summary "
                f"'{truth['summary']}' (we had {len(have)} deliveries, "
                f"{sum(h[0] for h in have)} runs, {sum(h[1] for h in have)} wickets; "
                f"summary says {len(want)}, {sum(w[0] for w in want)}, {sum(w[1] for w in want)})"
            )

        ordered = sorted(known.values(), key=lambda d: d.key)

        def _chain(score0: int, wickets0: int) -> list[LiveDelivery]:
            chained, score, wickets = [], score0, wickets0
            for d in ordered:
                score += d.total_runs
                wickets += int(d.wicket_kind is not None)
                chained.append(replace(d, feed_total=score, feed_wickets=wickets))
            return chained

        chained = _chain(base_score, base_wickets)
        # Re-anchor on the latest fully-known over that has a usable checkpoint,
        # so a gap or a stale base cannot leave the absolute score wrong.
        for (m_id, inn, over), truth in sorted(self._over_truth.items(), reverse=True):
            if m_id != match_id or inn != innings or truth["tokens"] is None or truth["checkpoint"] is None:
                continue
            if sum(1 for d in ordered if d.over == over) != len(truth["tokens"]):
                continue
            end = max((d for d in chained if d.over == over), key=lambda d: d.key)
            delta_score = truth["checkpoint"][0] - end.feed_total
            delta_wickets = truth["checkpoint"][1] - end.feed_wickets
            if delta_score or delta_wickets:
                changed = True
                self._pending_repairs.append(
                    f"innings {innings}: score re-anchored to Cricbuzz checkpoint after over {over} "
                    f"({end.feed_total}/{end.feed_wickets} -> {truth['checkpoint'][0]}/{truth['checkpoint'][1]})"
                )
                chained = _chain(base_score + delta_score, base_wickets + delta_wickets)
            break
        if not changed and all(c == d for c, d in zip(chained, ordered)):
            return None
        for delivery in chained:
            known[delivery.key] = delivery
        return chained[-1].feed_total, chained[-1].feed_wickets

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
