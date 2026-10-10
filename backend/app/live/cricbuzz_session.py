"""Per-match polling state for a Cricbuzz-backed live match.

Extracted 2026-09-18 from run_cricbuzz_match.py (previously all of this
lived inline in that single-match script's main() loop) so
run_cricbuzz_auto.py's multi-match discovery driver can reuse the exact
same real-feed-inconsistency defenses per match instead of re-deriving a
thinner version of them: the freshest-of-3 fetch and the never-go-
backwards progress guard (docs/live_field_test_2026-09-17.md, confirmed
live that Cricbuzz's own feed is not monotonic poll-to-poll), and the
auto-reseed-after-3-consecutive-errors recovery (same doc). Every one of
these exists because of a real bug seen live, not speculative hardening
-- dropping them for the auto-discovery path would be a real regression
against the single-match path's already-earned reliability.
"""

from __future__ import annotations

import re
import json
from datetime import datetime, timezone
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import requests

from app.live.commentary_log import append_commentary_log
from app.live.cricbuzz_reader import (
    CricbuzzFeedError,
    CricbuzzInningsNotStarted,
    CricbuzzLiveReader,
    _extract_flight_chunks,
)
from app.live.prematch_analysis import confirmed_xi, prematch_header, prematch_text, toss_line, xi_line
from app.live.pipeline import (
    MatchStateAnomaly,
    PredictionPublisher,
    VerifiedLivePredictionPipeline,
)
from app.live.shot_quality import classify_near_wicket, classify_shot_quality
from app.live.delivery_detail import extract_delivery_detail, format_delivery_detail
from app.live.verification import VerificationError

_EXTRA_LABELS = {"wides": "WIDE", "noballs": "NO-BALL", "byes": "BYE", "legbyes": "LEG-BYE"}

RESEED_AFTER_CONSECUTIVE_ERRORS = 3


def _progress(snapshot) -> tuple[int, int]:
    overs_completed, _, balls_into_over = str(snapshot.overs).partition(".")
    legal_balls = int(overs_completed or 0) * 6 + int(balls_into_over or 0)
    return snapshot.innings, legal_balls


def _fetch_freshest(reader: CricbuzzLiveReader, match_url: str, attempts: int = 3):
    """Try the fetch a few times within one poll and keep the best.

    docs/live_field_test_2026-09-17.md: confirmed live that three
    back-to-back fresh requests for the same match can return three
    different snapshots from Cricbuzz (one genuinely current, two stale)
    -- not a client-side caching artifact, a real upstream inconsistency.
    """
    best_snapshot = None
    best_rank = None
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            snapshot = reader.fetch(match_url)
        except (CricbuzzFeedError, requests.exceptions.RequestException) as error:
            last_error = error
            continue
        rank = _progress(snapshot)
        if best_rank is None or rank > best_rank:
            best_snapshot, best_rank = snapshot, rank
    return best_snapshot, last_error


def _match_result_text(snapshot) -> str | None:
    """None while the match is still live."""
    if snapshot.innings < 2:
        return None
    target_reached = snapshot.target > 0 and snapshot.score >= snapshot.target
    if target_reached:
        wickets_in_hand = 10 - snapshot.wickets
        return (
            f"{snapshot.batting_team} won by {wickets_in_hand} wicket"
            f"{'s' if wickets_in_hand != 1 else ''} "
            f"({snapshot.score}/{snapshot.wickets})"
        )
    if snapshot.innings_complete:
        margin = snapshot.target - 1 - snapshot.score
        return (
            f"{snapshot.bowling_team} won by {margin} run"
            f"{'s' if margin != 1 else ''} "
            f"({snapshot.batting_team} {snapshot.score}/{snapshot.wickets})"
        )
    return None


_RUN_OUT_NAME = re.compile(r"([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*)*)\s+[Rr]un\s+[Oo]ut")


def _dismissed_batter(delivery) -> str:
    """Who actually went out. For a run-out it can be the non-striker (seen
    live 2026-10-06: Shimron Hetmyer run out while Shai Hope faced the ball,
    and the event line named Hope). Every other dismissal is the striker."""
    if delivery.wicket_kind == "run out":
        match = _RUN_OUT_NAME.search(delivery.commentary)
        if match:
            return match.group(1)
    return delivery.striker


def _ball_label(snapshot, delivery) -> str:
    """Standard cricket position (7.1 = 1st ball of the 8th over), counting
    only legal balls, so event lines agree with the scorecard's "7.1 ov"."""
    illegal_before = sum(
        1
        for other in snapshot.deliveries
        if other.over == delivery.over and other.ball < delivery.ball and not other.is_legal
    )
    return f"{delivery.over - 1}.{max(1, delivery.ball - illegal_before)}"


def _newest_board(previous: dict | None, current: dict | None) -> dict | None:
    """Cricbuzz serves cached copies of different ages: consecutive polls gave
    boards at 79/2, 102/2, 117/2, 121/2, so the striker/bowler/score shown
    flickered back to stale values. Keep the board with the newest
    `last_updated_s` (then innings/overs) and ignore older copies."""
    if not current:
        return previous
    if not previous:
        return current

    def stamp(board: dict) -> tuple:
        try:
            overs = float(board.get("overs") or 0)
        except (TypeError, ValueError):
            overs = 0.0
        return (board.get("innings") or 0, board.get("last_updated_s") or 0, overs)

    return current if stamp(current) >= stamp(previous) else previous


_ANNOUNCED_NAME = re.compile(r"^\s*([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*)*)")


def _announced_bowler_name(note_text: str) -> str:
    """"Matthew Forde [1.0-0-15-0] is back into the attack" / "Naman Dhir,
    right-arm off break, comes into the attack" -> the bowler's name."""
    match = _ANNOUNCED_NAME.match(note_text)
    return match.group(1).strip() if match else ""


def _with_announced_bowler(snapshot, announced: tuple[int, str] | None):
    """Put the bowler named by a "comes/is back into the attack" note on the
    snapshot (the fields the TOI path already fills), but only when the note was
    posted after the last delivery -- i.e. it names the NEXT over's bowler.
    Cricbuzz posts it ~30-60 s after the over ends, so the over-end prediction
    usually runs before it exists (field test 2026-10-06)."""
    if not announced or not snapshot.deliveries or not str(snapshot.overs).endswith(".0"):
        return snapshot
    note_ts, name = announced
    if note_ts <= snapshot.deliveries[-1].timestamp_ms:
        return snapshot
    try:
        next_over = int(str(snapshot.overs).split(".")[0]) + 1
    except ValueError:
        return snapshot
    return replace(snapshot, announced_bowler=name, announced_bowler_over=next_over)


def _expected_next_over(snapshot, announced: tuple[int, str] | None) -> dict | None:
    """What the next over should look like, derived at the end of an over (see
    _next_over_line): {"over": n, "bowler": name or "", "striker": name or "",
    "other": name or ""}, or None when it is not an over end."""
    board = getattr(snapshot, "board", None)
    if (
        not board
        or not str(snapshot.overs).endswith(".0")
        or snapshot.innings_complete
        or not snapshot.deliveries
    ):
        return None
    last = snapshot.deliveries[-1]
    bowler = announced[1] if announced and announced[0] > last.timestamp_ms else ""
    pair = [name for name in (board.get("striker"), board.get("non_striker")) if name]
    striker = other = ""
    if not last.wicket_kind and len(pair) == 2 and last.striker in pair:
        # Derive from how the LAST ball was scored, not from the board, which
        # may or may not have switched ends yet (at 14.0 it already had; in
        # other overs it had not). A boundary or a dot/even run leaves the
        # batter on strike, then the ends change for the new over, so the
        # OTHER batter faces; an odd number of runs crossed them first, so the
        # same batter faces again.
        boundary = last.batter_runs in (4, 6) and last.total_runs == last.batter_runs
        crossed = (not boundary) and last.total_runs % 2 == 1
        striker = last.striker if crossed else next(n for n in pair if n != last.striker)
        other = next(n for n in pair if n != striker)
    return {"over": last.over + 1, "bowler": bowler, "striker": striker, "other": other}


def _next_over_line(snapshot, announced: tuple[int, str] | None) -> str:
    """At the end of an over Cricbuzz's board still shows the finished over's
    bowler and batting positions until the next over's first ball (user,
    2026-10-06: "still previous bowler and batter until one ball is bowled").
    Show what is expected instead, clearly marked as expected: the bowler
    announced in the commentary (if any since the last ball), and the batters
    at the opposite ends."""
    expected = _expected_next_over(snapshot, announced)
    if expected is None:
        return ""
    if expected["striker"]:
        names = f"{expected['striker']}* expected to face, {expected['other']} at the other end"
    else:
        names = "batters: see board after the first ball"
    return (
        f"   [NEXT OVER, expected: {expected['bowler'] or 'bowler not announced yet'} bowling; {names}]"
    )


def _check_next_over(expected: dict | None, snapshot) -> str | None:
    """Compare a previously derived next-over expectation with the first ball
    actually bowled; returns the report line or None if it cannot be judged yet."""
    if not expected:
        return None
    first = next(
        (d for d in sorted(snapshot.deliveries, key=lambda d: d.key) if d.over == expected["over"]),
        None,
    )
    if first is None:
        return None
    parts = []
    if expected["striker"]:
        parts.append(
            f"striker expected {expected['striker']}, actual {first.striker}: "
            + ("HIT" if expected["striker"] == first.striker else "MISS")
        )
    if expected["bowler"]:
        parts.append(
            f"bowler announced {expected['bowler']}, actual {first.bowler}: "
            + ("HIT" if expected["bowler"] == first.bowler else "MISS")
        )
    return f"[check] over {expected['over']}: " + ("; ".join(parts) or "nothing was derived")


def _board_players(snapshot) -> str:
    """Starred players from the page's live board: striker* / current bowler*."""
    board = getattr(snapshot, "board", None)
    if not board or not board.get("striker"):
        return ""
    return (
        f"   [NOW: {board.get('bowler', '?')}* bowling, {board['striker']}* on strike, "
        f"{board.get('non_striker') or '?'} at the other end]"
    )


def _board_check(snapshot) -> str | None:
    """Disagreement between the state we rebuilt from ball-by-ball commentary
    and the page's own live scoreboard, or None. This is the check that would
    have caught tonight's wrong "Score: 9/2" (board said 53/3), the phantom
    wide that stopped over 19 from firing (board overs 18.1, ours 17.5), and
    the batting-team flicker."""
    board = getattr(snapshot, "board", None)
    if not board or board.get("score") is None:
        return None
    problems = []
    if board.get("innings") not in (None, snapshot.innings):
        problems.append(f"innings {snapshot.innings} vs board {board['innings']}")
    if (board["score"], board.get("wickets")) != (snapshot.score, snapshot.wickets):
        problems.append(
            f"score {snapshot.score}/{snapshot.wickets} vs board {board['score']}/{board.get('wickets')}"
        )
    try:
        board_overs = float(board.get("overs"))
    except (TypeError, ValueError):
        board_overs = None
    if board_overs is not None and abs(board_overs - float(snapshot.overs)) > 0.001:
        problems.append(f"overs {snapshot.overs} vs board {board_overs:g}")
    return "; ".join(problems) or None


def _prediction_hold(snapshot) -> str | None:
    """Reason to withhold this poll's prediction, or None.

    When the page's own board is at the SAME over/ball as our rebuilt state but
    shows a different score or wicket count, our state is wrong and any model
    input built from it is garbage (the 2026-10-06 "Score: 9/2" for 53/3, and
    the phantom wicket after a reviewed catch). An older cached board has fewer
    overs, so it never triggers this; a newer one means we are behind, which the
    ball-by-ball catch-up handles. Not applied once the innings is over."""
    board = getattr(snapshot, "board", None)
    if snapshot.innings_complete or not board or board.get("score") is None:
        return None
    if board.get("innings") not in (None, snapshot.innings):
        return None
    try:
        same_point = abs(float(board.get("overs")) - float(snapshot.overs)) < 0.001
    except (TypeError, ValueError):
        return None
    if not same_point:
        return None
    if (board["score"], board.get("wickets")) == (snapshot.score, snapshot.wickets):
        return None
    return (
        f"at {snapshot.overs} ov ours {snapshot.score}/{snapshot.wickets} but the board says "
        f"{board['score']}/{board.get('wickets')}"
    )


def _audit_over(snapshot, summary: dict) -> str:
    """Compare one finished over as we rebuilt it with Cricbuzz's own per-over
    summary (runs and the ball-by-ball result string)."""
    ours = [
        d for d in snapshot.deliveries
        if d.innings == summary["innings"] and d.over == summary["over"]
    ]
    if not ours:
        return f"[audit] over {summary['over']}: not seen live (attached after it) -- Cricbuzz '{summary['summary']}'"
    ours_runs = sum(d.total_runs for d in ours)
    their_balls = len(summary["summary"].split())
    ok_runs = summary["over_runs"] is None or ours_runs == int(summary["over_runs"])
    ok_balls = not summary["summary"] or len(ours) == their_balls
    verdict = "OK" if (ok_runs and ok_balls) else "MISMATCH"
    return (
        f"[audit] over {summary['over']}: {verdict} -- ours {ours_runs} runs/{len(ours)} deliveries, "
        f"Cricbuzz {summary['over_runs']} runs/{their_balls} deliveries ('{summary['summary']}'), "
        f"{summary.get('bowler')} {summary.get('bowler_figures')}"
    )


def _archive_page(snapshot, log_path, state: dict) -> None:
    """Keep what the website sent us, not only what we derived from it.
    Writes the raw page (gzip) when the board or delivery count changes (at
    most every 10 s) and one line per change to board-<match>.jsonl with the
    board, Cricbuzz's own win probability and the match status text. Added
    2026-10-06 when the page was found to carry data (winProbability,
    matchHeader, scorecard...) that was neither read nor saved."""
    import gzip
    import hashlib
    import time as _time

    board = getattr(snapshot, "board", None) or {}
    signature = json.dumps(
        [board.get("score"), board.get("wickets"), board.get("overs"), board.get("status"),
         (board.get("cb_win") or {}).get("as_of_over"), len(snapshot.deliveries)],
        default=str,
    )
    now = _time.time()
    if signature == state.get("signature") or now - state.get("at", 0.0) < 10.0:
        return
    state["signature"], state["at"] = signature, now
    root = log_path.parent.parent / "raw" / "cricbuzz-pages" / str(snapshot.match_id)
    root.mkdir(parents=True, exist_ok=True)
    html = getattr(snapshot, "raw_html", "") or ""
    if html:
        digest = hashlib.sha1(html.encode("utf-8")).hexdigest()[:10]
        (root / f"{int(now * 1000)}-{digest}.html.gz").write_bytes(gzip.compress(html.encode("utf-8")))
    with log_path.with_name(f"board-{snapshot.match_id}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "fetched_at": now,
                    "innings": snapshot.innings,
                    "ours": {"score": snapshot.score, "wickets": snapshot.wickets, "overs": snapshot.overs},
                    "board": board,
                },
                default=str,
            )
            + "\n"
        )


def _delivery_event_lines(snapshot, seen_keys: set[tuple[int, int, int]]) -> list[str]:
    lines = []
    for delivery in snapshot.deliveries:
        if delivery.key in seen_keys:
            continue
        seen_keys.add(delivery.key)
        detail_text = format_delivery_detail(extract_delivery_detail(delivery.commentary))
        if detail_text:
            lines.append(
                f"[ball] {_ball_label(snapshot, delivery)} {delivery.bowler} to {delivery.striker}: "
                f"{delivery.total_runs} run(s); {detail_text}"
            )
        tags = []
        if delivery.batter_runs == 6:
            tags.append("SIX")
        for extra_key, label in _EXTRA_LABELS.items():
            if extra_key in delivery.extras:
                tags.append(label)
        if delivery.wicket_kind:
            tags.append(f"WICKET({delivery.wicket_kind})")
        if tags:
            lines.append(
                f"[event] inn{delivery.innings} {_ball_label(snapshot, delivery)} "
                f"{'/'.join(tags)} -- "
                f"{_dismissed_batter(delivery) if delivery.wicket_kind else delivery.striker} "
                f"off {delivery.bowler}"
            )
        # User-requested 2026-09-18: surface shot quality/near-wicket
        # signals live, not just log them for later -- see
        # app/live/shot_quality.py's docstring. An edge for four scores
        # exactly like a well-timed four in every structured field
        # above; this is the only place that distinction is visible.
        quality = classify_shot_quality(delivery.commentary)
        if quality == "edge" and not delivery.wicket_kind:
            note = (
                f"⚠️  STREAKY: inn{delivery.innings} {_ball_label(snapshot, delivery)} "
                f"{delivery.striker} edged that for {delivery.total_runs} off "
                f"{delivery.bowler} -- not a clean shot despite the runs"
            )
            lines.append(note)
        elif quality == "beaten" and not delivery.wicket_kind:
            lines.append(
                f"⚠️  BEATEN: inn{delivery.innings} {_ball_label(snapshot, delivery)} "
                f"{delivery.striker} played and missed off {delivery.bowler}"
            )
        near_wicket = classify_near_wicket(delivery.commentary)
        if near_wicket:
            lines.append(
                f"⚠️  {near_wicket.upper()}: inn{delivery.innings} {_ball_label(snapshot, delivery)} "
                f"{delivery.striker} off {delivery.bowler} -- real wicket chance, not on the scorecard"
            )
    return lines


def briefing_text(match_id: str, match_url: str, label: str) -> str | None:
    """One-time pre-match summary: toss, venue, playing XIs.

    Shared by both run_cricbuzz_match.py (single match, URL given
    directly) and run_cricbuzz_auto.py (discovery-driven) so a briefing
    is never available on only one of the two Cricbuzz entry points --
    real gap found 2026-09-18: this was first built only inside
    run_cricbuzz_auto.py, but run_cricbuzz_match.py is the driver both
    real field tests (09-15, 09-17) actually used, so it would have had
    no briefing at all despite the underlying extraction being real and
    working. Mirrors run_toi_auto.py's own _briefing_text -- same idea
    (surface what's known right when a match is first attached), same
    graceful None-on-any-failure contract so a briefing failure never
    blocks the per-over prediction flow that follows it. Cricbuzz's
    pitch/weather enrichment doesn't exist at all (checked directly, see
    cricbuzz_reader.py's module docstring), so those two lines from
    TOI's briefing have no Cricbuzz equivalent here.
    """
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; CricketBaba/3.0)"})
        response = session.get(match_url, timeout=12.0)
        response.raise_for_status()
        chunks = _extract_flight_chunks(response.text)
    except Exception:
        return None

    toss_won_by, toss_decision = CricbuzzLiveReader._toss_from_chunks(chunks)
    venue = CricbuzzLiveReader._venue_from_chunks(chunks, match_id)
    team_players = CricbuzzLiveReader._squads_from_chunks(chunks)
    # run_cricbuzz_match.py (single match, given only a URL) has no
    # discovery-derived label to pass in -- the page's own schema.org
    # JSON-LD team names (same source _resolve_full_team_name's callers
    # already trust) make a real one instead of falling back to the bare
    # match id.
    teams = CricbuzzLiveReader._teams_from_ldjson(response.text)
    label = " vs ".join(teams) if teams else label

    from app.live.prematch_model import venue_city

    city = venue_city(venue or "").title()
    lines = [f"🏏 {label}" + (f" -- {city}" if city else "")]
    if len(teams) == 2:
        try:
            lines.extend(
                prematch_header(
                    teams[0], teams[1], venue or "", toss_won_by, toss_decision, confirmed_xi(team_players),
                    start_utc_ms=CricbuzzLiveReader._start_ms_from_html(response.text),
                )
            )
        except Exception as error:  # never block the briefing, but do not hide the bug
            print(f"[briefing] header unavailable: {error!r}")
    lines.append(toss_line(toss_won_by, toss_decision, teams[0] if len(teams) == 2 else "", teams[1] if len(teams) == 2 else ""))
    lines.append(xi_line(team_players))
    if len(teams) == 2:
        analysis = prematch_text(teams[0], teams[1], venue or "", toss_won_by, toss_decision, confirmed_xi(team_players))
        if analysis:
            lines.append("")
            lines.append(analysis)
    return "\n".join(lines)


def _scorecard_line(snapshot) -> str:
    deliveries = snapshot.deliveries
    latest = deliveries[-1] if deliveries else None
    overs_completed, _, balls_into_over = str(snapshot.overs).partition(".")
    legal_balls = int(overs_completed or 0) * 6 + int(balls_into_over or 0)
    run_rate = (snapshot.score * 6 / legal_balls) if legal_balls else 0.0
    batter = latest.striker if latest else "?"
    bowler = latest.bowler if latest else "?"
    return (
        f"[scorecard] {snapshot.batting_team} {snapshot.score}/{snapshot.wickets} "
        f"({snapshot.overs} ov, RR {run_rate:.2f}) -- last ball: {bowler} to {batter}"
        + _board_players(snapshot)
    )


@dataclass
class CricbuzzMatchSession:
    """One match's full polling state -- fetch, guard, predict, print."""

    match_id: str
    match_url: str
    pipeline: VerifiedLivePredictionPipeline
    publisher: PredictionPublisher | None = None
    reader: CricbuzzLiveReader = field(default_factory=CricbuzzLiveReader)
    label: str = ""
    _seen_delivery_keys: set[tuple[int, int, int]] = field(default_factory=set)
    _last_over_sent: dict[int, int] = field(default_factory=dict)
    _consecutive_feed_errors: int = 0
    _best_progress: tuple[int, int] | None = None
    _finished: bool = False
    # Observability, punch-list item 11 (docs/live_field_test_2026-09-15.md):
    # every real bug this project has found live was only ever caught
    # because a human was watching the console. run_toi_live.py (an older,
    # no-longer-primary single-match TOI driver) already had exactly this
    # kind of alert for its own feed errors, but it never got ported to any
    # driver actually in use -- not run_toi_auto.py, not the Cricbuzz
    # drivers, not run_live_auto.py. Ported here so the ACTUAL current
    # driver alerts automatically instead of silently retrying forever
    # with only a console print nobody is watching.
    stuck_alert_seconds: float = 300.0
    _error_streak_started_at: float | None = None
    _waiting_notice_printed: bool = False
    _last_board_mismatch: str | None = None
    _newest_board_seen: dict | None = None
    _announced_bowler: tuple[int, str] | None = None
    _last_hold: str | None = None
    _expected_next: dict | None = None
    _archive_state: dict = field(default_factory=dict)
    _stuck_alert_sent: bool = False
    # User-requested 2026-09-18: a persistent per-delivery commentary log
    # for detailed post-match review -- see app/live/commentary_log.py's
    # module docstring for why (today's real bugs were only caught by
    # reading full commentary text, not the short tagged-event console
    # lines below). None disables logging.
    commentary_log_path: Path | None = None
    _logged_delivery_keys: set[tuple[int, int, int]] = field(default_factory=set)

    @property
    def finished(self) -> bool:
        return self._finished

    def poll_once(self) -> None:
        """Fetch, guard, and (if a boundary was reached) predict for one
        cycle. All output is printed; nothing is returned -- state (match
        completion, Telegram publication) lives on self/self.pipeline."""
        snapshot, error = _fetch_freshest(self.reader, self.match_url)
        if snapshot is None and isinstance(error, CricbuzzInningsNotStarted):
            if not self._waiting_notice_printed:
                print(f"[{self.match_id}] Waiting: {error}")
                self._waiting_notice_printed = True
            self._consecutive_feed_errors = 0
            self._error_streak_started_at = None
            return
        if snapshot is None:
            label = "Deferred" if isinstance(error, CricbuzzFeedError) else "Deferred (network)"
            print(f"[{self.match_id}] {label}: {error}")
            self._consecutive_feed_errors += 1
            if self._consecutive_feed_errors >= RESEED_AFTER_CONSECUTIVE_ERRORS:
                print(
                    f"[{self.match_id}] Reseeding Cricbuzz reader after "
                    f"{self._consecutive_feed_errors} consecutive feed errors."
                )
                self.reader = CricbuzzLiveReader()
                self._consecutive_feed_errors = 0
            self._check_stuck(error)
            return

        self._consecutive_feed_errors = 0
        self._error_streak_started_at = None
        self._stuck_alert_sent = False

        progress = _progress(snapshot)
        if self._best_progress is not None and progress < self._best_progress:
            print(
                f"[{self.match_id}] Stale snapshot skipped: innings "
                f"{snapshot.innings} at {snapshot.overs} ov is behind the "
                f"innings {self._best_progress[0]} / {self._best_progress[1]} "
                "legal balls already seen."
            )
            return
        self._best_progress = (
            progress if self._best_progress is None else max(self._best_progress, progress)
        )

        self._newest_board_seen = _newest_board(self._newest_board_seen, snapshot.board)
        if self._newest_board_seen is not snapshot.board:
            snapshot = replace(snapshot, board=self._newest_board_seen)
        if self.commentary_log_path is not None:
            try:
                _archive_page(snapshot, self.commentary_log_path, self._archive_state)
            except OSError:
                pass
        mismatch = _board_check(snapshot)
        if mismatch and mismatch != self._last_board_mismatch:
            print(f"[{self.match_id}] ⚠️  BOARD MISMATCH: ours {mismatch}")
        self._last_board_mismatch = mismatch
        drain_repairs = getattr(self.reader, "drain_repairs", None)
        for repair in (drain_repairs() if drain_repairs else []):
            print(f"[{self.match_id}] [repair] {repair}")
            if self.commentary_log_path is not None:
                repairs_path = self.commentary_log_path.with_name(f"repairs-{self.match_id}.jsonl")
                with repairs_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "repair": repair}) + "\n")
        drain_summaries = getattr(self.reader, "drain_over_summaries", None)
        for summary in (drain_summaries() if drain_summaries else []):
            print(f"[{self.match_id}] {_audit_over(snapshot, summary)}")
            if self.commentary_log_path is not None:
                archive = self.commentary_log_path.with_name(f"over-summaries-{self.match_id}.jsonl")
                with archive.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(summary) + "\n")
        drain_notes = getattr(self.reader, "drain_notes", None)
        for note_ts, category, note_text in (drain_notes() if drain_notes else []):
            print(f"[{self.match_id}] [note] {category}: {note_text[:240]}")
            if category == "BOWLING CHANGE":
                announced_name = _announced_bowler_name(note_text)
                if announced_name:
                    self._announced_bowler = (note_ts, announced_name)
            if self.commentary_log_path is not None:
                notes_path = self.commentary_log_path.with_name(
                    f"commentary-notes-{self.match_id}.jsonl"
                )
                with notes_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {"timestamp_ms": note_ts, "category": category, "text": note_text}
                        )
                        + "\n"
                    )
        check = _check_next_over(self._expected_next, snapshot)
        if check:
            print(f"[{self.match_id}] {check}")
            if self.commentary_log_path is not None:
                with self.commentary_log_path.with_name(
                    f"next-over-checks-{self.match_id}.jsonl"
                ).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"check": check, "expected": self._expected_next}) + "\n")
            self._expected_next = None
        expected_now = _expected_next_over(snapshot, self._announced_bowler)
        if expected_now is not None:
            self._expected_next = expected_now
        print(
            f"[{self.match_id}] {_scorecard_line(snapshot)}"
            f"{_next_over_line(snapshot, self._announced_bowler)}"
        )
        for line in _delivery_event_lines(snapshot, self._seen_delivery_keys):
            print(f"[{self.match_id}] {line}")
        if self.commentary_log_path is not None:
            # Log a delivery only once its over has finished (or the innings
            # has ended): Cricbuzz rewrites an entry's prose after posting it,
            # so logging at first sight stored the short version and lost the
            # edge/mistimed/dropped-catch wording (2026-10-06).
            last_over = max((d.over for d in snapshot.deliveries), default=0)
            settled = tuple(
                d for d in snapshot.deliveries if snapshot.innings_complete or d.over < last_over
            )
            append_commentary_log(
                self.commentary_log_path,
                self.match_id,
                replace(snapshot, deliveries=settled),
                self._logged_delivery_keys,
            )

        snapshot = _with_announced_bowler(snapshot, self._announced_bowler)
        hold = _prediction_hold(snapshot)
        if hold:
            if hold != self._last_hold:
                print(f"[{self.match_id}] ⚠️  PREDICTION HELD (state disagrees with the page's board): {hold}")
            self._last_hold = hold
            return
        self._last_hold = None
        try:
            output = self.pipeline.process(snapshot)
        except VerificationError as error:
            print(f"[{self.match_id}] Verification error: {error}")
            return
        except MatchStateAnomaly as error:
            print(f"[{self.match_id}] MATCH STATE ANOMALY (skipped): {error}")
            return

        if output:
            sent = "Sent" if self.publisher else "Recorded"
            print(
                f"[{self.match_id}] {sent} innings {output['innings']} "
                f"over {output['over']} prediction."
            )
            if self.publisher is None:
                # Shadow mode: show exactly what Telegram would have received,
                # not just that something was recorded (2026-10-06: the
                # one-line "Recorded ... prediction." left nothing to read).
                print(VerifiedLivePredictionPipeline._telegram_text(output))
            innings = output["innings"]
            over = output["over"]
            prior = self._last_over_sent.get(innings)
            if prior is not None and over > prior + 1:
                print(
                    f"[{self.match_id}] OVER GAP DETECTED: innings {innings} "
                    f"jumped from over {prior} to {over} -- over(s) "
                    f"{prior + 1}-{over - 1} never got a prediction."
                )
            self._last_over_sent[innings] = over

        result = _match_result_text(snapshot)
        if result:
            print(f"[{self.match_id}] MATCH COMPLETE: {result}")
            if self.publisher is not None:
                self.publisher.publish(f"result:{self.match_id}", f"🏆 {result}")
            self._finished = True

    def _check_stuck(self, error: Exception | None) -> None:
        """One Telegram alert (deduplicated, auto-resets on recovery) if
        the feed stays erroring continuously for stuck_alert_seconds --
        mirrors run_toi_live.py's own --stuck-alert-seconds exactly. A
        stuck feed and a working-but-quiet one look identical from
        outside without this."""
        now = time.time()
        if self._error_streak_started_at is None:
            self._error_streak_started_at = now
        stuck_seconds = now - self._error_streak_started_at
        if stuck_seconds < self.stuck_alert_seconds or self._stuck_alert_sent:
            return
        self._stuck_alert_sent = True
        stuck_minutes = stuck_seconds / 60.0
        duration_text = f"{stuck_minutes:.1f} min" if stuck_minutes >= 1 else f"{int(stuck_seconds)}s"
        message = (
            f"⚠️ CricketBaba: match {self.match_id}'s Cricbuzz feed has been "
            f"erroring for over {duration_text} ({error}) -- predictions are "
            "paused until it recovers. No action needed; will resume "
            "automatically."
        )
        print(f"[{self.match_id}] {message}")
        if self.publisher is not None:
            try:
                self.publisher.publish(
                    f"stuck_alert:{self.match_id}:{int(self._error_streak_started_at)}",
                    message,
                )
            except Exception:
                pass
