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
    dismissed_batter,
    CricbuzzFeedError,
    CricbuzzInningsNotStarted,
    CricbuzzLiveReader,
    _extract_flight_chunks,
    board_match_complete,
)
from app.live.cricbuzz_scorecard import compare_with_deliveries, fetch_scorecard
from app.live.batter_settled import settled_line
from app.live.bowler_records import bowler_line, overs_to_balls
from app.live.result_facts import result_line
from app.live.venue_form import batter_average_line, batter_ground_line, bowler_average_line, bowler_ground_line
from app.live.prematch_analysis import confirmed_xi, fact_o_meter_message, prematch_header, prematch_text, toss_line, venue_summary, xi_line, xi_names
from app.live.pipeline import (
    MatchStateAnomaly,
    PredictionPublisher,
    VerifiedLivePredictionPipeline,
)
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
    """None while the match is still live. The page's match header wins over our own inference."""
    header = _header_result(snapshot)
    if header is not None:
        return header[0]
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


def _header_result(snapshot) -> tuple[str | None, str | None] | None:
    """(status text, winning team or None) when the page's own match header says the match is over, else None.
    A win names the winner; a tie / no result / abandonment has no winner."""
    board = getattr(snapshot, "board", None) or {}
    if not board_match_complete(board):
        return None
    result = board.get("result") if isinstance(board.get("result"), dict) else {}
    winner = result.get("winningTeam") if result.get("resultType") == "win" else None
    return (board.get("status") or "Match complete"), winner


def _match_winner(snapshot) -> str | None:
    """The winning team once the match is decided, else None (mirrors _match_result_text)."""
    header = _header_result(snapshot)
    if header is not None:
        return header[1]
    if snapshot.innings < 2:
        return None
    if snapshot.target > 0 and snapshot.score >= snapshot.target:
        return snapshot.batting_team
    if snapshot.innings_complete:
        return snapshot.bowling_team
    return None


_RUN_OUT_NAME = re.compile(r"([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*)*)\s+[Rr]un\s+[Oo]ut")


def _dismissed_batter(delivery) -> str:
    """Who actually went out (see cricbuzz_reader.dismissed_batter); uses the stored field when present."""
    return getattr(delivery, "dismissed", "") or dismissed_batter(delivery.wicket_kind, delivery.commentary, delivery.striker)


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
    flickered back to stale values. Keep the board that is furthest along (innings, then balls bowled, then score) and ignore
    older copies. The page's own `last_updated_s` is only the tie-break: on 2026-10-11 a copy showing 123/4 at 13.0 overs came
    back with a NEWER timestamp than the 130/4 at 13.2 board before it, so the timestamp alone let a stale copy win."""
    if not current:
        return previous
    if not previous:
        return current

    def stamp(board: dict) -> tuple:
        try:
            overs = float(board.get("overs") or 0)
        except (TypeError, ValueError):
            overs = 0.0
        whole, _, part = f"{overs:.1f}".partition(".")
        return (board.get("innings") or 0, int(whole) * 6 + int(part or 0), board.get("score") or 0, board.get("last_updated_s") or 0)

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


def _board_announced_bowler(snapshot) -> str:
    """The next over's bowler as the page's mini scorecard shows it at an over break ("" when it does not say).
    At the break the board lists (bowler, other bowler) = (next, just bowled) once the page has moved on, but while it has not it
    simply lists the PREVIOUS over's bowler as `bowler`. So only a name that is neither the bowler who just bowled nor the one
    before him can be taken as the next bowler; a name equal to the previous over's bowler is ambiguous (two bowlers swapping
    overs look the same) and is left to the commentary note."""
    board = getattr(snapshot, "board", None)
    if not board or not snapshot.deliveries or not _at_over_break(snapshot) or not _board_is_current(snapshot, board):
        return ""
    last = snapshot.deliveries[-1]
    previous = next((d.bowler for d in reversed(snapshot.deliveries) if d.innings == last.innings and d.over == last.over - 1), "")
    candidate, other = str(board.get("bowler") or ""), str(board.get("other_bowler") or "")
    if candidate and candidate not in (last.bowler, previous) and other == last.bowler:
        return candidate
    return ""


_NEW_BATTER_NAME = re.compile(r"^\s*(.+?),\s*(?:right|left)[- ]?handed bat", re.IGNORECASE)


def _incoming_batter_name(note_text: str) -> str:
    """"Mayank Yadav, right handed bat, comes to the crease" -> "Mayank Yadav" ("" if the note is not that)."""
    match = _NEW_BATTER_NAME.match(note_text or "")
    return match.group(1).strip() if match else ""


def _expected_next_over(snapshot, announced: tuple[int, str] | None, crease: list[str] | None = None,
                        incoming: tuple[str, ...] = ()) -> dict | None:
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
    # The two batters at the crease: the scorecard page's "not out" pair when we have a fresh one, else the
    # board's pair -- in both cases minus anyone our own deliveries show as dismissed (last night the stale
    # board still listed Ishan Kishan after he was out, and the derived striker missed).
    out_names = {d.dismissed or d.striker for d in snapshot.deliveries if d.wicket_kind and d.innings == last.innings}
    pair = [n for n in (crease or [board.get("striker"), board.get("non_striker")]) if n and n not in out_names]
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
    elif last.wicket_kind and len(pair) == 2 and set(incoming) & set(pair) and len(set(incoming) & set(pair)) == 1:
        # A wicket fell on the last ball and the commentary named the batter walking in. The ends change for the new
        # over, so the SURVIVING batter faces and the new one is at the other end (checked later by the HIT/MISS audit;
        # a run out with a crossing can differ).
        other = next(n for n in pair if n in incoming)
        striker = next(n for n in pair if n != other)
    return {"over": last.over + 1, "bowler": bowler, "striker": striker, "other": other}


def _next_over_line(snapshot, announced: tuple[int, str] | None, crease: list[str] | None = None,
                    incoming: tuple[str, ...] = ()) -> str:
    """At the end of an over Cricbuzz's board still shows the finished over's
    bowler and batting positions until the next over's first ball (user,
    2026-10-06: "still previous bowler and batter until one ball is bowled").
    Show what is expected instead, clearly marked as expected: the bowler
    announced in the commentary (if any since the last ball), and the batters
    at the opposite ends."""
    expected = _expected_next_over(snapshot, announced, crease, incoming)
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


def _bat_figures(stats: dict | None) -> str:
    """"102(45)" -- runs (balls)."""
    return f" {stats['runs']}({stats['balls']})" if stats and "runs" in stats and "balls" in stats else ""


def _bowl_figures(stats: dict | None) -> str:
    """"3-0-36-3" -- overs-maidens-runs-wickets."""
    if stats and all(k in stats for k in ("overs", "maidens", "runs", "wickets")):
        return f" {stats['overs']}-{stats['maidens']}-{stats['runs']}-{stats['wickets']}"
    return ""


def _board_is_current(snapshot, board: dict) -> bool:
    """True only when the page's board is at the SAME point as our rebuilt state (same innings, score, wickets and over.ball).
    A board that is behind (a cached older page) still lists a batter who has since gone out -- 2026-10-11: Tilak Varma was
    shown "at the other end" after his run out at 16.4 because the board was still at 16.3."""
    try:
        if board.get("innings") is not None and board["innings"] != getattr(snapshot, "innings", board["innings"]):
            return False
        if board.get("score") is not None and (board["score"], board.get("wickets")) != (snapshot.score, snapshot.wickets):
            return False
        if board.get("overs") is not None and abs(float(board["overs"]) - float(snapshot.overs)) > 0.001:
            return False
    except (AttributeError, TypeError, ValueError):
        return False
    return True  # fields the board does not carry cannot show it is behind


def _board_players(snapshot) -> str:
    """Starred players from the page's live board with their exact figures:
    bowler* 3-0-36-3 bowling, striker* 38(19) on strike, other 19(7) at the other end.
    Only when the board is current; a stale board is left out (print less rather than a batter who is already out)."""
    board = getattr(snapshot, "board", None)
    if not board or not board.get("striker") or not _board_is_current(snapshot, board):
        return ""
    return (
        f"   [NOW: {board.get('bowler', '?')}*{_bowl_figures(board.get('bowler_stats'))} bowling, "
        f"{board['striker']}*{_bat_figures(board.get('striker_stats'))} on strike, "
        f"{board.get('non_striker') or '?'}{_bat_figures(board.get('non_striker_stats'))} at the other end]"
    )


def _over_break_figures(snapshot, crease: list[str] | None, card_batters: dict, card_bowlers: dict, announced: str = "") -> str:
    """Exact figures for the over break, from the fresh scorecard page (not the board's stale names):
    who just bowled and his figures, the two batters at the crease with their scores, the announced next bowler."""
    last = snapshot.deliveries[-1]
    parts = [f"JUST BOWLED: {last.bowler}{card_bowlers.get(last.bowler, '')}"]
    if crease:
        parts.append("AT THE CREASE: " + ", ".join(f"{name}{card_batters.get(name, '')}" for name in crease))
    if announced:
        parts.append(f"NEXT BOWLER: {announced}{card_bowlers.get(announced, ' (not bowled yet)')}")
    return "   [" + " | ".join(parts) + "]"


def _our_state_at(snapshot, board_overs: float) -> tuple[int, int] | None:
    """(score, wickets) of OUR record at the board's over.ball, or None when we cannot place it. The board is usually a ball or two
    behind the commentary; if our record at THAT ball equals the board, there is nothing wrong -- the board just has not caught up."""
    legal_in_over: dict[int, int] = {}
    state = None
    for delivery in sorted(snapshot.deliveries, key=lambda d: (d.over, d.ball)):
        if getattr(delivery, "is_legal", True):
            legal_in_over[delivery.over] = legal_in_over.get(delivery.over, 0) + 1
        legal = legal_in_over.get(delivery.over, 0)
        position = delivery.over if legal >= 6 else (delivery.over - 1) + legal / 10.0
        if position > board_overs + 0.001:
            break
        state = (delivery.feed_total, delivery.feed_wickets)
    return state


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
    if isinstance(board.get("innings"), int) and board["innings"] < snapshot.innings:
        return None  # the board still shows the previous innings: the page is simply behind our innings-break rollover
    if board.get("innings") not in (None, snapshot.innings):
        problems.append(f"innings {snapshot.innings} vs board {board['innings']}")
    else:
        try:
            behind = float(board.get("overs")) < float(snapshot.overs)
        except (TypeError, ValueError):
            behind = False
        if behind and _our_state_at(snapshot, float(board["overs"])) == (board["score"], board.get("wickets")):
            return None  # the board is simply behind us and agrees with our record at its own ball (the usual 1-ball lag)
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


def _audit_ok(snapshot, summary: dict) -> bool | None:
    """True when our rebuilt over matches Cricbuzz's own summary (runs and delivery count), False when it does not, None when
    we cannot tell (the over was not seen live)."""
    ours = [d for d in snapshot.deliveries if d.innings == summary["innings"] and d.over == summary["over"]]
    if not ours:
        return None
    ok_runs = summary["over_runs"] is None or sum(d.total_runs for d in ours) == int(summary["over_runs"])
    ok_balls = not summary["summary"] or len(ours) == len(summary["summary"].split())
    return bool(ok_runs and ok_balls)


def _board_verdict(snapshot) -> bool | None:
    """The page's board against our state AT THE SAME over.ball: True agree, False disagree, None if the board is not at
    this point (an older cached copy tells us nothing)."""
    board = getattr(snapshot, "board", None)
    if not board or board.get("score") is None or snapshot.innings_complete:
        return None
    if board.get("innings") not in (None, snapshot.innings):
        return None
    try:
        if abs(float(board.get("overs")) - float(snapshot.overs)) > 0.001:
            return None
    except (TypeError, ValueError):
        return None
    return (board["score"], board.get("wickets")) == (snapshot.score, snapshot.wickets)


GATE_SECONDS = 90.0  # how long an over-end forecast waits for the page to confirm our record before it goes out flagged UNCONFIRMED
RESYNC_MAX_PER_OVER = 2  # re-reads of the commentary per over before a contradiction is allowed to hold a prediction


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
                    "ours": {"score": snapshot.score, "wickets": snapshot.wickets, "overs": snapshot.overs,
                         "batting_team": snapshot.batting_team, "bowling_team": snapshot.bowling_team},
                    "board": board,
                },
                default=str,
            )
            + "\n"
        )


_TEAM_CODES = {
    "India": "IND", "England": "ENG", "Australia": "AUS", "Pakistan": "PAK", "South Africa": "SA", "New Zealand": "NZ",
    "West Indies": "WI", "Sri Lanka": "SL", "Afghanistan": "AFG", "Bangladesh": "BAN", "Ireland": "IRE", "Zimbabwe": "ZIM",
}


def _match_tag(batting_team: str, bowling_team: str, fallback: str) -> str:
    """"IND-WI": console prefix that tells simultaneous matches apart (user 2026-10-11: "mixing 2 matches")."""
    teams = sorted(_TEAM_CODES.get(t, t[:3].upper()) for t in (batting_team, bowling_team) if t)
    return "-".join(teams) if len(teams) == 2 else fallback


def _ball_summary(snapshot, delivery) -> str:
    """One short console line per ball: "6.1  1 run", "6.2  FOUR", "6.3  WICKET -- Kusal Mendis (caught)".
    The full commentary and the shot-quality flags stay in the commentary log (user 2026-10-11: summary only)."""
    extras = [label.lower() for key, label in _EXTRA_LABELS.items() if key in delivery.extras]
    if delivery.wicket_kind:
        what = f"WICKET -- {_dismissed_batter(delivery)} ({delivery.wicket_kind})"
        if delivery.total_runs:
            what += f", {delivery.total_runs} run" + ("s" if delivery.total_runs != 1 else "")
    elif extras:
        what = f"{'+'.join(extras)}, {delivery.total_runs} run" + ("s" if delivery.total_runs != 1 else "")
    elif delivery.batter_runs == 6:
        what = "SIX"
    elif delivery.batter_runs == 4:
        what = "FOUR"
    elif delivery.total_runs == 0:
        what = "dot"
    else:
        what = f"{delivery.total_runs} run" + ("s" if delivery.total_runs != 1 else "")
    return f"{_ball_label(snapshot, delivery)}  {what}"


def _delivery_event_lines(snapshot, seen_keys: set[tuple[int, int, int]]) -> list[str]:
    lines = []
    for delivery in snapshot.deliveries:
        if delivery.key in seen_keys:
            continue
        seen_keys.add(delivery.key)
        lines.append(_ball_summary(snapshot, delivery))
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
    lines = [f"🏏 {label}" + (f" -- {city} | {venue_summary(venue_city(venue or ''))}" if city else "")]
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
    if len(teams) == 2:
        analysis = prematch_text(
            teams[0], teams[1], venue or "", toss_won_by, toss_decision, confirmed_xi(team_players),
            log_context={"match_id": f"cricbuzz:{match_id}", "source": "cricbuzz"},
        )
        if analysis:
            lines.append("")
            lines.append(analysis)
    # The whole line-up block (status line, then the names once both XIs are full) goes last.
    lines.append("")
    lines.append(xi_line(team_players))
    lines.extend(xi_names(team_players))
    return "\n".join(lines)


def fact_text(match_id: str, match_url: str, label: str) -> str | None:
    """The Fact-o-Meter card message for a match (team cards, ground record, facts about the confirmed XIs), kept apart
    from briefing_text. None when the page cannot be read; "" when there is nothing to say."""
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
    teams = CricbuzzLiveReader._teams_from_ldjson(response.text)
    if len(teams) != 2:
        return ""
    return fact_o_meter_message(teams[0], teams[1], venue or "", toss_won_by, toss_decision, confirmed_xi(team_players))


def _at_over_break(snapshot) -> bool:
    """True between the last ball of an over and the first ball of the next: the page's board still shows the
    finished over's bowler and batters until a new ball is bowled (user, 2026-10-06 and again 2026-10-10)."""
    if not (snapshot.deliveries and str(snapshot.overs).endswith(".0") and not snapshot.innings_complete):
        return False
    # A wide or no-ball bowled FIRST in the next over leaves the overs reading "5.0" although that over has begun (2026-10-14, 5.1
    # wide): then the last delivery belongs to over 6, not 5, and it is no longer the break.
    try:
        completed = int(str(snapshot.overs).partition(".")[0])
        return getattr(snapshot.deliveries[-1], "over", completed) <= completed
    except ValueError:
        return True


def _score_line(snapshot) -> str:
    """"SL 105/6 (17.0 ov) | CRR 6.18 | need 62 off 18 balls | RRR 20.67" -- the chase part only in the second innings."""
    overs_completed, _, balls_into_over = str(snapshot.overs).partition(".")
    balls = int(overs_completed or 0) * 6 + int(balls_into_over or 0)
    parts = [f"{_TEAM_CODES.get(snapshot.batting_team, snapshot.batting_team)} {snapshot.score}/{snapshot.wickets} ({snapshot.overs} ov)"]
    if balls:
        parts.append(f"CRR {snapshot.score * 6 / balls:.2f}")
    target = getattr(snapshot, "target", 0) or 0
    if snapshot.innings == 2 and target:
        left = (getattr(snapshot, "scheduled_overs", 0) or 20) * 6 - balls
        need = target - snapshot.score
        if need <= 0:
            parts.append("target reached")
        elif left > 0:
            parts.append(f"need {need} off {left} balls | RRR {need * 6 / left:.2f}")
    return " | ".join(parts)


def _board_figures(snapshot) -> tuple[dict, dict]:
    """Batter and bowler figures from the page's mini scorecard, in the same " 31(18)" / " 2-0-19-1" form as the scorecard page's.
    Used when the scorecard page lags (its totals disagree with ours, so its figures are dropped); only when the board is
    current, so a stale board never supplies figures for the wrong over."""
    board = getattr(snapshot, "board", None)
    if not board or not _board_is_current(snapshot, board):
        return {}, {}
    batters, bowlers = {}, {}
    for name_key, stats_key in (("striker", "striker_stats"), ("non_striker", "non_striker_stats")):
        stats = board.get(stats_key) or {}
        if board.get(name_key) and stats.get("balls") is not None:
            batters[board[name_key]] = f" {stats.get('runs', 0)}({stats['balls']})"
    for name_key, stats_key in (("bowler", "bowler_stats"), ("other_bowler", "other_bowler_stats")):
        stats = board.get(stats_key) or {}
        if board.get(name_key) and stats.get("overs") is not None:
            bowlers[board[name_key]] = f" {stats['overs']}-{stats.get('maidens', 0)}-{stats.get('runs', 0)}-{stats.get('wickets', 0)}"
    return batters, bowlers


def _first_innings_at(log_path, over: int) -> tuple[int, int] | None:
    """(score, wickets) the side batting first had at the end of `over`, from the per-poll board log (every poll writes our
    rebuilt state), or None when that over was never polled -- e.g. the driver was started after it."""
    try:
        with log_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    ours = row["ours"]
                    if row["innings"] == 1 and abs(float(ours["overs"]) - over) < 0.001:
                        return int(ours["score"]), int(ours["wickets"])
                except (ValueError, KeyError, TypeError):
                    continue
    except OSError:
        return None
    return None


def _chase_comparison(snapshot, first: tuple[int, int] | None) -> str:
    """"IND after 5 ov: 52/1 | WI 35/1 | behind by 17" -- the other side's score at the same stage of its innings."""
    if snapshot.innings != 2 or first is None:
        return ""
    over = int(str(snapshot.overs).partition(".")[0])
    ahead = snapshot.score - first[0]
    gap = "level" if ahead == 0 else f"{'ahead' if ahead > 0 else 'behind'} by {abs(ahead)}"
    short = lambda team: _TEAM_CODES.get(team, team)  # noqa: E731
    return (f"{short(snapshot.bowling_team)} after {over} ov: {first[0]}/{first[1]} | "
            f"{short(snapshot.batting_team)} {snapshot.score}/{snapshot.wickets} | {gap}")


_REVISION_RISK = re.compile(
    r"saves? (?:a |the )?(?:six|four|boundary)|stepped on|steps on|touch(?:ed|es) the (?:rope|cushion|boundary)|third umpire|tv umpire|"
    r"\bumpires?\b|\breview|referr|checking|\bchecks?\b|replay|\bDRS\b",
    re.I,
)


def _revision_risk(snapshot) -> str:
    """The phrase in the just-finished over's commentary that says a ball may still be changed, or "".
    2026-10-13, Pakistan v Sri Lanka 6.6: "2 runs, Saim Ayub saves a six!" became SIX 70 s later ("he has actually stepped on the
    cushion"), so the over total was 4 low when the forecast was made."""
    if not snapshot.deliveries:
        return ""
    over = getattr(snapshot.deliveries[-1], "over", None)
    for delivery in snapshot.deliveries:
        if over is None or getattr(delivery, "over", None) != over:
            continue
        found = _REVISION_RISK.search(getattr(delivery, "commentary", "") or "")
        if found:
            return f"{delivery.over - 1}.{delivery.ball}: \"{found.group(0).strip()}\""
    return ""


def _over_break_block(snapshot, crease, card_batters: dict, card_bowlers: dict, announced: str, expected: dict | None,
                      first: tuple[int, int] | None = None, confirmed: bool = False) -> str:
    """The over-end summary as a few short lines: score with run rates (and what the chase needs), the bowler who just finished
    with his figures, the batters at the crease, and the next bowler and striker."""
    last = snapshot.deliveries[-1]
    over_number = str(snapshot.overs).partition(".")[0]
    lines = [f"--- end of over {over_number} ---", _score_line(snapshot)]
    comparison = _chase_comparison(snapshot, first)
    if comparison:
        lines.append(comparison)
    # Once Cricbuzz's own over summary has agreed with our over (runs and deliveries) nothing in it can still change: a review that
    # was upheld or overturned is already in the total (2026-10-20: "2.1: review" was shown under an over the audit had called OK).
    risk = "" if confirmed else _revision_risk(snapshot)
    if risk:
        lines.append(f"⚠️  this over's total may still change -- {risk}")
    lines.append(f"Bowled: {last.bowler}{card_bowlers.get(last.bowler, '')}")
    striker = expected["striker"] if expected is not None and not snapshot.innings_complete else ""
    if crease:
        def batter(name: str) -> str:
            """"Name 9*(9)" for the batter on strike (* marks him), "Name 8(7)" for the other one."""
            figures = card_batters.get(name, "").replace("*", "")
            if name != striker:
                return f"{name}{figures}"
            return f"{name}{figures.replace('(', '*(', 1)}" if "(" in figures else f"{name}*{figures}"

        lines.append("At the crease: " + ", ".join(batter(name) for name in crease))
    if expected is not None and not snapshot.innings_complete:
        bowler = expected["bowler"]
        if bowler:
            lines.append(f"Next: {bowler}*{card_bowlers.get(bowler, ' (not bowled yet)')}")   # * marks the bowler bowling
        else:
            # The mini scorecard was checked and still lists the previous over's bowler (or nobody new): say so rather than guess.
            board = getattr(snapshot, "board", None) or {}
            shown = board.get("bowler") if board and _board_is_current(snapshot, board) else ""
            lines.append("Next: not announced yet" + (f" (mini scorecard still shows {shown})" if shown and shown != last.bowler else ""))
    return "\n".join(lines)


def _scorecard_line(snapshot) -> str:
    deliveries = snapshot.deliveries
    latest = deliveries[-1] if deliveries else None
    overs_completed, _, balls_into_over = str(snapshot.overs).partition(".")
    legal_balls = int(overs_completed or 0) * 6 + int(balls_into_over or 0)
    run_rate = (snapshot.score * 6 / legal_balls) if legal_balls else 0.0
    batter = latest.striker if latest else "?"
    bowler = latest.bowler if latest else "?"
    tail = (
        f"over complete (last ball: {bowler} to {batter})"  # no board names: they are the finished over's
        if _at_over_break(snapshot)
        else f"last ball: {bowler} to {batter}" + _board_players(snapshot)
    )
    return (
        f"[scorecard] {snapshot.batting_team} {snapshot.score}/{snapshot.wickets} "
        f"({snapshot.overs} ov, RR {run_rate:.2f}) -- {tail}"
    )


@dataclass
class CricbuzzMatchSession:
    """One match's full polling state -- fetch, guard, predict, print."""

    match_id: str
    match_url: str
    pipeline: VerifiedLivePredictionPipeline
    publisher: PredictionPublisher | None = None
    # Where Fact-o-Meter cards go besides the console (user 2026-10-11: send them to Telegram too). Separate from `publisher`
    # so predictions can stay in shadow mode while the cards are sent.
    facts_publisher: PredictionPublisher | None = None
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
    _last_summary: str = ""
    _tag: str = ""
    tag_lines: bool = True
    _last_board_mismatch: str | None = None
    _newest_board_seen: dict | None = None
    _announced_bowler: tuple[int, str] | None = None
    _last_hold: str | None = None
    _expected_next: dict | None = None
    _carded: set = field(default_factory=set)
    _captured: bool = False
    _crease: list[str] | None = None
    # Sync gate (user 2026-10-11: "first match commentary and our recording in perfect sync, then only go for prediction"):
    # per (innings, completed over) which independent sources have confirmed / contradicted our record.
    _confirm: dict = field(default_factory=dict)
    _sync_wait_since: dict = field(default_factory=dict)
    _resyncs: dict = field(default_factory=dict)
    _previous_reader: object = None  # the reader replaced by a re-read, kept until the fresh one has produced a snapshot
    _last_gate: str | None = None
    forecast_enabled: bool = True  # False: record the match, make no forecasts (women's matches: the models are trained on men's cricket)
    forecast_label: str = ""  # printed above every forecast when set (ODIs: "EXPERIMENTAL: the T20 models applied to an ODI ...")
    _gate_started: dict = field(default_factory=dict)
    _gate_notice: str | None = None
    _card_not_out: list[str] = field(default_factory=list)  # the latest scorecard page's not-out names (it may lag)
    _incoming: dict = field(default_factory=dict)  # innings -> batters the commentary said "comes to the crease"
    _card_batters: dict = field(default_factory=dict)
    _card_bowlers: dict = field(default_factory=dict)
    _card_nums: dict = field(default_factory=dict)  # batter -> (runs, balls) from the fresh scorecard
    _card_bowl_nums: dict = field(default_factory=dict)  # bowler -> (legal balls, wickets) from the fresh scorecard
    _settled_shown: set = field(default_factory=set)  # (innings, batter) already given the "settled" stat
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

    def _sync_status(self, snapshot) -> tuple[str, str]:
        """("ok" | "unconfirmed" | "contradiction", reason) for the prediction pending after completed over C.
        ok: an independent source AT THE SAME POINT confirms our record (the page's mini scorecard / board, Cricbuzz's own
        over summary, or the scorecard page's total) and none contradicts it; contradiction: one of them disagrees;
        unconfirmed: none is at this point yet. Nothing waits here: poll_once answers either of the last two by RE-READING
        the commentary (a second), see _resync."""
        if not snapshot.deliveries or snapshot.innings_complete:
            return "ok", ""
        completed = int(float(snapshot.overs))
        if completed < 1 or self._last_over_sent.get(snapshot.innings, 0) >= completed + 1:
            return "ok", ""  # nothing pending: the next over's prediction was already made
        key = (snapshot.innings, completed)
        info = self._confirm.setdefault(key, {})
        board = _board_verdict(snapshot)
        if board is not None:
            info["board"] = board
        contradicted = [name for name, ok in info.items() if ok is False]
        if contradicted:
            return "contradiction", (
                f"over {completed}: {', '.join(contradicted)} disagree with our record at {snapshot.overs} ov "
                f"({snapshot.score}/{snapshot.wickets})"
            )
        if any(ok is True for ok in info.values()):
            return "ok", ""
        now = time.monotonic()
        if now - self._sync_wait_since.get((key, "card"), 0.0) >= 12.0:  # the scorecard page is a slow source: one look now and then
            self._sync_wait_since[(key, "card")] = now
            try:
                card = fetch_scorecard(self.match_id)
                current = next((i for i in (card or {}).get("innings", []) if i.get("innings") == snapshot.innings), None)
                if current and (current.get("runs"), current.get("wickets")) == (snapshot.score, snapshot.wickets):
                    info["card"] = True
                    return "ok", ""
            except Exception:
                pass
        return "unconfirmed", f"over {completed}: no source at this point yet to confirm {snapshot.score}/{snapshot.wickets} at {snapshot.overs} ov"

    def _hold_for_confirmation(self, snapshot, why: str) -> bool:
        """True while an over-end forecast should wait because no source confirms our record yet (user 2026-10-13: the forecast must
        not run ahead of correctly captured data; "gate, then publish flagged"). Waits at most GATE_SECONDS from the first poll of
        this over end, and only while that over end is still the latest thing that happened -- once the next over's first ball is in,
        waiting is over (the forecast then goes out marked LATE)."""
        if not _at_over_break(snapshot):
            return False
        key = (snapshot.innings, str(snapshot.overs))
        started = self._gate_started.setdefault(key, time.monotonic())
        waited = time.monotonic() - started
        if waited >= GATE_SECONDS:
            return False
        if self._gate_notice != why:
            print(f"{self.prefix}[gate] holding the forecast until the page confirms {snapshot.score}/{snapshot.wickets} at "
                  f"{snapshot.overs} ov (up to {GATE_SECONDS:.0f} s): {why}")
            self._gate_notice = why
        return True

    def _resync(self, snapshot, reason: str, rebuild: bool = True) -> bool:
        """Re-read the commentary instead of blocking (user 2026-10-11: "dont block, read commentary ... it will hardly take 1
        second to reread previous commentary from cricbuzz"): a fresh reader rebuilds the innings from everything the page still
        shows, with the page's board as the baseline, and poll_once runs again at once. At most RESYNC_MAX_PER_OVER times per
        over; False when that is used up."""
        key = (snapshot.innings, int(float(snapshot.overs)), rebuild)
        used = self._resyncs.get(key, 0)
        if used >= RESYNC_MAX_PER_OVER:
            return False
        self._resyncs[key] = used + 1
        what = "re-reading the commentary" if rebuild else "looking at a fresher copy of the page"
        print(f"{self.prefix}[sync] {what} ({used + 1}/{RESYNC_MAX_PER_OVER}): {reason}")
        if rebuild:  # a fresh reader rebuilds the innings from what the page shows now, the board as the baseline
            self._previous_reader = self.reader
            self.reader = CricbuzzLiveReader()
            self._best_progress = None
            self._confirm.pop((key[0], key[1]), None)
        return True

    def _log_announcement(self, snapshot, source: str, name: str) -> None:
        """Append when each source first named the next bowler (data/live/announce-sources-<id>.jsonl) for a later comparison."""
        if self.commentary_log_path is None or not snapshot.deliveries:
            return
        try:
            path = self.commentary_log_path.with_name(f"announce-sources-{self.match_id}.jsonl")
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"t": time.time(), "innings": snapshot.innings, "after_over": snapshot.deliveries[-1].over,
                                         "source": source, "bowler": name}) + "\n")
        except OSError:
            pass

    def _derived_crease(self, snapshot) -> list[str] | None:
        """The two batters at the crease when the scorecard page lags: the page's last not-out names (stale or not), the
        board's pair and every batter the commentary said "comes to the crease", minus everyone our own deliveries show
        as dismissed. Only when that leaves exactly two."""
        if not snapshot.deliveries:
            return None
        board = getattr(snapshot, "board", None) or {}
        out = {d.dismissed or d.striker for d in snapshot.deliveries if d.wicket_kind and d.innings == snapshot.innings}
        names: list[str] = []
        for name in [*self._card_not_out, board.get("striker"), board.get("non_striker"), *self._incoming.get(snapshot.innings, [])]:
            if name and name not in out and name not in names:
                names.append(name)
        return names if len(names) == 2 else None

    def _send_card(self, stat: str, key: tuple) -> None:
        """Print a live Fact-o-Meter card and send it to Telegram when a facts publisher is set (never raises)."""
        print(f"{self.prefix}" + stat.replace("\n", "\n    "))
        if self.facts_publisher is None:
            return
        try:
            self.facts_publisher.publish(f"factometer:live:{self.match_id}:" + ":".join(str(part) for part in key), stat)
        except Exception as error:  # a failed send must never disturb the feed
            print(f"{self.prefix}[stat] Telegram send failed: {error!r}")

    @property
    def prefix(self) -> str:
        """"[IND-WI] " before every console line -- empty when the match has its own console (nothing to tell apart)."""
        return f"[{self.tag}] " if self.tag_lines else ""

    @property
    def tag(self) -> str:
        """Console prefix: the two teams once a snapshot has named them, else the match id."""
        return self._tag or self.match_id

    def poll_once(self) -> None:
        """Fetch, guard, and (if a boundary was reached) predict for one
        cycle. All output is printed; nothing is returned -- state (match
        completion, Telegram publication) lives on self/self.pipeline."""
        snapshot, error = _fetch_freshest(self.reader, self.match_url)
        if self._previous_reader is not None:
            if snapshot is None:  # the fresh reader could not rebuild (e.g. no baseline yet): go back to the record we had
                print(f"{self.prefix}[sync] re-read produced no snapshot ({error}); keeping the previous record")
                self.reader, self._previous_reader = self._previous_reader, None
                return
            self._previous_reader = None
        if snapshot is None and isinstance(error, CricbuzzInningsNotStarted):
            if not self._waiting_notice_printed:
                print(f"{self.prefix}Waiting: {error}")
                self._waiting_notice_printed = True
            self._consecutive_feed_errors = 0
            self._error_streak_started_at = None
            return
        if snapshot is None:
            label = "Deferred" if isinstance(error, CricbuzzFeedError) else "Deferred (network)"
            print(f"{self.prefix}{label}: {error}")
            self._consecutive_feed_errors += 1
            if self._consecutive_feed_errors >= RESEED_AFTER_CONSECUTIVE_ERRORS:
                print(
                    f"{self.prefix}Reseeding Cricbuzz reader after "
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
            return  # an older cached copy of the page; nothing to show
        self._best_progress = (
            progress if self._best_progress is None else max(self._best_progress, progress)
        )

        self._tag = _match_tag(snapshot.batting_team, snapshot.bowling_team, self.match_id)
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
            print(f"{self.prefix}⚠️  BOARD MISMATCH: ours {mismatch}")
        self._last_board_mismatch = mismatch
        drain_repairs = getattr(self.reader, "drain_repairs", None)
        for repair in (drain_repairs() if drain_repairs else []):
            print(f"{self.prefix}[repair] {repair}")
            if self.commentary_log_path is not None:
                repairs_path = self.commentary_log_path.with_name(f"repairs-{self.match_id}.jsonl")
                with repairs_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "repair": repair}) + "\n")
        drain_summaries = getattr(self.reader, "drain_over_summaries", None)
        for summary in (drain_summaries() if drain_summaries else []):
            print(f"{self.prefix}{_audit_over(snapshot, summary)}")
            verdict = _audit_ok(snapshot, summary)
            if verdict is not None:
                self._confirm.setdefault((summary["innings"], summary["over"]), {})["summary"] = verdict
            if self.commentary_log_path is not None:
                archive = self.commentary_log_path.with_name(f"over-summaries-{self.match_id}.jsonl")
                with archive.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(summary) + "\n")
        drain_notes = getattr(self.reader, "drain_notes", None)
        for note_ts, category, note_text in (drain_notes() if drain_notes else []):
            # not printed (user 2026-10-11: details not to be shown); kept in commentary-notes-<id>.jsonl below
            if category == "NEW BATTER":
                incoming_name = _incoming_batter_name(note_text)
                if incoming_name and incoming_name not in self._incoming.setdefault(snapshot.innings, []):
                    self._incoming[snapshot.innings].append(incoming_name)
            if category == "BOWLING CHANGE":
                announced_name = _announced_bowler_name(note_text)
                if announced_name:
                    self._announced_bowler = (note_ts, announced_name)
                    self._log_announcement(snapshot, "commentary note", announced_name)
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
        # The mini scorecard can name the next bowler too (user 2026-10-11: go back to it for the bowler change): take it when the
        # commentary note has not said so yet, and log which source spoke first so the two can be compared after the match.
        from_board = _board_announced_bowler(snapshot)
        if from_board and not (self._announced_bowler and self._announced_bowler[0] > snapshot.deliveries[-1].timestamp_ms):
            self._announced_bowler = (snapshot.deliveries[-1].timestamp_ms + 1, from_board)
            self._log_announcement(snapshot, "mini scorecard", from_board)
        check = _check_next_over(self._expected_next, snapshot)
        if check:
            # the HIT/MISS audit is kept in next-over-checks-<id>.jsonl, not printed (user 2026-10-11)
            if self.commentary_log_path is not None:
                with self.commentary_log_path.with_name(
                    f"next-over-checks-{self.match_id}.jsonl"
                ).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"check": check, "expected": self._expected_next}) + "\n")
            self._expected_next = None
        self._card_check(snapshot)  # at an over end this also refreshes the not-out pair used just below
        crease = (self._crease or self._derived_crease(snapshot)) if _at_over_break(snapshot) else None
        incoming = tuple(self._incoming.get(snapshot.innings, []))
        expected_now = _expected_next_over(snapshot, self._announced_bowler, crease, incoming)
        if expected_now is not None:
            self._expected_next = expected_now
        figures = ""
        if _at_over_break(snapshot):
            announced = self._announced_bowler[1] if self._announced_bowler and self._announced_bowler[0] > snapshot.deliveries[-1].timestamp_ms else ""
            figures = _over_break_figures(snapshot, crease, self._card_batters, self._card_bowlers, announced)
        for line in _delivery_event_lines(snapshot, self._seen_delivery_keys):
            print(f"{self.prefix}{line}")
        # Summary only (user 2026-10-11): one score line per over break / innings end, not one per poll.
        if _at_over_break(snapshot) or snapshot.innings_complete:
            announced_now = (self._announced_bowler[1] if self._announced_bowler
                             and self._announced_bowler[0] > snapshot.deliveries[-1].timestamp_ms else "")
            board_batters, board_bowlers = _board_figures(snapshot)
            first = None
            if snapshot.innings == 2 and self.commentary_log_path is not None:
                over_now = int(str(snapshot.overs).partition(".")[0])
                first = _first_innings_at(self.commentary_log_path.with_name(f"board-{self.match_id}.jsonl"), over_now)
            summary = _over_break_block(snapshot, crease, {**board_batters, **self._card_batters},
                                        {**board_bowlers, **self._card_bowlers}, announced_now, expected_now, first,
                                        confirmed=self._confirm.get((snapshot.innings, int(str(snapshot.overs).partition(".")[0])), {}).get("summary") is True)
            if summary != self._last_summary:
                self._last_summary = summary
                print("\n".join(f"{self.prefix}{line}" for line in summary.splitlines()))
        for name in (crease or []):
            runs_balls = self._card_nums.get(name)
            key = (snapshot.innings, name)
            if runs_balls is None or key in self._settled_shown:
                continue
            stat = ""
            try:
                stat = settled_line(name, snapshot.batting_team, runs_balls[1], runs_balls[0], opponent=snapshot.bowling_team)
            except Exception as error:  # a side stat must never disturb the feed
                print(f"{self.prefix}[stat] unavailable: {error!r}")
            if stat:
                self._settled_shown.add(key)
                self._send_card(stat, key)
        for name in (crease or []):
            runs_balls = self._card_nums.get(name)
            key = (snapshot.innings, "result", name)
            if runs_balls is None or key in self._settled_shown:
                continue
            stat = ""
            try:
                from app.live.prematch_xi import resolve_player

                stat = result_line(name, resolve_player(name, snapshot.batting_team), snapshot.batting_team, "bat",
                                   runs_balls[0], runs_balls[1], snapshot.innings, opponent=snapshot.bowling_team)
            except Exception as error:
                print(f"{self.prefix}[stat] unavailable: {error!r}")
            if stat:
                self._settled_shown.add(key)
                self._send_card(stat, key)
        for name in (crease or []):
            runs_balls = self._card_nums.get(name)
            key = (snapshot.innings, "ground", name)
            if runs_balls is None or key in self._settled_shown:
                continue
            stat = ""
            try:
                from app.live.prematch_model import venue_city
                from app.live.prematch_xi import resolve_player

                resolved = resolve_player(name, snapshot.batting_team)
                city = venue_city(getattr(snapshot, "venue_name", "") or "")
                stat = (batter_ground_line(name, resolved, city, snapshot.innings, runs_balls[0], runs_balls[1])
                        or batter_average_line(name, resolved, city, runs_balls[0], runs_balls[1]))  # 100% record first
            except Exception as error:  # a side stat must never disturb the feed
                print(f"{self.prefix}[stat] unavailable: {error!r}")
            if stat:
                self._settled_shown.add(key)
                self._send_card(stat, key)
        if _at_over_break(snapshot):
            for name in dict.fromkeys([snapshot.deliveries[-1].bowler, announced]):
                figures_now = self._card_bowl_nums.get(name) if name else None
                if figures_now is None:
                    continue
                for kind_key in ("bowler", "result", "ground"):
                    key = (snapshot.innings, kind_key, name)
                    if key in self._settled_shown:
                        continue
                    stat = ""
                    try:
                        if kind_key == "bowler":
                            stat = bowler_line(name, snapshot.bowling_team, figures_now[0], figures_now[1], opponent=snapshot.batting_team)
                        elif kind_key == "ground":
                            from app.live.prematch_model import venue_city
                            from app.live.prematch_xi import resolve_player

                            resolved = resolve_player(name, snapshot.bowling_team)
                            city = venue_city(getattr(snapshot, "venue_name", "") or "")
                            stat = (bowler_ground_line(name, resolved, city, snapshot.innings, figures_now[1], figures_now[0])
                                    or bowler_average_line(name, resolved, city, figures_now[1], figures_now[0]))
                        else:
                            from app.live.prematch_xi import resolve_player

                            stat = result_line(name, resolve_player(name, snapshot.bowling_team), snapshot.bowling_team,
                                               "bowl", figures_now[1], figures_now[0], snapshot.innings,
                                               opponent=snapshot.batting_team)
                    except Exception as error:  # a side stat must never disturb the feed
                        print(f"{self.prefix}[stat] unavailable: {error!r}")
                    if stat:
                        self._settled_shown.add(key)
                        self._send_card(stat, key)
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
        if not self.forecast_enabled:
            status, why, hold = "ok", "", None   # record only: nothing to confirm before a forecast that is not made
        else:
            status, why = self._sync_status(snapshot)
            hold = _prediction_hold(snapshot)  # the page's board disagreeing at the same over.ball
        if hold and status == "ok":
            status, why = "contradiction", hold
        if status != "ok":
            # Do not block: re-read the commentary (about a second) and look again straight away.
            if self._resync(snapshot, why, rebuild=(status == "contradiction")):
                self.poll_once()
                return
            if status == "contradiction":
                if why != self._last_hold:
                    print(f"{self.prefix}⚠️  PREDICTION HELD (still disagrees after re-reading the commentary): {why}")
                self._last_hold = why
                return
            if self._hold_for_confirmation(snapshot, why):
                return
            if why != self._last_gate:  # unconfirmed after the re-reads and the wait: go on, say so
                print(f"{self.prefix}[sync] predicting UNCONFIRMED (the page has no source at this point yet): {why}")
            self._last_gate = why
        else:
            self._last_gate = None
        self._last_hold = None
        try:
            output = self.pipeline.process(snapshot) if self.forecast_enabled else None
        except VerificationError as error:
            print(f"{self.prefix}Verification error: {error}")
            return
        except MatchStateAnomaly as error:
            print(f"{self.prefix}MATCH STATE ANOMALY (skipped): {error}")
            return

        if output:
            sent = "Sent" if self.publisher else "Recorded"
            if self.forecast_label:
                print(f"{self.prefix}⚠️  {self.forecast_label}")
            started = [d for d in snapshot.deliveries if d.innings == output["innings"] and d.over == output["over"]]
            if started:  # 2026-10-13: over 4's forecast came out at 3.5 after a recovery; it still uses the pre-over state but is no use live
                print(f"{self.prefix}⚠️  LATE: over {output['over']} was already under way ({snapshot.overs} ov, "
                      f"{len(started)} ball(s) bowled) when this forecast came out -- do not use it live")
            print(
                f"{self.prefix}{sent} innings {output['innings']} "
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
                    f"{self.prefix}OVER GAP DETECTED: innings {innings} "
                    f"jumped from over {prior} to {over} -- over(s) "
                    f"{prior + 1}-{over - 1} never got a prediction."
                )
            self._last_over_sent[innings] = over

        result = _match_result_text(snapshot)
        if result:
            from app.live.prematch_log import record_result

            record_result(f"cricbuzz:{self.match_id}", _match_winner(snapshot), result)  # lets the pre-match log be scored
            self._capture_all_pages()
            print(f"{self.prefix}MATCH COMPLETE: {result}")
            if self.publisher is not None:
                self.publisher.publish(f"result:{self.match_id}", f"🏆 {result}")
            self._finished = True

    def _capture_all_pages(self) -> None:
        """Once, at the end of the match: save every Cricbuzz page for it and Cricbuzz's own win-probability
        curve (see cricbuzz_capture.py). Background thread, failures ignored."""
        if getattr(self, "_captured", False):
            return
        self._captured = True
        try:
            import threading

            from app.live.cricbuzz_capture import capture_match

            threading.Thread(target=capture_match, args=(self.match_id,), daemon=True).start()
            print(f"{self.prefix}capturing all Cricbuzz pages + win curve in the background")
        except Exception:
            pass

    def _log_scorecard(self, card: dict) -> None:
        """Append Cricbuzz's scorecard (every batter and bowler figure) to data/live/scorecard-<id>.jsonl at each over end.
        audit_vs_scorecard.py compares the DIFFERENCE between two of these with the deliveries we recorded in between, which checks every
        ball's runs, striker and bowler against the scorecard even when we joined the match late (2026-10-20 deadline)."""
        if self.commentary_log_path is None:
            return
        try:
            row = {
                "fetched_at": time.time(),
                "header": card.get("header"),
                "innings": [
                    {
                        "innings": inn.get("innings"), "team": inn.get("team"), "runs": inn.get("runs"), "wickets": inn.get("wickets"),
                        "overs": inn.get("overs"),
                        "batting": [{k: b.get(k) for k in ("name", "runs", "balls", "fours", "sixes", "out")} for b in inn.get("batting", [])],
                        "bowling": [{k: b.get(k) for k in ("name", "overs", "maidens", "runs", "wickets", "wides", "no_balls")}
                                    for b in inn.get("bowling", [])],
                        "extras": inn.get("extras"),
                    }
                    for inn in card.get("innings", [])
                ],
            }
            with self.commentary_log_path.with_name(f"scorecard-{self.match_id}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        except (OSError, TypeError, ValueError):
            pass

    def _card_check(self, snapshot) -> None:
        """At each over end (and at the finish) compare our rebuilt innings with Cricbuzz's own scorecard
        page: totals always, per-batter runs and fall of wickets when we saw the innings from ball 1.
        One extra request per over; any failure is silent (it is a cross-check, never required)."""
        if not snapshot.deliveries:
            return
        at_over_end = str(snapshot.overs).endswith(".0")
        finished = bool(getattr(snapshot, "match_complete", False))
        key = (snapshot.innings, str(snapshot.overs), finished)
        if not (at_over_end or finished) or key in self._carded:
            return
        self._carded.add(key)
        self._crease = None  # never carry the previous over's pair into this one if the fetch fails
        self._card_not_out = []  # nor the previous innings' / over's not-out names
        card = fetch_scorecard(self.match_id)
        if card is None:
            return
        self._log_scorecard(card)
        problems = []
        for innings in sorted({d.innings for d in snapshot.deliveries}):
            problems += compare_with_deliveries(card, snapshot.deliveries, innings)
        # Who is at the crease right now, from the card, but only if the card is as current as we are.
        current = next((i for i in card["innings"] if i.get("innings") == snapshot.innings), None)
        if current:
            self._card_not_out = [b["name"] for b in current["batting"] if str(b.get("out", "")).lower() == "not out"]
        self._card_batters, self._card_bowlers, self._card_nums, self._card_bowl_nums = {}, {}, {}, {}
        if current and (current.get("runs"), current.get("wickets")) == (snapshot.score, snapshot.wickets):
            self._confirm.setdefault((snapshot.innings, snapshot.deliveries[-1].over), {})["card"] = True
            pair = [b["name"] for b in current["batting"] if str(b.get("out", "")).lower() == "not out"]
            self._crease = pair if len(pair) == 2 else None
            self._card_batters = {b["name"]: f" {b['runs']}*({b['balls']})" for b in current["batting"] if b.get("balls") is not None}
            self._card_nums = {b["name"]: (b["runs"], b["balls"]) for b in current["batting"] if b.get("balls") is not None}
            bowlers = next((i for i in card["innings"] if i.get("innings") == snapshot.innings), current)["bowling"]
            self._card_bowlers = {
                b["name"]: f" {b['overs']}-{b['maidens']}-{b['runs']}-{b['wickets']}" for b in bowlers if b.get("overs") is not None
            }
            self._card_bowl_nums = {
                b["name"]: (overs_to_balls(b["overs"]), b["wickets"])
                for b in bowlers if b.get("overs") is not None and overs_to_balls(b["overs"]) is not None and b.get("wickets") is not None
            }
        verdict = "OK" if not problems else "MISMATCH: " + "; ".join(problems[:4])
        if problems:  # an OK check is not printed (user 2026-10-11); every verdict goes to the card-checks log
            print(f"{self.prefix}[card] {snapshot.overs} ov: {verdict}")
        if self.commentary_log_path is not None:
            try:
                with self.commentary_log_path.with_name(f"card-checks-{self.match_id}.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"innings": snapshot.innings, "overs": str(snapshot.overs), "problems": problems,
                                             "header": card.get("header")}) + "\n")
            except OSError:
                pass

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
        print(f"{self.prefix}{message}")
        if self.publisher is not None:
            try:
                self.publisher.publish(
                    f"stuck_alert:{self.match_id}:{int(self._error_streak_started_at)}",
                    message,
                )
            except Exception:
                pass
