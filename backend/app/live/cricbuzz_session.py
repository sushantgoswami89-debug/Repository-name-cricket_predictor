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

import time
from dataclasses import dataclass, field

import requests

from app.live.cricbuzz_reader import CricbuzzFeedError, CricbuzzLiveReader, _extract_flight_chunks
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


def _delivery_event_lines(snapshot, seen_keys: set[tuple[int, int, int]]) -> list[str]:
    lines = []
    for delivery in snapshot.deliveries:
        if delivery.key in seen_keys:
            continue
        seen_keys.add(delivery.key)
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
                f"[event] inn{delivery.innings} {delivery.over}.{delivery.ball} "
                f"{'/'.join(tags)} -- {delivery.striker} off {delivery.bowler}"
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

    lines = [f"🏏 {label} -- Match Briefing"]
    lines.append(
        f"Toss: {toss_won_by} won, chose to {toss_decision.lower() or 'unknown'}"
        if toss_won_by
        else "Toss: not yet available"
    )
    lines.append(f"Venue: {venue}" if venue else "Venue: not yet available")
    if team_players:
        for team_name, players in team_players.items():
            confirmed = sum(1 for player in players if player.confirm_xi)
            lines.append(f"{team_name}: {confirmed}/{len(players)} confirmed XI")
    else:
        lines.append("Playing XIs not yet available.")
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
        f"({snapshot.overs} ov, RR {run_rate:.2f}) -- {bowler} to {batter}"
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

    @property
    def finished(self) -> bool:
        return self._finished

    def poll_once(self) -> None:
        """Fetch, guard, and (if a boundary was reached) predict for one
        cycle. All output is printed; nothing is returned -- state (match
        completion, Telegram publication) lives on self/self.pipeline."""
        snapshot, error = _fetch_freshest(self.reader, self.match_url)
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
            return

        self._consecutive_feed_errors = 0

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

        print(f"[{self.match_id}] {_scorecard_line(snapshot)}")
        for line in _delivery_event_lines(snapshot, self._seen_delivery_keys):
            print(f"[{self.match_id}] {line}")

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
