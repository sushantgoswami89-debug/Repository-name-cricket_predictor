"""Unified live-match auto-discovery: Cricbuzz primary, TOI backup.

User decision 2026-09-18, after investigating why TOI's own discovery
silently missed the real 09-15 Afghanistan-vs-India match with no way to
retroactively root-cause it (TOI exposes no queryable historical
archive -- every /results, /recent-results, /series path returns the
same empty SPA shell): stop trying to explain TOI's specific coverage
gaps and remove the human-in-the-loop dependency instead. Both real
misses so far (09-15, 09-17) were only caught because someone noticed
and manually switched to the Cricbuzz fallback script. This runs both
discoveries every scan cycle so that never has to happen again:

- Cricbuzz (app/live/cricbuzz_discovery.py) is scanned first and owns
  any match it finds -- richer discovery data, no per-candidate fetch
  needed, and it's what actually covered the 09-17 match TOI missed.
- TOI (app/live/toi_discovery.py) is scanned second, backup only: a
  match is attached via TOI only if no active session (from either
  source) already covers the same two teams. Once a match is claimed by
  one source, it stays there for the rest of that match's life -- no
  attempt to switch sources mid-match, which would risk two independent
  prediction streams (and two sets of Telegram messages) for one real
  event.

Team-pair matching is a best-effort case-insensitive comparison of each
source's own "Team A vs Team B" label -- both app/live/cricbuzz_
discovery.py's CricbuzzMatchSummary.label and app/live/toi_discovery.py's
ToiMatchSummary.label are built the same way, so this works whenever
both sources spell a team's name identically (true for standard full-
member country names; not guaranteed for every associate nation or
alias).

Usage:
    /usr/bin/python3 run_live_auto.py [--shadow]
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import requests

from app.live.commentary_log import append_commentary_log
from app.live.cricbuzz_discovery import CricbuzzMatchDiscovery
from app.live.cricbuzz_discovery import is_t20_international as cricbuzz_is_t20i
from app.live.cricbuzz_session import CricbuzzMatchSession
from app.live.cricbuzz_session import briefing_text as cricbuzz_briefing_text
from app.live.pipeline import MatchStateAnomaly, VerifiedLivePredictionPipeline
from app.live.telegram import TelegramPublisher
from app.live.toi_discovery import ToiMatchDiscovery
from app.live.toi_discovery import is_t20_international as toi_is_t20i
from app.live.toi_reader import ToiFeedError, ToiLiveReader
from app.live.verification import VerificationError
from run_toi_auto import _briefing_text as toi_briefing_text


def _telegram(project_root: Path) -> TelegramPublisher:
    path = project_root / "backend" / ".telegram.json"
    if not path.exists():
        raise SystemExit("Run setup_telegram.py before starting automatic mode.")
    config = json.loads(path.read_text(encoding="utf-8"))
    return TelegramPublisher(config["bot_token"], config["chat_id"])


def _team_pair(label: str) -> frozenset[str] | None:
    """Order-independent, case-insensitive team-pair key from a "Team A
    vs Team B" label, or None if the label doesn't look like one (the
    fallback-to-bare-id case both discovery modules use when team names
    couldn't be resolved)."""
    parts = [part.strip().casefold() for part in label.split(" vs ")]
    if len(parts) != 2 or not all(parts):
        return None
    return frozenset(parts)


class LiveAutoState:
    """Active sessions from both sources, plus which team-pairs are
    already covered -- extracted from main()'s loop body so the scan/
    attach/detach dedup logic (the actual "Cricbuzz primary, TOI backup"
    behavior) can be exercised directly in tests against fake discovery
    objects, not only by hand against whatever happens to be live."""

    def __init__(
        self,
        project_root: Path,
        publisher: TelegramPublisher | None,
        stuck_alert_seconds: float = 300.0,
    ) -> None:
        self.project_root = project_root
        self.publisher = publisher
        self.stuck_alert_seconds = stuck_alert_seconds
        self.active_cricbuzz: dict[str, CricbuzzMatchSession] = {}
        self.active_toi: dict[
            str, tuple[str, ToiLiveReader, VerifiedLivePredictionPipeline]
        ] = {}
        # Observability, punch-list item 11 -- mirrors CricbuzzMatchSession's
        # own stuck-alert exactly (see its docstring), applied to the TOI
        # backup path here since it has no session class of its own.
        self._toi_error_streak_started_at: dict[str, float] = {}
        self._toi_stuck_alert_sent: dict[str, bool] = {}
        # User-requested 2026-09-18: same per-delivery commentary log as
        # CricbuzzMatchSession (see app/live/commentary_log.py), applied
        # to the TOI backup path here since it has no session class.
        self._toi_logged_delivery_keys: dict[str, set[tuple[int, int, int]]] = {}
        # team-pair -> which source currently owns it, so the backup scan
        # never double-attaches a match the primary already claimed.
        self.covered_pairs: dict[frozenset[str], str] = {}

    def scan(self, cricbuzz_discovery: CricbuzzMatchDiscovery, toi_discovery: ToiMatchDiscovery) -> None:
        # 1. Cricbuzz scan -- primary, always gets first claim.
        try:
            cricbuzz_matches = tuple(
                match
                for match in cricbuzz_discovery.live_matches()
                if cricbuzz_is_t20i(match)
            )
        except requests.exceptions.RequestException as error:
            print(f"Deferred Cricbuzz discovery scan: {error}")
            cricbuzz_matches = ()

        cricbuzz_live_ids = {match.match_id for match in cricbuzz_matches}
        for match in cricbuzz_matches:
            pair = _team_pair(match.label)
            if match.match_id in self.active_cricbuzz:
                continue
            if pair is not None and pair in self.covered_pairs:
                continue
            output_file = self.project_root / "data" / "live" / f"cricbuzz-{match.match_id}.json"
            self.active_cricbuzz[match.match_id] = CricbuzzMatchSession(
                match_id=match.match_id,
                match_url=match.url,
                pipeline=VerifiedLivePredictionPipeline(
                    publisher=self.publisher, output_file=output_file
                ),
                publisher=self.publisher,
                stuck_alert_seconds=self.stuck_alert_seconds,
                commentary_log_path=self.project_root
                / "data"
                / "live"
                / f"commentary-{match.match_id}.jsonl",
            )
            if pair is not None:
                self.covered_pairs[pair] = "cricbuzz"
            print(f"[cricbuzz] Attached: {match.label} ({match.match_desc})")
            briefing = cricbuzz_briefing_text(match.match_id, match.url, match.label)
            if briefing:
                print(briefing)
                if self.publisher is not None:
                    self.publisher.publish(f"briefing:cricbuzz:{match.match_id}", briefing)
        for match_id in set(self.active_cricbuzz) - cricbuzz_live_ids:
            del self.active_cricbuzz[match_id]
            print(f"[cricbuzz] No longer live, detaching: {match_id}")
        # Free any cricbuzz-owned pair Cricbuzz no longer lists *before*
        # the TOI section below checks covered_pairs -- otherwise a pair
        # Cricbuzz just dropped this same cycle would still look "covered"
        # to TOI's backup check below, blocking it from ever picking up a
        # match Cricbuzz stops seeing (confirmed by a real test failure,
        # not just reasoned out: the single end-of-scan rebuild that used
        # to live here ran too late, after TOI's attach loop had already
        # consulted the stale entry).
        self.covered_pairs = {
            pair: source
            for pair, source in self.covered_pairs.items()
            if source != "cricbuzz"
            or any(_team_pair(m.label) == pair for m in cricbuzz_matches)
        }

        # 2. TOI scan -- backup only: skip anything a covered pair
        # already claims (from either source).
        try:
            toi_matches = tuple(
                match for match in toi_discovery.live_summaries() if toi_is_t20i(match)
            )
        except requests.exceptions.RequestException as error:
            print(f"Deferred TOI discovery scan: {error}")
            toi_matches = ()

        toi_live_ids = {match.match_id for match in toi_matches}
        for match in toi_matches:
            if match.match_id in self.active_toi:
                continue
            pair = _team_pair(match.label)
            if pair is not None and pair in self.covered_pairs:
                continue
            reader = ToiLiveReader()
            self.active_toi[match.match_id] = (
                match.url,
                reader,
                VerifiedLivePredictionPipeline(
                    publisher=self.publisher,
                    output_file=self.project_root / "data" / "live" / f"{match.match_id}.json",
                ),
            )
            if pair is not None:
                self.covered_pairs[pair] = "toi"
            print(f"[toi] Attached as backup (Cricbuzz missed this one): {match.label}")
            briefing = toi_briefing_text(reader, match.match_id, match.label)
            if briefing:
                print(briefing)
                if self.publisher is not None:
                    self.publisher.publish(f"briefing:toi:{match.match_id}", briefing)
        for match_id in set(self.active_toi) - toi_live_ids:
            del self.active_toi[match_id]
            print(f"[toi] No longer live, detaching: {match_id}")
        # Symmetric cleanup for TOI-owned pairs -- see the comment on the
        # cricbuzz-side cleanup above for why this can't wait until some
        # single end-of-scan pass.
        self.covered_pairs = {
            pair: source
            for pair, source in self.covered_pairs.items()
            if source != "toi" or any(_team_pair(m.label) == pair for m in toi_matches)
        }

        if not cricbuzz_matches and not toi_matches:
            print("No live T20I on either source right now; continuing to watch.")

    def poll(self) -> None:
        for match_id, session in tuple(self.active_cricbuzz.items()):
            try:
                session.poll_once()
            except Exception as error:  # one bad match must not kill the watcher
                print(f"[cricbuzz] Unexpected error on {match_id}: {type(error).__name__}: {error}")
                continue
            if session.finished:
                del self.active_cricbuzz[match_id]

        for match_id, (url, reader, pipeline) in tuple(self.active_toi.items()):
            try:
                snapshot = reader.fetch(url)
                output = pipeline.process(snapshot)
            except (ToiFeedError, VerificationError, MatchStateAnomaly) as error:
                print(f"[toi] Deferred {match_id}: {error}")
                self._check_toi_stuck(match_id, error)
                continue
            append_commentary_log(
                self.project_root / "data" / "live" / f"commentary-{match_id}.jsonl",
                match_id,
                snapshot,
                self._toi_logged_delivery_keys.setdefault(match_id, set()),
            )
            self._toi_error_streak_started_at.pop(match_id, None)
            self._toi_stuck_alert_sent.pop(match_id, None)
            if output:
                print(
                    f"[toi] {'Sent' if self.publisher else 'Recorded'} {match_id} "
                    f"innings {output['innings']} over {output['over']} prediction."
                )
                if not self.publisher:
                    print(VerifiedLivePredictionPipeline._telegram_text(output))

    def _check_toi_stuck(self, match_id: str, error: Exception) -> None:
        """Same one-alert, auto-resetting behavior as
        CricbuzzMatchSession._check_stuck, for the TOI backup path."""
        now = time.time()
        started_at = self._toi_error_streak_started_at.setdefault(match_id, now)
        stuck_seconds = now - started_at
        if stuck_seconds < self.stuck_alert_seconds or self._toi_stuck_alert_sent.get(match_id):
            return
        self._toi_stuck_alert_sent[match_id] = True
        stuck_minutes = stuck_seconds / 60.0
        duration_text = f"{stuck_minutes:.1f} min" if stuck_minutes >= 1 else f"{int(stuck_seconds)}s"
        message = (
            f"⚠️ CricketBaba: match {match_id}'s TOI feed has been erroring "
            f"for over {duration_text} ({error}) -- predictions are paused "
            "until it recovers. No action needed; will resume automatically."
        )
        print(f"[toi] {message}")
        if self.publisher is not None:
            try:
                self.publisher.publish(f"stuck_alert:toi:{match_id}:{int(started_at)}", message)
            except Exception:
                pass


class _Prefetched:
    """Stands in for a discovery object during LiveAutoState.scan: hands back the
    result (or re-raises the error) a background fetch already produced."""

    def __init__(self, result, error: Exception | None) -> None:
        self._result = result
        self._error = error

    def live_matches(self):
        if self._error is not None:
            raise self._error
        return self._result

    live_summaries = live_matches


class _ScanWorker:
    """Runs the slow discovery HTTP requests (12 s timeouts, two sources) off the
    polling thread. A scan used to run inline, so one stalled request froze every
    match poll -- the likely cause of the missed over-17 prediction on 2026-10-06
    ("a stalled poll spanning two overs"). The attach/detach logic itself still
    runs on the main thread, on the prefetched results."""

    def __init__(self, cricbuzz_discovery, toi_discovery) -> None:
        self._cricbuzz = cricbuzz_discovery
        self._toi = toi_discovery
        self._thread: threading.Thread | None = None
        self._results: tuple[_Prefetched, _Prefetched] | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.busy:
            return
        self._results = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        def _fetch(call):
            try:
                return _Prefetched(tuple(call()), None)
            except requests.exceptions.RequestException as error:
                return _Prefetched((), error)
            except Exception as error:  # a discovery bug must not kill the watcher
                return _Prefetched((), requests.exceptions.RequestException(str(error)))

        cricbuzz = _fetch(self._cricbuzz.live_matches)
        toi = _fetch(self._toi.live_summaries)
        self._results = (cricbuzz, toi)

    def take(self) -> tuple[_Prefetched, _Prefetched] | None:
        if self.busy or self._results is None:
            return None
        results, self._results = self._results, None
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-seconds", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument(
        "--stuck-alert-seconds",
        type=float,
        default=300.0,
        help=(
            "If an active match's feed (either source) stays erroring "
            "continuously for this long, send one Telegram alert instead "
            "of only printing to a console nobody may be watching."
        ),
    )
    parser.add_argument(
        "--shadow", action="store_true", help="Do not publish to Telegram."
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cricbuzz_discovery = CricbuzzMatchDiscovery()
    toi_discovery = ToiMatchDiscovery()
    publisher = None if args.shadow else _telegram(project_root)
    state = LiveAutoState(project_root, publisher, stuck_alert_seconds=args.stuck_alert_seconds)

    last_scan = 0.0
    mode = "shadow (no Telegram)" if args.shadow else "LIVE (publishing to Telegram)"
    print(
        f"CricketBaba is watching Cricbuzz (primary) + TOI (backup) for live "
        f"T20Is -- mode: {mode}. Press Control+C to stop."
    )

    worker = _ScanWorker(cricbuzz_discovery, toi_discovery)
    while True:
        now = time.monotonic()
        if now - last_scan >= max(10.0, args.scan_seconds) and not worker.busy:
            worker.start()
            last_scan = now
        fetched = worker.take()
        if fetched is not None:
            state.scan(fetched[0], fetched[1])

        state.poll()
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCricketBaba stopped.")
