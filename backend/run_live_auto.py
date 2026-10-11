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
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

from app.live.commentary_log import append_commentary_log
from app.live.cricbuzz_discovery import CricbuzzMatchDiscovery
from app.live.cricbuzz_discovery import is_t20_international as cricbuzz_is_t20i
from app.live.cricbuzz_session import CricbuzzMatchSession
from app.live.cricbuzz_session import briefing_text as cricbuzz_briefing_text
from app.live.cricbuzz_session import fact_text as cricbuzz_fact_text
from app.live.odi_watch import OdiBriefingWatcher
from app.live.pipeline import MatchStateAnomaly, VerifiedLivePredictionPipeline
from app.live.telegram import TelegramPublisher
from app.live.toi_discovery import ToiMatchDiscovery
from app.live.toi_discovery import is_t20_international as toi_is_t20i
from app.live.toi_reader import ToiFeedError, ToiLiveReader
from app.live.verification import VerificationError
from run_toi_auto import _briefing_text as toi_briefing_text
from run_toi_auto import _fact_text as toi_fact_text


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


POLL_GAP_ALERT_SECONDS = 120.0


class PollWatchdog:
    """Notices when the driver itself stopped running for a while -- the Mac went to sleep or the process
    was suspended. On 2026-10-09 the first archived page of two matches was ~56 min after the start:
    nothing was polled for the first hour and nobody was told. The loop sleeps 10 s, so a wall-clock
    jump of two minutes or more between two iterations means the machine (or process) was not running."""

    def __init__(self, threshold_seconds: float = POLL_GAP_ALERT_SECONDS, clock=time.time) -> None:
        self.threshold = threshold_seconds
        self._clock = clock
        self._last = clock()

    def tick(self) -> float | None:
        """Call once per loop iteration; returns the gap in seconds when it exceeds the threshold."""
        now = self._clock()
        gap = now - self._last
        self._last = now
        return gap if gap >= self.threshold else None


def keep_awake() -> subprocess.Popen | None:
    """Stop macOS idle-sleeping while this driver runs (`caffeinate -i -s -w <pid>` exits with us).
    Closing the lid on battery still sleeps the Mac. Returns the helper process, or None elsewhere."""
    if sys.platform != "darwin":
        return None
    try:
        return subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return None


BRIEFING_REFRESH_SECONDS = 60.0
FULL_XI = 11  # a line-up counts as full only when all 11 players are confirmed
BRIEFING_REFRESH_MAX_HOURS = 8.0


def briefing_signature(text: str) -> tuple[str, bool]:
    """What matters in a briefing: the Toss line and whether BOTH full line-ups (11 confirmed each) are in.
    The briefing is re-sent only when this changes (toss announced, XIs confirmed),
    not on every wording or number change, so it cannot spam."""
    toss = next((line for line in text.splitlines() if line.startswith("Toss:")), "")
    xi = next((line for line in text.splitlines() if line.startswith("XIs:")), "")
    counts = [int(m.group(1)) for m in re.finditer(r"(\d+) confirmed", xi)]
    full = "FULL LINE-UPS CONFIRMED" in xi or (len(counts) == 2 and all(count >= FULL_XI for count in counts))
    return toss, full


def _briefing_update_header(old: tuple[str, bool], new: tuple[str, bool]) -> str:
    toss_known = lambda sig: bool(sig[0]) and "not yet available" not in sig[0]
    changes = []
    if toss_known(new) and not toss_known(old):
        changes.append("toss now known")
    elif toss_known(new) and new[0] != old[0]:
        changes.append("toss updated")
    if new[1] and not old[1]:
        changes.append("full line-ups confirmed")
    return "🔄 Updated briefing -- " + " and ".join(changes or ["details changed"])


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
        # Briefing refresh (toss / XIs arrive after the first briefing). Keyed
        # "cricbuzz:<id>" / "toi:<id>" -> [last signature, next check time, give-up time].
        self._briefings: dict[str, list] = {}
        self._briefing_results: queue.Queue = queue.Queue()
        # Fact-o-Meter cards are their own messages (never inside the briefing): last text sent per match key.
        self._facts_sent: dict[str, str] = {}
        self._briefing_running: set[str] = set()
        self._toi_started: set[str] = set()
        self.briefing_refresh_seconds = BRIEFING_REFRESH_SECONDS

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
            self._track_briefing(f"cricbuzz:{match.match_id}", briefing, match.label)
            self._emit_facts(f"cricbuzz:{match.match_id}", cricbuzz_fact_text(match.match_id, match.url, match.label))
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
            self._track_briefing(f"toi:{match.match_id}", briefing, match.label)
            self._emit_facts(f"toi:{match.match_id}", toi_fact_text(reader, match.match_id, match.label))
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

    # -- briefing refresh -------------------------------------------------
    def _track_briefing(self, key: str, briefing: str | None, label: str = "") -> None:
        """Start watching a match's briefing for the toss / XIs, unless it is already complete."""
        signature = briefing_signature(briefing) if briefing else ("Toss: not yet available", False)
        done = bool(briefing) and "not yet available" not in signature[0] and signature[1]
        if done:
            self._briefings.pop(key, None)
            return
        self._briefings[key] = [
            signature,
            time.monotonic() + self.briefing_refresh_seconds,
            time.monotonic() + BRIEFING_REFRESH_MAX_HOURS * 3600,
            label,
        ]

    def _emit_facts(self, key: str, text: str | None) -> None:
        """Print (and publish, outside shadow mode) a match's Fact-o-Meter card message when it is new or changed."""
        if not text or self._facts_sent.get(key) == text:
            return
        self._facts_sent[key] = text
        print(text)
        if self.publisher is not None:
            source, _, match_id = key.partition(":")
            tag = hashlib.md5(text.encode()).hexdigest()[:8]
            self.publisher.publish(f"factometer:{source}:{match_id}:{tag}", text)

    def _match_started(self, key: str) -> bool:
        source, _, match_id = key.partition(":")
        if source == "cricbuzz":
            session = self.active_cricbuzz.get(match_id)
            return session is None or bool(session._seen_delivery_keys) or session.finished
        return match_id in self._toi_started or match_id not in self.active_toi

    def _build_briefing(self, key: str) -> None:
        """Runs in a worker thread: one fetch, result queued for the main thread."""
        try:
            source, _, match_id = key.partition(":")
            label = self._briefings.get(key, [None, 0, 0, match_id])[3]
            facts = None
            if source == "cricbuzz":
                session = self.active_cricbuzz.get(match_id)
                text = cricbuzz_briefing_text(match_id, session.match_url, label) if session else None
                facts = cricbuzz_fact_text(match_id, session.match_url, label) if session else None
            else:
                entry = self.active_toi.get(match_id)
                text = toi_briefing_text(entry[1], match_id, label) if entry else None
                facts = toi_fact_text(entry[1], match_id, label) if entry else None
        except Exception:
            text, facts = None, None
        self._briefing_results.put((key, text, facts))

    def refresh_briefings(self) -> None:
        """Call every loop: start due refreshes in the background, then apply finished ones
        (print + publish a new briefing when the toss or the XI status changed)."""
        now = time.monotonic()
        for key, (signature, next_check, give_up, _label) in list(self._briefings.items()):
            if now > give_up or self._match_started(key):
                del self._briefings[key]
            elif now >= next_check and key not in self._briefing_running:
                self._briefing_running.add(key)
                self._briefings[key][1] = now + self.briefing_refresh_seconds
                threading.Thread(target=self._build_briefing, args=(key,), daemon=True).start()
        while True:
            try:
                key, text, facts = self._briefing_results.get_nowait()
            except queue.Empty:
                break
            self._briefing_running.discard(key)
            entry = self._briefings.get(key)
            if not text or entry is None or self._match_started(key):
                continue
            self._emit_facts(key, facts)  # a changed card (toss / XIs in) goes out as its own message
            new_signature = briefing_signature(text)
            if new_signature == entry[0]:
                continue
            message = f"{_briefing_update_header(entry[0], new_signature)}\n{text}"
            entry[0] = new_signature
            print(message)
            if self.publisher is not None:
                tag = hashlib.md5(repr(new_signature).encode()).hexdigest()[:8]
                source, _, match_id = key.partition(":")
                self.publisher.publish(f"briefing:{source}:{match_id}:{tag}", message)
            if "not yet available" not in new_signature[0] and new_signature[1]:
                del self._briefings[key]  # toss known and XIs confirmed: nothing left to wait for

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
            if snapshot.deliveries:
                self._toi_started.add(match_id)
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
    parser.add_argument(
        "--no-odi", action="store_true", help="Do not watch ODIs for pre-match briefings."
    )
    parser.add_argument(
        "--odi-all-teams", action="store_true",
        help="ODI briefings for every national side, not only ICC top-10 vs top-10 (prematch_log.ICC_TOP_10).",
    )
    parser.add_argument(
        "--no-keep-awake", action="store_true", help="Do not stop the Mac from idle-sleeping while running."
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

    awake = None if args.no_keep_awake else keep_awake()
    print("Keep-awake: " + ("on (caffeinate, Mac will not idle-sleep while this runs)" if awake else "off"))
    watchdog = PollWatchdog()
    worker = _ScanWorker(cricbuzz_discovery, toi_discovery)
    odi_watcher = None if args.no_odi else OdiBriefingWatcher(publisher, only_top10=not args.odi_all_teams)
    if odi_watcher is not None:
        print("Also watching men's ODIs for pre-match briefings only (" + ("all national sides" if args.odi_all_teams else "ICC top-10 sides") + "); no in-play ODI predictions.")
    while True:
        gap = watchdog.tick()
        if gap is not None:
            message = (
                f"⚠️ CricketBaba: nothing was polled for {gap / 60:.1f} min (the Mac slept or the process was "
                "suspended). Balls bowled in that gap may be missing from the log; the live state is rebuilt "
                "from the page on the next poll."
            )
            print(message)
            if publisher is not None:
                try:
                    publisher.publish(f"pollgap:{int(time.time())}", message)
                except Exception:
                    pass
        now = time.monotonic()
        if now - last_scan >= max(10.0, args.scan_seconds) and not worker.busy:
            worker.start()
            last_scan = now
        fetched = worker.take()
        if fetched is not None:
            state.scan(fetched[0], fetched[1])

        state.poll()
        state.refresh_briefings()
        if odi_watcher is not None:
            odi_watcher.tick()
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCricketBaba stopped.")
