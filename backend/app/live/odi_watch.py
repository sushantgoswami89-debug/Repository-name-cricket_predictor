"""ODI pre-match briefings for the live driver (run_live_auto.py).

The driver's T20 machinery (live sessions, per-over predictions) does not apply to ODIs, so this is a small separate
watcher: it looks at the same Cricbuzz match list, and for each men's international ODI whose teams are both in
the ICC top 10 (prematch_log.ICC_TOP_10; `only_top10=False` widens it to every national side) it
  * prints (and, when not in shadow mode, publishes) the ODI briefing once the match shows as Preview,
  * prints it again when the toss is known (then stops: the ODI model has no toss / XI block yet),
  * records the result when Cricbuzz marks the match Complete, so score_prematch_log.py can score it.
It never makes in-play predictions for ODIs. All network work runs in one background thread; tick() never blocks.
"""

from __future__ import annotations

import queue
import re
import threading
import time

from app.live import prematch_log

PREVIEW_STATES = ("preview", "toss")


def is_men_international_odi(match, only_top10: bool = True) -> bool:
    """Men's national-team ODI from the Cricbuzz list (no women's, A-team, or domestic sides)."""
    from app.live.prematch_odi import current_state

    if str(match.match_format).upper() != "ODI" or str(match.match_type).lower() == "domestic":
        return False
    label = match.label
    if "women" in label.lower() or re.search(r"\bA\b", label):
        return False
    teams = [t.strip() for t in label.split(" vs ")]
    if len(teams) != 2:
        return False
    if only_top10:
        return all(t in prematch_log.ICC_TOP_10 for t in teams)
    national = set(current_state().ratings) | prematch_log.ICC_TOP_10
    return all(t in national for t in teams)


def winner_from_status(status: str, teams: list[str]) -> str | None:
    """"Pakistan won by 5 wkts" -> "Pakistan"; ties / no result / unreadable -> None."""
    text = (status or "").strip()
    for team in teams:
        if text.lower().startswith(team.lower() + " won"):
            return team
    return None


def _is_decided_text(status: str) -> bool:
    return " won " in f" {(status or '').lower()} " or any(
        word in (status or "").lower() for word in ("no result", "tied", "tie", "abandoned", "drawn")
    )


def fetch_toss(match) -> tuple[str, str]:
    """(toss winner, "bat"/"bowl"-style decision) from the match page, or ("", "") if not known / unreachable."""
    import requests

    from app.live.cricbuzz_reader import CricbuzzLiveReader, _extract_flight_chunks

    try:
        response = requests.get(match.url, timeout=12.0, headers={"User-Agent": "Mozilla/5.0 (compatible; CricketBaba/3.0)"})
        response.raise_for_status()
        return CricbuzzLiveReader._toss_from_chunks(_extract_flight_chunks(response.text))
    except Exception:
        return "", ""


class OdiBriefingWatcher:
    def __init__(self, publisher=None, only_top10: bool = True, discovery=None, toss_fetcher=fetch_toss,
                 refresh_seconds: float = 60.0, facts_publisher=None) -> None:
        self.publisher = publisher
        self.facts_publisher = facts_publisher if facts_publisher is not None else publisher
        self.only_top10 = only_top10
        self._discovery = discovery
        self._toss_fetcher = toss_fetcher
        self.refresh_seconds = refresh_seconds
        self._state: dict[str, dict] = {}
        self._results: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._next_cycle = 0.0

    def tick(self) -> None:
        """Call every loop of the driver: starts a background cycle when due, prints/publishes finished output."""
        while True:
            try:
                kind, key, text = self._results.get_nowait()
            except queue.Empty:
                break
            print(text)
            target = self.facts_publisher if kind == "facts" else self.publisher if kind == "briefing" else None
            if target is not None:
                try:
                    target.publish(key, text)
                except Exception:
                    pass
        now = time.monotonic()
        if now >= self._next_cycle and (self._thread is None or not self._thread.is_alive()):
            self._next_cycle = now + self.refresh_seconds
            self._thread = threading.Thread(target=self._safe_cycle, daemon=True)
            self._thread.start()

    def _safe_cycle(self) -> None:
        try:
            self.cycle()
        except Exception as error:  # a side watcher must never disturb the T20 driver
            self._results.put(("note", "", f"[odi] watcher error: {type(error).__name__}: {error}"))

    def cycle(self) -> None:
        """One pass over the match list (runs in the worker thread; output goes through the queue)."""
        from app.live.cricbuzz_discovery import CricbuzzMatchDiscovery
        from app.live.prematch_odi import odi_briefing, odi_fact_message

        discovery = self._discovery or CricbuzzMatchDiscovery()
        for match in discovery.matches():
            if not is_men_international_odi(match, self.only_top10):
                continue
            match_key = f"cricbuzz:{match.match_id}"
            teams = [t.strip() for t in match.label.split(" vs ")]
            state = self._state.setdefault(match.match_id, {"toss": None, "frozen": False, "recorded": False})
            phase = match.state.strip().lower()
            if phase in PREVIEW_STATES and not state["frozen"]:
                toss = tuple(self._toss_fetcher(match))
                if toss != state["toss"]:
                    first = state["toss"] is None
                    state["toss"] = toss
                    text = odi_briefing(match.label, match.venue, toss[0], toss[1], match_id=match_key, source="cricbuzz")
                    header = "" if first else "🔔 Toss is in -- updated ODI briefing\n"
                    self._results.put(("briefing", f"briefing:cricbuzz-odi:{match.match_id}:{'toss' if toss[0] else 'first'}",
                                       f"[odi] {match.match_desc} -- {match.series}\n{header}{text}"))
                    try:  # the Fact-o-Meter card is its own message, never part of the briefing
                        facts = odi_fact_message(match.label, match.venue, toss[0], toss[1])
                    except Exception:
                        facts = ""
                    if facts and facts != state.get("facts"):
                        state["facts"] = facts
                        self._results.put(("facts", f"factometer:cricbuzz-odi:{match.match_id}:{'toss' if toss[0] else 'first'}", facts))
                if toss[0]:
                    state["frozen"] = True
            elif phase in ("in progress", "innings break", "stumps", "toss") and state["toss"] is None:
                state["frozen"] = True  # we joined after the start: a "pre-match" briefing would be stale
            elif phase == "complete" and not state["recorded"] and prematch_log.has_prediction(match_key):
                if _is_decided_text(match.status):
                    winner = winner_from_status(match.status, teams)
                    prematch_log.record_result(match_key, winner, match.status)
                    state["recorded"] = True
                    self._results.put(("note", "", f"[odi] result recorded: {match.label} -- {match.status}"))
