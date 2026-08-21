"""Wait for a live T20/ODI match, then measure two things about the
"announced bowler" signal the bowler_shadow_predictor needs:

1. Hit rate: for what fraction of overs does TOI pre-announce the next
   bowler (a numbered next-over row with a bowler but zero balls) before
   the first delivery of that over lands?
2. Lead time: when it IS announced early, how much wall-clock time elapses
   between first seeing the announcement and the over's first delivery
   landing? That's the real time budget available to compute and publish
   a shadow-adjusted prediction before the over starts.

Writes results incrementally to announced_bowler_monitor.log (JSON lines)
so progress survives even if this process is interrupted.
"""

import json
import time
from pathlib import Path

from app.live.toi_discovery import ToiMatchDiscovery
from app.live.toi_reader import ToiFeedError, ToiLiveReader

LOG_PATH = Path(__file__).resolve().parent / "announced_bowler_monitor.log"
DISCOVERY_POLL_SECONDS = 90
ACTIVE_POLL_SECONDS = 15
EXCLUDED_FORMATS = {"Test", "First Class", "List A"}


def log(event: dict) -> None:
    event["ts"] = time.time()
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(event) + "\n")
    print(json.dumps(event))


def find_live_limited_overs_match():
    discovery = ToiMatchDiscovery()
    for summary in discovery.live_summaries():
        if summary.is_live and summary.match_format not in EXCLUDED_FORMATS:
            return summary
    return None


def monitor_match(summary) -> None:
    log({"event": "monitoring_started", "match": summary.label, "format": summary.match_format, "url": summary.url})
    reader = ToiLiveReader()

    # over_number -> first time we saw it pre-announced (no balls yet, bowler set)
    announced_at: dict[int, float] = {}
    # over_number -> first time we saw a delivery recorded for it
    first_delivery_at: dict[int, float] = {}
    finalized_overs: set[int] = set()

    consecutive_errors = 0
    while True:
        try:
            snapshot = reader.fetch(summary.url)
            consecutive_errors = 0
        except ToiFeedError as exc:
            consecutive_errors += 1
            log({"event": "feed_error", "detail": str(exc)[:200], "consecutive": consecutive_errors})
            if consecutive_errors > 40:
                log({"event": "giving_up_too_many_errors"})
                return
            time.sleep(ACTIVE_POLL_SECONDS)
            continue
        except Exception as exc:
            log({"event": "unexpected_error", "detail": f"{type(exc).__name__}: {exc}"[:200]})
            time.sleep(ACTIVE_POLL_SECONDS)
            continue

        if not snapshot.is_live:
            log({"event": "match_ended", "final_score": snapshot.score, "final_wickets": snapshot.wickets})
            break

        now = time.time()

        if snapshot.announced_bowler and snapshot.announced_bowler_over:
            over = snapshot.announced_bowler_over
            if over not in announced_at and over not in first_delivery_at:
                announced_at[over] = now
                log({"event": "bowler_announced_early", "over": over, "bowler": snapshot.announced_bowler})

        for delivery in snapshot.deliveries:
            over = delivery.over
            if over not in first_delivery_at:
                first_delivery_at[over] = now
                lead_time = announced_at.get(over)
                if lead_time is not None and over not in finalized_overs:
                    finalized_overs.add(over)
                    log({
                        "event": "over_started_with_lead_time",
                        "over": over,
                        "lead_time_seconds": round(now - lead_time, 1),
                    })
                elif over not in finalized_overs:
                    finalized_overs.add(over)
                    log({"event": "over_started_no_advance_announcement", "over": over})

        time.sleep(ACTIVE_POLL_SECONDS)

    announced_overs = len(announced_at)
    total_overs = len(first_delivery_at)
    log({
        "event": "summary",
        "total_overs_seen": total_overs,
        "overs_with_advance_announcement": announced_overs,
        "hit_rate_percent": round(announced_overs / total_overs * 100, 1) if total_overs else None,
    })


def main():
    log({"event": "watch_started"})
    while True:
        summary = find_live_limited_overs_match()
        if summary is not None:
            monitor_match(summary)
            log({"event": "watch_resuming_after_match_end"})
        time.sleep(DISCOVERY_POLL_SECONDS)


if __name__ == "__main__":
    main()
