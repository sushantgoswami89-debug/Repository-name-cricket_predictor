"""Automatically discover and process live Cricbuzz T20I matches.

Closes the "Cricbuzz match discovery still hardcoded to one URL" gap --
run_cricbuzz_match.py has always required the match URL as a command-line
argument (built 2026-09-15 under time pressure as a single-match fallback
after TOI failed to publish a real live match at all). This mirrors
run_toi_auto.py's discover -> attach -> poll loop, but sourced from
app/live/cricbuzz_discovery.py's CricbuzzMatchDiscovery instead of TOI's,
and reusing run_cricbuzz_match.py's exact same per-match polling defenses
via app/live/cricbuzz_session.py's CricbuzzMatchSession (the freshest-of-3
fetch, the never-go-backwards progress guard, and the auto-reseed-after-
errors recovery -- all three exist because of real bugs seen live on
2026-09-17, not speculative hardening).

Usage:
    /usr/bin/python3 run_cricbuzz_auto.py [--shadow]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from app.live.cricbuzz_discovery import CricbuzzMatchDiscovery, is_t20_international
from app.live.cricbuzz_session import CricbuzzMatchSession, briefing_text
from app.live.pipeline import VerifiedLivePredictionPipeline
from app.live.telegram import TelegramPublisher


def _telegram(project_root: Path) -> TelegramPublisher:
    path = project_root / "backend" / ".telegram.json"
    if not path.exists():
        raise SystemExit("Run setup_telegram.py before starting automatic mode.")
    config = json.loads(path.read_text(encoding="utf-8"))
    return TelegramPublisher(config["bot_token"], config["chat_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-seconds", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--shadow", action="store_true", help="Do not publish to Telegram."
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    discovery = CricbuzzMatchDiscovery()
    publisher = None if args.shadow else _telegram(project_root)
    active: dict[str, CricbuzzMatchSession] = {}
    last_scan = 0.0
    mode = "shadow (no Telegram)" if args.shadow else "LIVE (publishing to Telegram)"
    print(f"CricketBaba is watching Cricbuzz for live T20Is -- mode: {mode}. Press Control+C to stop.")

    while True:
        now = time.monotonic()
        if now - last_scan >= max(10.0, args.scan_seconds):
            try:
                matches = tuple(
                    match
                    for match in discovery.live_matches()
                    if is_t20_international(match)
                )
            except requests.exceptions.RequestException as error:
                print(f"Deferred discovery scan: {error}")
                last_scan = now
                time.sleep(max(1.0, args.poll_seconds))
                continue
            live_ids = {match.match_id for match in matches}
            for match in matches:
                if match.match_id not in active:
                    output_file = (
                        project_root / "data" / "live" / f"cricbuzz-{match.match_id}.json"
                    )
                    active[match.match_id] = CricbuzzMatchSession(
                        match_id=match.match_id,
                        match_url=match.url,
                        pipeline=VerifiedLivePredictionPipeline(
                            publisher=publisher, output_file=output_file
                        ),
                        publisher=publisher,
                    )
                    print(f"Attached automatically: {match.label} ({match.match_desc})")
                    briefing = briefing_text(match.match_id, match.url, match.label)
                    if briefing:
                        print(briefing)
                        if publisher is not None:
                            publisher.publish(f"briefing:{match.match_id}", briefing)
            for match_id in set(active) - live_ids:
                if active[match_id].finished:
                    print(f"Match finished: {match_id}")
                else:
                    print(f"Match no longer listed live, detaching: {match_id}")
                del active[match_id]
            if not matches:
                print("No live T20I on Cricbuzz right now; continuing to watch.")
            last_scan = now

        for match_id, session in tuple(active.items()):
            session.poll_once()
            if session.finished:
                del active[match_id]

        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCricketBaba stopped.")
