"""Run the live prediction pipeline against a single Cricbuzz match.

Fallback path for 2026-09-15's field test: TOI never published tonight's
Afghanistan vs India 2nd T20I at all (see
docs/live_field_test_2026-09-15.md), so this points the same
VerifiedLivePredictionPipeline at app/live/cricbuzz_reader.py's
CricbuzzLiveReader instead of ToiLiveReader.

Single-match only, no discovery -- for automatic discovery + attach
across multiple live T20Is, see run_cricbuzz_auto.py (built 2026-09-18),
which shares this script's per-match polling logic via
app/live/cricbuzz_session.py's CricbuzzMatchSession. Sends a one-time
pre-match briefing (toss/venue/confirmed XIs) via that same module's
briefing_text() before the first poll -- added 2026-09-18 after this
script (the one both real field tests, 09-15 and 09-17, actually used)
was found to have no briefing at all despite run_cricbuzz_auto.py
getting one from the start.

Usage:
    /usr/bin/python3 run_cricbuzz_match.py "<cricbuzz live-cricket-scores URL>" [--shadow]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.live.cricbuzz_reader import CricbuzzLiveReader
from app.live.cricbuzz_session import CricbuzzMatchSession, briefing_text
from app.live.pipeline import VerifiedLivePredictionPipeline
from app.live.telegram import TelegramPublisher


def _telegram(project_root: Path) -> TelegramPublisher:
    path = project_root / "backend" / ".telegram.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    return TelegramPublisher(config["bot_token"], config["chat_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("match_url", help="Cricbuzz live-cricket-scores URL")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--shadow", action="store_true", help="Do not publish to Telegram."
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    publisher = None if args.shadow else _telegram(project_root)
    match_id = CricbuzzLiveReader.match_id_from_url(args.match_url)
    output_file = project_root / "data" / "live" / f"cricbuzz-{match_id}.json"
    pipeline = VerifiedLivePredictionPipeline(publisher=publisher, output_file=output_file)
    session = CricbuzzMatchSession(
        match_id=match_id,
        match_url=args.match_url,
        pipeline=pipeline,
        publisher=publisher,
    )

    mode = "shadow (no Telegram)" if args.shadow else "LIVE (publishing to Telegram)"
    print(f"CricketBaba watching Cricbuzz match {match_id} -- mode: {mode}")

    briefing = briefing_text(match_id, args.match_url, match_id)
    if briefing:
        print(briefing)
        if publisher is not None:
            publisher.publish(f"briefing:{match_id}", briefing)

    while True:
        session.poll_once()
        if session.finished:
            break
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCricketBaba stopped.")
