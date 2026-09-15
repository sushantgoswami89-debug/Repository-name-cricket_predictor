"""Run the live prediction pipeline against a single Cricbuzz match.

Fallback path for 2026-09-15's field test: TOI never published tonight's
Afghanistan vs India 2nd T20I at all (see
docs/live_field_test_2026-09-15.md), so this points the same
VerifiedLivePredictionPipeline at app/live/cricbuzz_reader.py's
CricbuzzLiveReader instead of ToiLiveReader. Single-match only, no
discovery -- Cricbuzz's own match discovery isn't built yet, so the URL
is passed explicitly.

Usage:
    /usr/bin/python3 run_cricbuzz_match.py "<cricbuzz live-cricket-scores URL>" [--shadow]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from app.live.cricbuzz_reader import CricbuzzFeedError, CricbuzzLiveReader
from app.live.pipeline import VerifiedLivePredictionPipeline
from app.live.telegram import TelegramPublisher
from app.live.verification import VerificationError


def _telegram(project_root: Path) -> TelegramPublisher:
    path = project_root / "backend" / ".telegram.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    return TelegramPublisher(config["bot_token"], config["chat_id"])


def _scorecard_line(snapshot) -> str:
    deliveries = snapshot.deliveries
    latest = deliveries[-1] if deliveries else None
    # snapshot.deliveries only covers balls seen since this reader
    # attached (Cricbuzz's rolling window, not full innings history --
    # see docs/live_field_test_2026-09-15.md item 8), so legal-ball count
    # for run rate must come from the innings-wide snapshot.overs figure,
    # not from counting the (partial) delivery list.
    overs_completed, _, balls_into_over = str(snapshot.overs).partition(".")
    legal_balls = int(overs_completed or 0) * 6 + int(balls_into_over or 0)
    run_rate = (snapshot.score * 6 / legal_balls) if legal_balls else 0.0
    batter = latest.striker if latest else "?"
    bowler = latest.bowler if latest else "?"
    return (
        f"[scorecard] {snapshot.batting_team} {snapshot.score}/{snapshot.wickets} "
        f"({snapshot.overs} ov, RR {run_rate:.2f}) -- {bowler} to {batter}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("match_url", help="Cricbuzz live-cricket-scores URL")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--shadow", action="store_true", help="Do not publish to Telegram."
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    reader = CricbuzzLiveReader()
    publisher = None if args.shadow else _telegram(project_root)
    match_id = reader.match_id_from_url(args.match_url)
    output_file = project_root / "data" / "live" / f"cricbuzz-{match_id}.json"
    pipeline = VerifiedLivePredictionPipeline(publisher=publisher, output_file=output_file)

    mode = "shadow (no Telegram)" if args.shadow else "LIVE (publishing to Telegram)"
    print(f"CricketBaba watching Cricbuzz match {match_id} -- mode: {mode}")

    while True:
        try:
            snapshot = reader.fetch(args.match_url)
        except CricbuzzFeedError as error:
            print(f"Deferred: {error}")
            time.sleep(max(1.0, args.poll_seconds))
            continue
        except requests.exceptions.RequestException as error:
            print(f"Deferred (network): {error}")
            time.sleep(max(1.0, args.poll_seconds))
            continue

        print(_scorecard_line(snapshot))

        try:
            output = pipeline.process(snapshot)
        except VerificationError as error:
            print(f"Verification error: {error}")
            time.sleep(max(1.0, args.poll_seconds))
            continue

        if output:
            print(
                f"{'Sent' if publisher else 'Recorded'} innings {output['innings']} "
                f"over {output['over']} prediction."
            )

        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCricketBaba stopped.")
