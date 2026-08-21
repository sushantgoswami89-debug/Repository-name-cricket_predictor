"""Automatically discover and process TOI live cricket matches."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from app.live.pipeline import VerifiedLivePredictionPipeline
from app.live.telegram import TelegramPublisher
from app.live.toi_discovery import ToiMatchDiscovery, ToiMatchSummary
from app.live.toi_reader import ToiFeedError, ToiLiveReader
from app.live.verification import VerificationError


def _telegram(project_root: Path) -> TelegramPublisher:
    path = project_root / "backend" / ".telegram.json"
    if not path.exists():
        raise SystemExit("Run setup_telegram.py before starting automatic mode.")
    config = json.loads(path.read_text(encoding="utf-8"))
    return TelegramPublisher(config["bot_token"], config["chat_id"])


def _is_t20_international(match: ToiMatchSummary) -> bool:
    match_format = match.match_format.strip().upper()
    team_type = match.team_type.strip().lower()
    series = match.series.strip().lower()
    if "indian premier league" in series or match_format == "IPL":
        return False
    if match_format in {"T20I", "IT20"} or "international" in team_type:
        return True
    international_markers = (
        "world cup",
        "asia cup",
        "tour of",
        "tri-series",
        "international",
    )
    return match_format in {"T20", "TWENTY20"} and any(
        marker in series for marker in international_markers
    )


def _write_shadow_prediction(
    root: Path, match_id: str, output: dict[str, object]
) -> Path:
    directory = root / match_id / "predictions"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (
        f"innings-{output['innings']}-over-{output['over']}.json"
    )
    if destination.exists():
        return destination
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=directory, delete=False
    ) as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-seconds", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--shadow-t20i",
        action="store_true",
        help="Monitor T20 internationals without Telegram publication.",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    discovery = ToiMatchDiscovery()
    publisher = None if args.shadow_t20i else _telegram(project_root)
    shadow_root = project_root / "data" / "live" / "shadow-t20i"
    active: dict[str, tuple[str, ToiLiveReader, VerifiedLivePredictionPipeline]] = {}
    last_scan = 0.0
    mode = "T20I shadow matches" if args.shadow_t20i else "live matches"
    print(f"CricketBaba is watching TOI for {mode}. Press Control+C to stop.")
    while True:
        now = time.monotonic()
        if now - last_scan >= max(10.0, args.scan_seconds):
            if args.shadow_t20i:
                matches = tuple(
                    match
                    for match in discovery.live_summaries()
                    if _is_t20_international(match)
                )
            else:
                matches = discovery.live_matches()
            live_ids = {match.match_id for match in matches}
            for match in matches:
                if match.match_id not in active:
                    active[match.match_id] = (
                        match.url,
                        ToiLiveReader(),
                        VerifiedLivePredictionPipeline(
                            publisher=publisher,
                            output_file=(
                                shadow_root / match.match_id / "latest.json"
                                if args.shadow_t20i
                                else project_root
                                / "data"
                                / "live"
                                / f"{match.match_id}.json"
                            ),
                        ),
                    )
                    print(f"Attached automatically: {match.label}")
            for match_id in set(active) - live_ids:
                del active[match_id]
                print(f"Match finished: {match_id}")
            if not matches:
                print("No match is live; continuing to watch.")
            last_scan = now
        for match_id, (url, reader, pipeline) in tuple(active.items()):
            try:
                snapshot = reader.fetch(url)
                output = pipeline.process(snapshot)
            except (ToiFeedError, VerificationError) as error:
                print(f"Deferred {match_id}: {error}")
                continue
            if output:
                if args.shadow_t20i:
                    _write_shadow_prediction(shadow_root, match_id, output)
                    print(
                        f"Recorded {match_id} innings {output['innings']} "
                        f"over {output['over']} shadow prediction."
                    )
                else:
                    print(
                        f"Sent {match_id} innings {output['innings']} "
                        f"over {output['over']} prediction."
                    )
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCricketBaba stopped.")
