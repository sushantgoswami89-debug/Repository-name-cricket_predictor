"""Automatically discover and process TOI live cricket matches."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests

from app.live.commentary_log import append_commentary_log
from app.live.pipeline import MatchStateAnomaly, VerifiedLivePredictionPipeline
from app.live.prematch_analysis import confirmed_xi, fact_o_meter_block, prematch_header, team_card_block, prematch_text, toss_line, venue_summary, xi_line, xi_names
from app.live.telegram import TelegramPublisher
from app.live.toi_discovery import ToiMatchDiscovery, is_t20_international
from app.live.toi_reader import ToiFeedError, ToiLiveReader
from app.live.verification import VerificationError


def _telegram(project_root: Path) -> TelegramPublisher:
    path = project_root / "backend" / ".telegram.json"
    if not path.exists():
        raise SystemExit("Run setup_telegram.py before starting automatic mode.")
    config = json.loads(path.read_text(encoding="utf-8"))
    return TelegramPublisher(config["bot_token"], config["chat_id"])


def _briefing_text(reader: ToiLiveReader, match_id: str, label: str) -> str | None:
    """One-time pre-match summary: toss, pitch, weather, playing XIs.

    Same raw TOI feed and extraction the standalone match_briefing.py
    already used for this -- reused here so the automated Telegram flow
    carries it too instead of requiring a human to run that script
    separately. Returns None on any feed error so a briefing failure
    never blocks the per-over prediction flow that follows it.
    """
    try:
        raw = reader._get_json(ToiLiveReader.RAW_URL.format(match_id=match_id))
    except Exception:
        return None
    team_players = ToiLiveReader._team_players(raw)
    toss_won_by, toss_decision = ToiLiveReader._toss(raw)
    pitch_type, pitch_surface = ToiLiveReader._pitch(raw)
    weather_condition, humidity, temperature, wind = ToiLiveReader._weather(raw)

    team_names = list(team_players)
    city = ToiLiveReader._venue_name(raw).rsplit(",", 1)[-1].strip()
    lines = [f"🏏 {label}" + (f" -- {city} | {venue_summary(city.lower())}" if city else "")]
    if len(team_names) == 2:
        toi_note = " | ".join(x for x in (pitch_type, pitch_surface) if x and "unknown" not in x.lower())
        try:
            lines.extend(
                prematch_header(
                    team_names[0], team_names[1], ToiLiveReader._venue_name(raw), toss_won_by, toss_decision,
                    confirmed_xi(team_players), pitch_note=toi_note,
                )
            )
        except Exception as error:  # never block the briefing, but do not hide the bug
            print(f"[briefing] header unavailable: {error!r}")
    lines.append(
        toss_line(toss_won_by, toss_decision, team_names[0] if len(team_names) == 2 else "", team_names[1] if len(team_names) == 2 else "")
    )
    if len(team_names) == 2:
        analysis = prematch_text(
            team_names[0], team_names[1], ToiLiveReader._venue_name(raw), toss_won_by, toss_decision,
            confirmed_xi(team_players), log_context={"match_id": f"toi:{match_id}", "source": "toi"},
        )
        if analysis:
            lines.append("")
            lines.append(analysis)
    if len(team_names) == 2:
        facts = team_card_block(
            team_names[0], team_names[1], ToiLiveReader._venue_name(raw), toss_won_by, toss_decision
        ) + fact_o_meter_block(
            team_names[0], team_names[1], confirmed_xi(team_players), ToiLiveReader._venue_name(raw), toss_won_by, toss_decision
        )
        if facts:
            lines.append("")
            lines.extend(facts)
    # The whole line-up block (status line, then the names once both XIs are full) goes last.
    lines.append("")
    lines.append(xi_line(team_players))
    lines.extend(xi_names(team_players))
    return "\n".join(lines)


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
    # User-requested 2026-09-18: per-delivery commentary log for detailed
    # post-match review -- see app/live/commentary_log.py's docstring.
    logged_delivery_keys: dict[str, set[tuple[int, int, int]]] = {}
    last_scan = 0.0
    mode = "T20I shadow matches" if args.shadow_t20i else "live matches"
    print(f"CricketBaba is watching TOI for {mode}. Press Control+C to stop.")
    while True:
        now = time.monotonic()
        if now - last_scan >= max(10.0, args.scan_seconds):
            try:
                matches = tuple(
                    match
                    for match in discovery.live_summaries()
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
                    reader = ToiLiveReader()
                    active[match.match_id] = (
                        match.url,
                        reader,
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
                    briefing = _briefing_text(reader, match.match_id, match.label)
                    if briefing:
                        print(briefing)
                        if publisher is not None:
                            publisher.publish(f"briefing:{match.match_id}", briefing)
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
            except (ToiFeedError, VerificationError, MatchStateAnomaly) as error:
                print(f"Deferred {match_id}: {error}")
                continue
            append_commentary_log(
                project_root / "data" / "live" / f"commentary-{match_id}.jsonl",
                match_id,
                snapshot,
                logged_delivery_keys.setdefault(match_id, set()),
            )
            if output:
                if args.shadow_t20i:
                    _write_shadow_prediction(shadow_root, match_id, output)
                    print(
                        f"Recorded {match_id} innings {output['innings']} "
                        f"over {output['over']} shadow prediction."
                    )
                    print(VerifiedLivePredictionPipeline._telegram_text(output))
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
