"""Study-only commentary logger: records every live Cricbuzz match's
full per-delivery commentary (any format, any level) and makes NO
predictions and sends NO Telegram messages.

Why: the prediction driver only attaches T20Is, but reading commentary
from every live match (domestic, associate, women's) is how the prose
signals in app/live/shot_quality.py get verified against real text and
how new parsing bugs get found. Cricbuzz's page carries only a rolling
window of recent balls, so anything not logged while live is gone.

Output: data/live/study/commentary-<match_id>.jsonl (same row format as
the prediction driver's log) and data/live/study/index.json (labels).

Usage:
    python3 run_commentary_study.py [--poll-seconds 15] [--scan-seconds 60]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.live.cricbuzz_discovery import CricbuzzMatchDiscovery
from app.live.cricbuzz_reader import CricbuzzLiveReader
from app.live.commentary_log import append_commentary_log


# Top-10 men's full-member national sides. Static list written from
# general knowledge of the ICC men's T20I rankings as of early 2026, NOT
# fetched live -- edit it, or pass --teams, when the rankings move.
TOP_TEAMS = (
    "India", "Australia", "England", "South Africa", "New Zealand",
    "Pakistan", "West Indies", "Sri Lanka", "Afghanistan", "Bangladesh",
)


def involves_top_team(label: str, teams: tuple[str, ...]) -> bool:
    """True only for a men's match where BOTH sides of "A vs B" are in
    `teams` (case-insensitive, exact names). Women's sides ("... Women"),
    "A"/age-group/domestic sides, and any match with one unlisted side
    never qualify."""
    wanted = {t.lower() for t in teams}
    sides = [side.strip().lower() for side in label.split(" vs ")]
    if len(sides) != 2:
        return False
    return all(side in wanted for side in sides)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-seconds", type=float, default=60.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--teams", default=",".join(TOP_TEAMS),
        help="Comma-separated team names to follow (a match qualifies only if BOTH sides are listed; men's only). "
             "Pass an empty string to follow every live match.",
    )
    args = parser.parse_args()
    teams = tuple(t.strip() for t in args.teams.split(",") if t.strip())
    out_dir = Path(__file__).resolve().parents[1] / "data" / "live" / "study"
    out_dir.mkdir(parents=True, exist_ok=True)
    discovery = CricbuzzMatchDiscovery()
    active: dict[str, tuple[str, CricbuzzLiveReader, set]] = {}
    index: dict[str, str] = {}
    last_scan = 0.0
    print(f"Commentary study logger running (no predictions, no Telegram). Following: {', '.join(teams) or 'every live match'}. Control+C to stop.")

    while True:
        if time.monotonic() - last_scan >= max(20.0, args.scan_seconds):
            last_scan = time.monotonic()
            try:
                live = list(discovery.live_matches())
            except Exception as error:  # noqa: BLE001 - keep watching
                print(f"Discovery failed: {type(error).__name__}: {error}")
                live = []
            live_ids = {
                m.match_id for m in live if not teams or involves_top_team(m.label, teams)
            }
            for match in live:
                if teams and not involves_top_team(match.label, teams):
                    continue
                if match.match_id not in active:
                    active[match.match_id] = (match.url, CricbuzzLiveReader(), set())
                    index[match.match_id] = f"{match.label} ({match.match_desc}, {match.match_format})"
                    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
                    print(f"Attached: {index[match.match_id]}")
            for match_id in tuple(active):
                if match_id not in live_ids:
                    print(f"Finished/not live, detaching: {index.get(match_id, match_id)}")
                    del active[match_id]

        for match_id, (url, reader, logged) in tuple(active.items()):
            try:
                snapshot = reader.fetch(url)
            except Exception as error:  # noqa: BLE001 - feed hiccups are routine
                print(f"[{match_id}] fetch error: {type(error).__name__}")
                continue
            before = len(logged)
            append_commentary_log(out_dir / f"commentary-{match_id}.jsonl", match_id, snapshot, logged)
            if len(logged) > before:
                print(f"[{match_id}] +{len(logged) - before} balls ({snapshot.batting_team} {snapshot.score}/{snapshot.wickets})")
        time.sleep(max(5.0, args.poll_seconds))


if __name__ == "__main__":
    main()
