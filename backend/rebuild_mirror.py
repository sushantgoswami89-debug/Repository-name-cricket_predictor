"""Rebuild the full over-summary mirror of a finished match from the Cricbuzz pages the driver saved (data/raw/cricbuzz-pages/<id>/).

Every saved page holds the over separators of the last ~8 overs, so the union over all polls covers every over even when the driver was
restarted or the Mac slept for a while (the live file over-summaries-<id>.jsonl only has what a running session drained). Writes
data/live/over-summaries-<id>.full.jsonl (one line per over, last copy wins) and prints the coverage per innings.

    cd backend && ../.venv/bin/python rebuild_mirror.py 155541 [--overs 20]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from app.live.cricbuzz_reader import _SEPARATOR_MARKER, _extract_balanced, _extract_flight_chunks
from replay_saved_pages import ROOT, load_pages


def separators_in(html: str) -> list[dict]:
    found = []
    for chunk in _extract_flight_chunks(html):
        for hit in re.finditer(re.escape(_SEPARATOR_MARKER), chunk):
            raw = _extract_balanced(chunk, hit.start())
            if raw is None:
                continue
            try:
                holder = json.loads(raw)
            except json.JSONDecodeError:
                continue
            separator = holder.get("overSeparator")
            if not isinstance(separator, dict) or separator.get("overNumber") is None:
                continue
            innings = holder.get("inningsId")
            if not isinstance(innings, int):
                continue
            bowler = separator.get("bowlerObj") if isinstance(separator.get("bowlerObj"), dict) else {}
            found.append({
                "innings": innings, "over": int(separator["overNumber"]), "summary": str(separator.get("overSummary", "")).strip(),
                "over_runs": separator.get("overRuns"), "score": (separator.get("batTeamObj") or {}).get("teamScore"),
                "bowler": bowler.get("playerName"), "bowler_figures": bowler.get("playerScore"),
                "striker": (separator.get("batStrikerObj") or {}).get("playerName"),
                "non_striker": (separator.get("batNonStrikerObj") or {}).get("playerName"),
                "timestamp_ms": holder.get("timestamp"),
            })
    return found


def rebuild(match_id: str, overs: int = 20) -> dict[int, list[int]]:
    latest: dict[tuple[int, int], dict] = {}
    pages = load_pages(match_id)
    for _stamp, html in pages:
        for row in separators_in(html):
            key = (row["innings"], row["over"])
            if key not in latest or (row["timestamp_ms"] or 0) >= (latest[key]["timestamp_ms"] or 0):
                latest[key] = row
    out = ROOT / "data" / "live" / f"over-summaries-{match_id}.full.jsonl"
    out.write_text("".join(json.dumps(latest[key]) + "\n" for key in sorted(latest)), encoding="utf-8")
    missing: dict[int, list[int]] = {}
    print(f"=== {match_id}: {len(pages)} saved pages -> {len(latest)} overs with Cricbuzz's own over summary -> {out.name}")
    for innings in sorted({k[0] for k in latest}):
        have = {over for (i, over) in latest if i == innings}
        missing[innings] = [o for o in range(1, max(have) + 1) if o not in have]
        print(f"  innings {innings}: overs 1-{max(have)} held {len(have)} of {max(have)}"
              + (f", missing {missing[innings]}" if missing[innings] else ", none missing"))
    return missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("match_ids", nargs="+")
    args = parser.parse_args()
    for match in args.match_ids:
        rebuild(match)
