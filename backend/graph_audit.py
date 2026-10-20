"""Cricbuzz's graphs page as a second official record: runs and wickets of EVERY over of both innings.

The graphs page (live-cricket-graphs) holds runsPerOverChartData -- for each over the runs of both teams and a wicket flag -- for the whole
match, and it can be fetched at any time (during the match or days later), unlike the main page whose window keeps only the last 2-3 overs
and the over-by-over page which keeps the last 10. So it shows which overs we never recorded and whether the ones we did record are right.

    cd backend && ../.venv/bin/python graph_audit.py 155541 [--live]    (--live fetches the page now instead of the saved final copy)
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from app.live.cricbuzz_capture import fetch_page, per_over_from_graphs
from replay_saved_pages import ROOT


def our_overs(match_id: str) -> dict[tuple[int, int], dict]:
    path = ROOT / "data" / "live" / f"commentary-{match_id}.jsonl"
    balls: dict[tuple[int, int, int], dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                balls[(row["innings"], row["over"], row["ball"])] = row
    table: dict[tuple[int, int], dict] = defaultdict(lambda: {"runs": 0, "wicket": False, "balls": 0})
    for (innings, over, _ball), row in balls.items():
        entry = table[(innings, over)]
        entry["runs"] += row["total_runs"]
        entry["wicket"] = entry["wicket"] or bool(row.get("wicket_kind"))
        entry["balls"] += 1
    return dict(table)


def audit(match_id: str, live: bool = False) -> None:
    saved = ROOT / "data" / "raw" / "cricbuzz-pages" / match_id / "final" / "live-cricket-graphs.html.gz"
    html = fetch_page("live-cricket-graphs", match_id) if live or not saved.exists() else gzip.decompress(saved.read_bytes()).decode("utf-8", "ignore")
    theirs = per_over_from_graphs(html or "")
    if not theirs:
        print(f"=== {match_id}: no graphs data")
        return
    ours = our_overs(match_id)
    print(f"=== {match_id}: Cricbuzz graphs hold {len(theirs)} overs; we recorded {len(ours)}")
    for innings in sorted({k[0] for k in theirs}):
        overs = sorted(o for (i, o) in theirs if i == innings)
        exact = different = missing = 0
        notes = []
        for over in overs:
            mine = ours.get((innings, over))
            their = theirs[(innings, over)]
            if mine is None:
                missing += 1
            elif (mine["runs"], mine["wicket"]) == (their["runs"], their["wicket"]):
                exact += 1
            else:
                different += 1
                notes.append(f"over {over}: ours {mine['runs']} runs{' +W' if mine['wicket'] else ''} ({mine['balls']} balls) vs Cricbuzz {their['runs']}{' +W' if their['wicket'] else ''}")
        absent = [o for o in overs if (innings, o) not in ours]
        print(f"  innings {innings}: {len(overs)} overs | exact {exact} | different {different} | not recorded {missing}"
              + (f" (overs {absent})" if absent else ""))
        for note in notes:
            print("     ", note)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("match_ids", nargs="+")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    for match in args.match_ids:
        audit(match, args.live)
