"""The mirror: Cricbuzz's own over summaries and scorecard, over by over, against what our commentary-built record says.

User 2026-10-20: the scorecard and the over summary already hold the full facts (every ball, every bowler's figures, every batter's balls);
commentary is only needed for the feel of the pitch and for pressure on a bowler or batter. So the OFFICIAL record is:
  data/live/over-summaries-<id>.jsonl   one line per finished over: ball tokens, over runs, team score, bowler + figures, striker, non-striker
  data/live/scorecard-<id>.jsonl        the whole scorecard at each over end
This prints that record as a table and checks our commentary-built deliveries (data/live/commentary-<id>.jsonl) token for token against it.

    cd backend && ../.venv/bin/python mirror_report.py 170723 [--quiet]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def token(row: dict) -> str:
    """Our delivery written the way Cricbuzz writes it in an over summary ("4", "W", "Wd", "Wd3", "N1", "L2", "B4", "W1")."""
    extras = row.get("extras") or {}
    total = row["total_runs"]
    if "wides" in extras:
        return "Wd" if total <= 1 else f"Wd{total}"
    if "noballs" in extras:
        bat = row["batter_runs"]
        return "N" if bat == 0 else f"N{bat}"
    if row.get("wicket_kind"):
        return "W" if total == 0 else f"W{total}"
    if "legbyes" in extras:
        return f"L{extras['legbyes']}"
    if "byes" in extras:
        return f"B{extras['byes']}"
    return str(total)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def report(match_id: str, quiet: bool = False) -> tuple[int, int, int]:
    live = ROOT / "data" / "live"
    summaries = load_jsonl(live / f"over-summaries-{match_id}.jsonl")
    ours: dict[tuple[int, int], dict[int, dict]] = {}
    for row in load_jsonl(live / f"commentary-{match_id}.jsonl"):
        ours.setdefault((row["innings"], row["over"]), {})[row["ball"]] = row   # last copy of a ball wins
    latest: dict[tuple[int, int], dict] = {}
    for s in summaries:
        latest[(s["innings"], s["over"])] = s   # the last separator Cricbuzz sent for an over
    print(f"=== {match_id}: {len(latest)} overs in Cricbuzz's own over summaries, {len(load_jsonl(live / f'scorecard-{match_id}.jsonl'))} scorecard snapshots")
    held = same = 0
    for (innings, over), s in sorted(latest.items()):
        theirs = str(s["summary"]).split()
        mine_rows = [ours[(innings, over)][b] for b in sorted(ours.get((innings, over), {}))]
        mine = [token(r) for r in mine_rows]
        if not quiet:
            print(f"  inn {innings} over {over:>2}: {' '.join(theirs):<22} = {s['over_runs']:>2} runs | {s['score']:>7} | {s['bowler']} {s['bowler_figures']} | "
                  f"{s['striker']}, {s['non_striker']}")
        if not mine_rows:
            continue
        held += 1
        if mine == theirs:
            same += 1
        else:
            print(f"      ours differs: {' '.join(mine)}  (Cricbuzz {' '.join(theirs)})")
    print(f"  commentary record vs the mirror, token for token: {same} of {held} overs we hold are identical")
    return len(latest), held, same


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    for match in [a for a in sys.argv[1:] if not a.startswith("--")]:
        report(match, quiet)
