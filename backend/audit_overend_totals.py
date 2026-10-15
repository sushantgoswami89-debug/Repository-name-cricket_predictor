"""At the moment each over ended LIVE, was our total right?  (the forecast is made from that total)

For every over end: the first live poll whose overs read N.0 (data/live/board-<id>.jsonl, 'ours') against Cricbuzz's own team score
after that over (the over separator in the saved pages). Wrong-then-corrected totals are exactly what a forecast could be built on.
    cd backend && ../.venv/bin/python audit_overend_totals.py 170697 151587 170690
"""
import json
import re
import sys
from pathlib import Path

from audit_recording import separators
from replay_saved_pages import load_pages

ROOT = Path(__file__).resolve().parents[1]

for match in sys.argv[1:]:
    truth = separators(match, load_pages(match))
    first = {}
    for line in open(ROOT / "data" / "live" / f"board-{match}.jsonl", encoding="utf-8"):
        row = json.loads(line)
        overs = float(row["ours"]["overs"])
        if abs(overs - round(overs)) < 1e-9 and overs > 0:
            first.setdefault((row["innings"], int(round(overs))), row)
    ok = bad = 0
    print(f"=== {match}")
    for key, row in sorted(first.items()):
        sep = truth.get(key)
        score = str((sep or {}).get("batTeamObj", {}).get("teamScore", ""))
        hit = re.fullmatch(r"(\d+)-(\d+)", score)
        if not hit:
            continue
        ours = (row["ours"]["score"], row["ours"]["wickets"])
        if ours == (int(hit.group(1)), int(hit.group(2))):
            ok += 1
        else:
            bad += 1
            print(f"  inn {key[0]} over {key[1]}: our total at the over end {ours[0]}/{ours[1]} vs Cricbuzz {score}")
    print(f"  {ok} over ends exact, {bad} wrong when first polled")
