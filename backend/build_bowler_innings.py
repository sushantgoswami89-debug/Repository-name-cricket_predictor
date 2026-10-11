"""Per-bowler innings table for the bowler "100% record" stat (e.g. he takes a wicket every time he bowls 4 overs).

One row per bowler innings in men's T20I + IPL (Cricsheet): [date, bowler, legal balls, runs conceded, wickets, source].
Wickets are bowler-credited (not run outs, retirements, obstructing, timed out). Runs conceded exclude byes / leg-byes.
Output data/features/bowler_innings.json. Usage: python build_bowler_innings.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ODI = len(sys.argv) > 1 and sys.argv[1] == "odi"  # `python build_bowler_innings.py odi` builds the ODI table
OUT = ROOT / "data" / "features" / ("bowler_innings_odi.json" if ODI else "bowler_innings.json")
SOURCES = ("odi",) if ODI else ("t20i", "ipl")
NOT_BOWLER = {"run out", "retired hurt", "retired out", "obstructing the field", "timed out"}


def main() -> None:
    rows = []
    for source in SOURCES:
        for path in sorted((ROOT / "data" / "raw" / "cricsheet" / source).glob("*.json")):
            match = json.loads(path.read_text(encoding="utf-8"))
            info = match["info"]
            if info.get("gender") not in (None, "male"):
                continue
            date = info["dates"][0]
            for inn_no, innings in enumerate(match.get("innings", [])):
                if innings.get("super_over") or inn_no > 1:
                    continue
                state: dict[str, list[int]] = {}
                for over in innings.get("overs", []):
                    for d in over.get("deliveries", []):
                        rec = state.setdefault(d["bowler"], [0, 0, 0])
                        extras = d.get("extras", {})
                        if "wides" not in extras and "noballs" not in extras:
                            rec[0] += 1
                        rec[1] += d["runs"]["total"] - extras.get("byes", 0) - extras.get("legbyes", 0)
                        rec[2] += sum(1 for w in d.get("wickets", []) if w.get("kind") not in NOT_BOWLER)
                for name, (balls, runs, wkts) in state.items():
                    rows.append([date, name, balls, runs, wkts, source])
    rows.sort(key=lambda r: r[0])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows))
    print(f"{len(rows)} bowler innings -> {OUT}")


if __name__ == "__main__":
    main()
