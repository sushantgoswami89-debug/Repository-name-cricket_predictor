"""Per-batter innings table for the "settled batter" stat (when he faces N balls, how often does he get 50?).

One row per batter innings in men's T20I + IPL (Cricsheet): [date, batter, balls faced, runs, dismissed, runs after 15 balls
(or -1 if he faced fewer), balls to 50 (or 0)]. Wides are not balls faced; no-balls are. Output data/features/batter_innings.json.
Usage: python build_batter_innings.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "features" / "batter_innings.json"
SOURCES = ("t20i", "ipl")
CHECKPOINT = 15


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
                state: dict[str, dict] = {}
                for over in innings.get("overs", []):
                    for d in over.get("deliveries", []):
                        name = d["batter"]
                        rec = state.setdefault(name, {"balls": 0, "runs": 0, "out": 0, "at15": -1, "to50": 0})
                        if "wides" not in d.get("extras", {}):
                            rec["balls"] += 1
                        rec["runs"] += d["runs"]["batter"]
                        if rec["balls"] == CHECKPOINT and rec["at15"] < 0 and "wides" not in d.get("extras", {}):
                            rec["at15"] = rec["runs"]
                        if rec["runs"] >= 50 and not rec["to50"]:
                            rec["to50"] = rec["balls"]
                        for w in d.get("wickets", []):
                            if w.get("player_out") in state:
                                state[w["player_out"]]["out"] = 1
                for name, rec in state.items():
                    rows.append([date, name, rec["balls"], rec["runs"], rec["out"], rec["at15"], rec["to50"], source, inn_no + 1])
    rows.sort(key=lambda r: r[0])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows))
    print(f"{len(rows)} batter innings -> {OUT}")


if __name__ == "__main__":
    main()
