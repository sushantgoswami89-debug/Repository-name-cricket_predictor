"""Player performance -> team result table for the Fact-o-Meter "when he does X his team wins/loses" facts.

One row per player per match (men's, decided matches only; ties / no results / super-over-decided skipped):
  [date, player, team, batted innings (1/2/0), runs, balls, bowled innings (1/2/0), wickets, balls bowled, team won (1/0), source, opponent]
"batted innings" is the innings number his team batted in (0 if he did not bat); "bowled innings" is the innings number in
which he bowled (the other one). Wickets are bowler-credited. Output data/features/player_results[_t20|_odi].json.
Usage: python build_player_results.py [odi]   (default: all men's T20 competitions)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ODI = len(sys.argv) > 1 and sys.argv[1] == "odi"
OUT = ROOT / "data" / "features" / ("player_results_odi.json" if ODI else "player_results.json")
SOURCES = ("odi",) if ODI else ("t20i", "ipl", "bbl", "bpl", "cpl", "ilt", "lpl", "mlc", "msl", "ntb", "psl", "sat", "sma", "ssm")
NOT_BOWLER = {"run out", "retired hurt", "retired out", "obstructing the field", "timed out"}


def main() -> None:
    rows = []
    for source in SOURCES:
        for path in sorted((ROOT / "data" / "raw" / "cricsheet" / source).glob("*.json")):
            match = json.loads(path.read_text(encoding="utf-8"))
            info = match["info"]
            outcome = info.get("outcome", {})
            teams = info.get("teams", [])
            if info.get("gender") not in (None, "male") or len(teams) != 2 or outcome.get("winner") not in teams:
                continue
            if "eliminator" in outcome or "bowl_out" in outcome:
                continue
            players: dict[str, dict] = {}
            for inn_no, innings in enumerate(match.get("innings", [])):
                if innings.get("super_over") or inn_no > 1:
                    continue
                team = innings.get("team")
                if team not in teams:
                    continue
                bowling_team = teams[1] if team == teams[0] else teams[0]
                for over in innings.get("overs", []):
                    for d in over.get("deliveries", []):
                        bat = players.setdefault(d["batter"], {"team": team, "bat_inn": inn_no + 1, "runs": 0, "balls": 0,
                                                               "bowl_inn": 0, "wkts": 0, "bowled": 0})
                        bat["bat_inn"] = inn_no + 1  # he may already exist as a bowler (his side bowled first)
                        bat["runs"] += d["runs"]["batter"]
                        if "wides" not in d.get("extras", {}):
                            bat["balls"] += 1
                        bowl = players.setdefault(d["bowler"], {"team": bowling_team, "bat_inn": 0, "runs": 0, "balls": 0,
                                                                "bowl_inn": 0, "wkts": 0, "bowled": 0})
                        bowl["bowl_inn"] = inn_no + 1
                        extras = d.get("extras", {})
                        if "wides" not in extras and "noballs" not in extras:
                            bowl["bowled"] += 1
                        bowl["wkts"] += sum(1 for w in d.get("wickets", []) if w.get("kind") not in NOT_BOWLER)
            for name, p in players.items():
                rows.append([info["dates"][0], name, p["team"], p["bat_inn"], p["runs"], p["balls"], p["bowl_inn"], p["wkts"],
                             p["bowled"], int(outcome["winner"] == p["team"]), source,
                             teams[1] if p["team"] == teams[0] else teams[0]])
    rows.sort(key=lambda r: r[0])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows))
    print(f"{len(rows)} player-matches -> {OUT}")


if __name__ == "__main__":
    main()
