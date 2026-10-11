"""Per-match facts for the conditions-based pre-match model.

For every men's T20I and IPL match in the local Cricsheet data, record how each
team batted against, and bowled with, three bowler groups: pace, finger spin
(off-break + left-arm orthodox) and wrist spin. Bowler style comes from
data/external/cricsheet_player_styles.csv (about 80% coverage; unknown bowlers
are skipped, not guessed). Also records the match's spin wicket share and scoring,
which the model uses (from earlier matches only) as the pitch-type signal.

Output: data/features/prematch_match_facts.json (rebuild any time; ~1 minute).
`python build_match_facts.py odi` builds the men's ODI version (ground records only) into
data/features/prematch_match_facts_odi.json; the T20 model never reads it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.ml.player_style_registry import PlayerStyleRegistry

ROOT = Path(__file__).resolve().parents[1]
ODI = len(sys.argv) > 1 and sys.argv[1] == "odi"
OUT = ROOT / "data" / "features" / ("prematch_match_facts_odi.json" if ODI else "prematch_match_facts.json")
GROUPS = {
    "left_arm_pace": "pace",
    "right_arm_pace": "pace",
    "right_arm_offbreak": "finger",
    "left_arm_orthodox": "finger",
    "wrist_spin": "wrist",
    "left_arm_wristspin": "wrist",
}
NOT_BOWLER_WICKETS = {"run out", "retired hurt", "retired out", "obstructing the field", "timed out"}


def _best_by_group(bowler_wkts: dict, group) -> dict:
    """Most wickets by a single pace bowler and by a single spinner in the match
    (finger + wrist spin = "spin"). Bowlers with no known style are skipped."""
    best = {"pace": 0, "spin": 0}
    for name, wickets in bowler_wkts.items():
        g = group(name)
        key = "pace" if g == "pace" else "spin" if g in ("finger", "wrist") else None
        if key:
            best[key] = max(best[key], wickets)
    return best


def main() -> None:
    registry = PlayerStyleRegistry()
    style_cache: dict[str, str | None] = {}

    def group(bowler: str) -> str | None:
        if bowler not in style_cache:
            style_cache[bowler] = GROUPS.get(registry.lookup(bowler)["bowling_style"])
        return style_cache[bowler]

    facts = []
    for source in (("odi",) if ODI else ("t20i", "ipl")):
        for path in sorted((ROOT / "data" / "raw" / "cricsheet" / source).glob("*.json")):
            match = json.loads(path.read_text(encoding="utf-8"))
            info = match["info"]
            if info.get("gender") not in (None, "male") or len(info.get("teams", [])) != 2:
                continue
            bat = {t: {g: [0, 0, 0] for g in ("pace", "finger", "wrist")} for t in info["teams"]}
            bowl = {t: {g: [0, 0, 0] for g in ("pace", "finger", "wrist")} for t in info["teams"]}
            totals = []
            innings_wkts = []
            innings_by_group = []  # per innings: wickets credited to pace / spin bowlers (known styles only)
            bowler_wkts: dict[str, int] = {}
            for innings in match.get("innings", []):
                team = innings.get("team")
                if team not in bat:
                    continue
                other = next(t for t in info["teams"] if t != team)
                total = 0
                fallen = 0
                taken = {"pace": 0, "spin": 0}
                for over in innings.get("overs", []):
                    for d in over.get("deliveries", []):
                        runs = d["runs"]
                        total += runs["total"]
                        fallen += len(d.get("wickets", []))
                        for w in d.get("wickets", []):
                            if w.get("kind") not in NOT_BOWLER_WICKETS:
                                bowler_wkts[d["bowler"]] = bowler_wkts.get(d["bowler"], 0) + 1
                                kind = group(d["bowler"])
                                if kind in ("finger", "wrist"):
                                    taken["spin"] += 1
                                elif kind == "pace":
                                    taken["pace"] += 1
                        g = group(d["bowler"])
                        if g is None or "wides" in d.get("extras", {}):
                            continue
                        legal = 0 if "noballs" in d.get("extras", {}) else 1
                        wicket = any(
                            w.get("kind") not in NOT_BOWLER_WICKETS for w in d.get("wickets", [])
                        )
                        for table, side in ((bat, team), (bowl, other)):
                            cell = table[side][g]
                            cell[0] += legal
                            cell[1] += runs["batter"]
                            cell[2] += int(wicket)
                totals.append(total)
                innings_wkts.append(fallen)
                innings_by_group.append(taken)
            outcome = info.get("outcome", {})
            facts.append(
                {
                    "src": source,
                    "date": info["dates"][0],
                    "city": (str(info.get("city") or "").strip().lower()
                             or str(info.get("venue") or "").rsplit(",", 1)[-1].strip().lower()),
                    "teams": info["teams"],
                    "winner": outcome.get("winner"),
                    "first_team": (match.get("innings") or [{}])[0].get("team"),
                    "totals": totals,
                    "innings_wkts": innings_wkts,
                    "innings_pace_wkts": [t["pace"] for t in innings_by_group],
                    "innings_spin_wkts": [t["spin"] for t in innings_by_group],
                    "best_by_group": _best_by_group(bowler_wkts, group),
                    "bat": bat,
                    "bowl": bowl,
                    "toss_winner": info.get("toss", {}).get("winner"),
                    "toss_decision": info.get("toss", {}).get("decision"),
                }
            )
    facts.sort(key=lambda f: f["date"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(facts))
    covered = sum(1 for f in facts for t in f["bat"] for g in f["bat"][t].values() if g[0])
    print(f"{len(facts)} matches -> {OUT}; team-group cells with data: {covered}")


if __name__ == "__main__":
    main()
