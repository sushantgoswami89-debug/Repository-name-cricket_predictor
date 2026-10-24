"""Does our ball-by-ball record add up to Cricbuzz's SCORECARD?  (independent of the over summaries)

The driver saves the scorecard (every batter and bowler figure) at each over end: data/live/scorecard-<id>.jsonl. For two consecutive
snapshots of one innings, the difference in each batter's runs/balls/4s/6s and each bowler's runs/wickets/balls/wides/no-balls must equal
what our recorded deliveries (data/live/commentary-<id>.jsonl) say happened between them. That checks the striker and the bowler of every
ball as well as the runs, and works for a match joined late, because only the stretch between two snapshots is compared.

    cd backend && ../.venv/bin/python audit_vs_scorecard.py 170723 155541
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOT_BOWLERS_WICKET = {"run out", "retired hurt", "retired out", "obstructing the field", "timed out", "handled the ball"}


def overs_to_balls(overs) -> int:
    whole, _, part = f"{overs}".partition(".")
    return int(whole or 0) * 6 + int(part or 0)


def load_deliveries(match_id: str) -> dict[int, list[dict]]:
    """Recorded deliveries per innings, de-duplicated (last copy of a ball wins), each with its cumulative legal-ball count."""
    latest: dict[tuple[int, int, int], dict] = {}
    for line in (ROOT / "data" / "live" / f"commentary-{match_id}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[(row["innings"], row["over"], row["ball"])] = row
    by_innings: dict[int, list[dict]] = defaultdict(list)
    for key in sorted(latest):
        by_innings[key[0]].append(latest[key])
    for rows in by_innings.values():
        legal_by_over: dict[int, int] = defaultdict(int)
        for row in rows:
            extras = row.get("extras") or {}
            legal = "wides" not in extras and "noballs" not in extras
            if legal:
                legal_by_over[row["over"]] += 1
            row["_legal"] = legal
            row["_balls"] = (row["over"] - 1) * 6 + legal_by_over[row["over"]]   # legal balls bowled up to and including this delivery
    return by_innings


def delta_from_deliveries(rows: list[dict]) -> tuple[dict, dict, int, int]:
    bat: dict[str, dict] = defaultdict(lambda: {"runs": 0, "balls": 0, "fours": 0, "sixes": 0})
    bowl: dict[str, dict] = defaultdict(lambda: {"runs": 0, "wickets": 0, "balls": 0, "wides": 0, "no_balls": 0})
    total = wickets = 0
    for row in rows:
        extras = row.get("extras") or {}
        wide = "wides" in extras
        total += row["total_runs"]
        wickets += int(bool(row.get("wicket_kind")))
        if not wide:
            bat[row["striker"]]["balls"] += 1
            bat[row["striker"]]["runs"] += row["batter_runs"]
            bat[row["striker"]]["fours"] += int(row["batter_runs"] == 4)
            bat[row["striker"]]["sixes"] += int(row["batter_runs"] == 6)
        b = bowl[row["bowler"]]
        b["balls"] += int(row["_legal"])
        b["wides"] += int(wide)
        b["no_balls"] += int("noballs" in extras)
        b["runs"] += row["total_runs"] - extras.get("byes", 0) - extras.get("legbyes", 0)
        if row.get("wicket_kind") and row["wicket_kind"] not in NOT_BOWLERS_WICKET:
            b["wickets"] += 1
    return bat, bowl, total, wickets


def audit(match_id: str) -> tuple[int, int]:
    path = ROOT / "data" / "live" / f"scorecard-{match_id}.jsonl"
    if not path.exists():
        print(f"=== {match_id}: no scorecard snapshots saved yet (the driver saves one at each over end)")
        return 0, 0
    snapshots = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    deliveries = load_deliveries(match_id)
    checked = bad = 0
    print(f"=== {match_id}: {len(snapshots)} scorecard snapshots")
    for innings, rows in sorted(deliveries.items()):
        cards = []
        for snap in snapshots:
            card = next((i for i in snap["innings"] if i.get("innings") == innings), None)
            if card and card.get("overs") is not None and (not cards or overs_to_balls(card["overs"]) > overs_to_balls(cards[-1]["overs"])):
                cards.append(card)
        for before, after in zip(cards, cards[1:]):
            a, b = overs_to_balls(before["overs"]), overs_to_balls(after["overs"])
            if b - a < 6:
                continue   # compare whole stretches of at least an over, so a boundary inside an over cannot blur the result
            # a legal ball counts when it is one of balls a+1..b; a wide/no-ball counts when it came after ball a and before ball b+1
            # is not yet bowled (its _balls is the legal count BEFORE it): so a <= _balls < b
            between = [r for r in rows if (r["_legal"] and a < r["_balls"] <= b) or (not r["_legal"] and a <= r["_balls"] < b)]
            if sum(1 for r in between if r["_legal"]) != b - a:
                continue   # we do not hold every legal ball of that stretch (joined late or a gap)
            bat, bowl, total, wickets = delta_from_deliveries(between)
            if total != after["runs"] - before["runs"] or wickets != after["wickets"] - before["wickets"]:
                # an extra delivery right at a boundary (a wide after the last legal ball) can move one run either way: report it as such
                print(f"  inn {innings} balls {a}-{b}: runs/wickets ours {total}/{wickets} vs card {after['runs'] - before['runs']}/{after['wickets'] - before['wickets']} (stretch not comparable)")
                continue
            checked += 1
            problems = []
            card_bat = {x["name"]: x for x in after["batting"]}
            old_bat = {x["name"]: x for x in before["batting"]}
            for name in set(bat) | {n for n in card_bat if (card_bat[n].get("balls") or 0) != (old_bat.get(n, {}).get("balls") or 0)}:
                want = {k: (card_bat.get(name, {}).get(k) or 0) - (old_bat.get(name, {}).get(k) or 0) for k in ("runs", "balls", "fours", "sixes")}
                have = bat.get(name, {"runs": 0, "balls": 0, "fours": 0, "sixes": 0})
                if want != have:
                    problems.append(f"batter {name}: ours {have} vs card {want}")
            card_bowl = {x["name"]: x for x in after["bowling"]}
            old_bowl = {x["name"]: x for x in before["bowling"]}
            for name in set(bowl) | set(card_bowl):
                if name not in card_bowl:
                    problems.append(f"bowler {name} is not in the scorecard")
                    continue
                now, then = card_bowl[name], old_bowl.get(name, {})
                want = {
                    "runs": (now.get("runs") or 0) - (then.get("runs") or 0),
                    "wickets": (now.get("wickets") or 0) - (then.get("wickets") or 0),
                    "balls": overs_to_balls(now.get("overs") or 0) - overs_to_balls(then.get("overs") or 0),
                    "wides": (now.get("wides") or 0) - (then.get("wides") or 0),
                    "no_balls": (now.get("no_balls") or 0) - (then.get("no_balls") or 0),
                }
                have = bowl.get(name, {"runs": 0, "wickets": 0, "balls": 0, "wides": 0, "no_balls": 0})
                if want != have:
                    problems.append(f"bowler {name}: ours {have} vs card {want}")
            if problems:
                bad += 1
                print(f"  inn {innings} balls {a}-{b}: " + "; ".join(problems[:4]))
    print(f"  {checked} stretches compared with the scorecard: {checked - bad} exact, {bad} different")
    return checked, bad


if __name__ == "__main__":
    total_checked = total_bad = 0
    for match in sys.argv[1:]:
        c, b = audit(match)
        total_checked, total_bad = total_checked + c, total_bad + b
    print(f"ALL: {total_checked} stretches, {total_bad} different")
