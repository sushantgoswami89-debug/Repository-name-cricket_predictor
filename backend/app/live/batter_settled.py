"""The commentator's "once he has faced 15 balls he usually goes on to a fifty" stat, from the data.

From every men's T20I + IPL batter innings (data/features/batter_innings.json, build_batter_innings.py):
among a batter's innings in which he faced 15+ balls, how often did he finish with 30+ and 50+?
Checked 2026-10-11 out of sample (2022+ innings, rates built only from earlier ones): the player's own
rate beats the league rate (22% reach 50 once settled) -- Brier 0.1617 vs 0.1686 on all 14,729 innings and
0.1963 vs 0.2074 for batters with 10+ earlier 15-ball innings; batters flagged at 45%+ then went on to 50
in 47% of their next 15-ball innings. So it is real but not "always": the line shows the count.

Shown only for batters whose rate is at least 1.5x the league's with enough innings to mean something.
"""

from __future__ import annotations

import json
from functools import lru_cache

from app.live.prematch_model import ROOT

PATH = ROOT / "data" / "features" / "batter_innings.json"
CHECKPOINT = 15
MIN_INNINGS = 10
MIN_RATIO = 1.5


@lru_cache(maxsize=1)
def _table() -> tuple[dict, float]:
    try:
        rows = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, 0.0
    per: dict[str, list[int]] = {}
    total = hit = 0
    for _date, name, balls, runs, *_rest in rows:
        if balls < CHECKPOINT:
            continue
        cell = per.setdefault(name, [0, 0, 0])
        cell[0] += 1
        cell[1] += int(runs >= 30)
        cell[2] += int(runs >= 50)
        total += 1
        hit += int(runs >= 50)
    return per, (hit / total if total else 0.0)


def league_rate() -> float:
    return _table()[1]


def settled_record(cricsheet_name: str) -> dict | None:
    """{"innings": n, "thirty": k30, "fifty": k50, "rate50": ..} for innings of 15+ balls, or None."""
    cell = _table()[0].get(cricsheet_name)
    if not cell:
        return None
    return {"innings": cell[0], "thirty": cell[1], "fifty": cell[2], "rate50": cell[2] / cell[0]}


def settled_line(full_name: str, team: str, balls: int, runs: int) -> str:
    """"STAT: Shai Hope 38*(19) -- reached 50 in 24 of 41 innings (59%) where he faced 15+ balls; average batter 22%"
    for a converter who has faced 15+ balls and is still short of 50, else ""."""
    if balls < CHECKPOINT or runs >= 50:
        return ""
    from app.live.prematch_xi import resolve_player

    name = resolve_player(full_name, team)
    record = settled_record(name) if name else None
    base = league_rate()
    if not record or record["innings"] < MIN_INNINGS or base <= 0 or record["rate50"] < MIN_RATIO * base:
        return ""
    return (
        f"STAT: {full_name} {runs}*({balls}) -- reached 50 in {record['fifty']} of {record['innings']} innings "
        f"({record['rate50']:.0%}) where he faced {CHECKPOINT}+ balls; average batter {base:.0%}"
    )
