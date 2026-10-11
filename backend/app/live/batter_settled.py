"""The commentator's "every time he gets to 40 balls he makes a fifty" stat: only 100% records, never a rate.

Per format (T20 = men's T20I + IPL, ODI = men's ODIs; never mixed): from every batter innings (data/features/batter_innings[_odi].json, build_batter_innings.py [odi]) a batter
has a "50 record" at N balls when EVERY innings of his in which he faced N or more balls ended on 50+, with
  * at least MIN_INNINGS such innings,
  * a big gap: the league's own rate of reaching 50 from N balls is at most 70% (so "50 after 40 balls", which
    83% of batters manage, is not a fact), and
  * the record not being a fluke: an average batter's chance of going that many out of that many (the league's
    own rate of reaching 50 from N balls, to the power n) is at most MAX_FLUKE (1%).
His claim is the lowest such N. Checked out of sample 2026-10-11 (record built from earlier innings, tested on
the next one he faced N+ balls in): such records held 81% of the time (70 tests; 88% / 88% / 91% at 30 / 35 / 40
balls, no better than the league at 20-25). A record is a fact about the past, not a guarantee -- the line
says "N of N innings". With these limits only a handful of batters qualify at any time (4 on 2026-10-11).
There is no 15-ball record: no batter has gone 6-for-6 on fifties from 15 balls.
"""

from __future__ import annotations

import json
from functools import lru_cache

from app.live.fact_o_meter import MAX_BASELINE, card
from app.live.prematch_model import ROOT

PATHS = {"t20": ROOT / "data" / "features" / "batter_innings.json",           # men's T20I + IPL
         "odi": ROOT / "data" / "features" / "batter_innings_odi.json"}       # men's ODIs
PATH = PATHS["t20"]  # legacy name
# Ball counts tried per format (a fifty comes from a longer stay in an ODI): the lowest that qualifies is the claim.
BALL_THRESHOLDS = {"t20": tuple(range(10, 46, 5)), "odi": tuple(range(20, 101, 10))}
TARGET = 50
MIN_INNINGS = 6
MAX_FLUKE = 0.01


@lru_cache(maxsize=2)
def _records(fmt: str = "t20") -> dict[str, tuple[int, int, float]]:
    """batter -> (N, innings, league rate) for the lowest N at which his 50 record qualifies, per format."""
    try:
        rows = json.loads(PATHS[fmt].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    per: dict[str, list[tuple[int, int]]] = {}
    thresholds = BALL_THRESHOLDS[fmt]
    league = {n: [0, 0] for n in thresholds}
    for _date, name, balls, runs, *_rest in rows:
        per.setdefault(name, []).append((balls, runs))
        for n in thresholds:
            if balls >= n:
                league[n][0] += 1
                league[n][1] += int(runs >= TARGET)
    rate = {n: (hit / seen if seen else 0.0) for n, (seen, hit) in league.items()}
    out: dict[str, tuple[int, int, float]] = {}
    for name, innings in per.items():
        for n in thresholds:
            picked = [runs for balls, runs in innings if balls >= n]
            if (len(picked) >= MIN_INNINGS and min(picked) >= TARGET and rate[n] <= MAX_BASELINE
                    and rate[n] ** len(picked) <= MAX_FLUKE):
                out[name] = (n, len(picked), rate[n])
                break
    return out


def fifty_record(cricsheet_name: str, fmt: str = "t20") -> tuple[int, int, float] | None:
    """(N balls, innings, league rate) of the batter's 100% fifty record in that format, or None."""
    return _records(fmt).get(cricsheet_name)


def all_records(fmt: str = "t20") -> dict[str, tuple[int, int, float]]:
    return dict(_records(fmt))


def settled_line(full_name: str, team: str, balls: int, runs: int, fmt: str = "t20") -> str:
    """"Fact-o-Meter (T20): Suryakumar Yadav 31*(42) -- 50+ in 26 of 26 innings where he has faced 40+ balls" for a batter with a
    100% record who has reached its ball count and is still short of 50, else ""."""
    if runs >= TARGET:
        return ""
    from app.live.prematch_xi import resolve_player

    name = resolve_player(full_name, team)
    record = fifty_record(name, fmt) if name else None
    if not record or balls < record[0]:
        return ""
    n, innings, baseline = record
    return card(fmt, f"{full_name} {runs}*({balls})",
                f"Every single time he has faced {n}+ balls, he has made a FIFTY", innings, baseline, "batter")
