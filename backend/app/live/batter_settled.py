"""Fact-o-Meter for batters: "every time he has faced N balls he has gone on to X" -- 100% records with a BIG ask.

Per format (T20 = men's T20I + IPL, ODI = men's ODIs; never mixed), from every batter innings
(data/features/batter_innings[_odi].json, build_batter_innings.py [odi]). A batter has a record at (N balls, X runs)
when EVERY innings of his with N or more balls faced ended on X+, with
  * at least MIN_INNINGS such innings,
  * the league's own rate of reaching X from N balls at most 70% (not something everyone does), and
  * a fluke chance (that rate to the power n) of at most 1%.
Targets run up the scale (T20: 50/60/75/100; ODI: 50/75/100/125/150) and for each target the lowest N is kept.
It is SHOWN live only when the ask is large: he has faced his N balls and still needs at least ASK_MIN_RUNS more to
reach X (user 2026-10-11: "on 31 after 41 balls, always makes 50" is not exciting; a big innings still to come is).
Out of sample (record built from earlier innings, tested on his next N+-ball innings) the old 50-only records held
81% of 70 tests, so this is a record of the past, never a promise: the card says "n/n".
"""

from __future__ import annotations

import json
from functools import lru_cache

from app.live.fact_o_meter import MAX_BASELINE, card
from app.live.prematch_model import ROOT

PATHS = {"t20": ROOT / "data" / "features" / "batter_innings.json",           # men's T20I + IPL
         "odi": ROOT / "data" / "features" / "batter_innings_odi.json"}       # men's ODIs
PATH = PATHS["t20"]  # legacy name
BALL_THRESHOLDS = {"t20": tuple(range(10, 46, 5)), "odi": tuple(range(20, 101, 10))}
TARGETS = {"t20": (50, 60, 75, 100), "odi": (50, 75, 100, 125, 150)}
MIN_INNINGS = 6
MAX_FLUKE = 0.01
ASK_MIN_RUNS = 30


@lru_cache(maxsize=2)
def _records(fmt: str = "t20") -> dict[str, dict[int, tuple[int, int, float]]]:
    """batter -> {target X: (lowest N balls, innings, league rate)} for every qualifying record, per format."""
    try:
        rows = json.loads(PATHS[fmt].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    thresholds, targets = BALL_THRESHOLDS[fmt], TARGETS[fmt]
    per: dict[str, list[tuple[int, int]]] = {}
    league = {(n, x): [0, 0] for n in thresholds for x in targets}
    for _date, name, balls, runs, *_rest in rows:
        per.setdefault(name, []).append((balls, runs))
        for n in thresholds:
            if balls >= n:
                for x in targets:
                    league[(n, x)][0] += 1
                    league[(n, x)][1] += int(runs >= x)
    rate = {key: (hit / seen if seen else 0.0) for key, (seen, hit) in league.items()}
    out: dict[str, dict[int, tuple[int, int, float]]] = {}
    for name, innings in per.items():
        for x in targets:
            for n in thresholds:
                picked = [runs for balls, runs in innings if balls >= n]
                if len(picked) < MIN_INNINGS or min(picked) < x:
                    continue
                if rate[(n, x)] <= MAX_BASELINE and rate[(n, x)] ** len(picked) <= MAX_FLUKE:
                    out.setdefault(name, {})[x] = (n, len(picked), rate[(n, x)])
                    break
    return out


def batter_records(cricsheet_name: str, fmt: str = "t20") -> dict[int, tuple[int, int, float]]:
    """{target X: (N balls, innings, league rate)} for the batter in that format (empty if none)."""
    return dict(_records(fmt).get(cricsheet_name, {}))


def all_records(fmt: str = "t20") -> dict[str, dict[int, tuple[int, int, float]]]:
    return {name: dict(claims) for name, claims in _records(fmt).items()}


def _what(x: int) -> str:
    return {50: "a FIFTY", 100: "a HUNDRED", 150: "150"}.get(x, f"{x}+ runs")


def settled_line(full_name: str, team: str, balls: int, runs: int, fmt: str = "t20") -> str:
    """The Fact-o-Meter card for the batter's biggest record that applies right now (he has faced its N balls and
    still needs ASK_MIN_RUNS+ to reach X), else ""."""
    from app.live.prematch_xi import resolve_player

    name = resolve_player(full_name, team)
    claims = batter_records(name, fmt) if name else {}
    live = [(x, n, innings, base) for x, (n, innings, base) in claims.items() if balls >= n and x - runs >= ASK_MIN_RUNS]
    if not live:
        return ""
    x, n, innings, base = max(live)
    return card(fmt, f"{full_name} {runs}*({balls}) -- needs {x - runs} more",
                f"Every single time he has faced {n}+ balls, he has gone on to make {_what(x)}", innings, base, "batter")
