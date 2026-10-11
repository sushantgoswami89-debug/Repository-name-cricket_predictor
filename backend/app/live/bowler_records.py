"""Bowler "100% record" stat: he took a wicket (or two) in EVERY innings in which he bowled N or more balls.

Per format (T20 = men's T20I + IPL, ODI = men's ODIs; never mixed): from every bowler innings (data/features/bowler_innings[_odi].json, build_bowler_innings.py [odi]). A claim is
"wickets >= k in every innings with N+ legal balls" (k = 1 or 2, N = 6/12/18/24 balls = 1-4 overs), needing at least
MIN_INNINGS such innings and a fluke chance <= MAX_FLUKE (the league's own rate of taking k+ wickets from N+ balls,
to the power n). The claim with the smallest fluke chance is kept. Out-of-sample 2026-10-11 (record built from
earlier innings, tested on his next qualifying innings): held 70% of 138 tests vs 61% for the league -- a modest edge,
so the line states the record ("N of N innings"), not a prediction. Nine bowlers qualified on 2026-10-11.
"""

from __future__ import annotations

import json
from functools import lru_cache

from app.live.fact_o_meter import MAX_BASELINE, card
from app.live.prematch_model import ROOT

PATHS = {"t20": ROOT / "data" / "features" / "bowler_innings.json",           # men's T20I + IPL
         "odi": ROOT / "data" / "features" / "bowler_innings_odi.json"}       # men's ODIs
PATH = PATHS["t20"]  # legacy name
BALL_THRESHOLDS = {"t20": (6, 12, 18, 24), "odi": (12, 24, 36, 48, 60)}  # 1-4 overs / 2-10 overs
MIN_INNINGS = 6
MAX_FLUKE = 0.01
ASK_MIN_WICKETS = 2  # shown only when he still needs this many wickets to keep the record (same "big ask" idea as batters)


@lru_cache(maxsize=2)
def _records(fmt: str = "t20") -> dict[str, tuple[int, int, int, float]]:
    """bowler -> (wickets k, balls N, innings n, league rate) for the claim with the smallest fluke chance."""
    try:
        rows = json.loads(PATHS[fmt].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    per: dict[str, list[tuple[int, int]]] = {}
    thresholds = BALL_THRESHOLDS[fmt]
    league = {(n, k): [0, 0] for n in thresholds for k in (1, 2)}
    for _date, name, balls, _runs, wickets, *_rest in rows:
        per.setdefault(name, []).append((balls, wickets))
        for n in thresholds:
            if balls >= n:
                for k in (1, 2):
                    league[(n, k)][0] += 1
                    league[(n, k)][1] += int(wickets >= k)
    rate = {key: (hit / seen if seen else 0.0) for key, (seen, hit) in league.items()}
    out: dict[str, tuple[int, int, int, float]] = {}
    for name, innings in per.items():
        best = None
        for k in (1, 2):
            for n in thresholds:
                picked = [w for balls, w in innings if balls >= n]
                if len(picked) < MIN_INNINGS or min(picked) < k or rate[(n, k)] > MAX_BASELINE:
                    continue
                fluke = rate[(n, k)] ** len(picked)
                if fluke <= MAX_FLUKE and (best is None or fluke < best[0]):
                    best = (fluke, k, n, len(picked), rate[(n, k)])
        if best:
            out[name] = best[1:]
    return out


def all_records(fmt: str = "t20") -> dict[str, tuple[int, int, int, float]]:
    return dict(_records(fmt))


def overs_to_balls(overs) -> int | None:
    """"3.2" / 3.2 -> 20 legal balls; None if it cannot be read."""
    try:
        whole, _, part = str(overs).partition(".")
        return int(whole) * 6 + int(part or 0)
    except ValueError:
        return None


def bowler_line(full_name: str, team: str, balls: int, wickets: int, fmt: str = "t20") -> str:
    """"Fact-o-Meter (T20): J Coetzee 3.0-0-22-0 ..." text for a bowler with a 100% record who has bowled his record's ball count
    and still needs ASK_MIN_WICKETS+ wickets to reach its wicket count, else ""."""
    from app.live.prematch_xi import resolve_player

    name = resolve_player(full_name, team)
    record = _records(fmt).get(name) if name else None
    if not record:
        return ""
    k, n, innings, baseline = record
    if balls < n or k - wickets < ASK_MIN_WICKETS:
        return ""
    what = "a wicket" if k == 1 else f"{k}+ wickets"
    spell = f"{n // 6}+ overs" if n % 6 == 0 else f"{n}+ balls"
    return card(fmt, f"{full_name} {balls // 6}.{balls % 6} ov, {wickets} wkt{'' if wickets == 1 else 's'}",
                f"Has taken {what} in every one of his innings of {spell}", innings)
