"""Bowler "100% record" stat: he took a wicket (or two) in EVERY innings in which he bowled N or more balls.

Per format (T20 = all men's T20 (internationals + IPL, BBL, PSL, CPL, ...), ODI = men's ODIs; never mixed): from every bowler innings (data/features/bowler_innings[_odi].json, build_bowler_innings.py [odi]). A claim is
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

PATHS = {"t20": ROOT / "data" / "features" / "bowler_innings.json",           # all men's T20 (internationals + IPL, BBL, PSL, CPL, ...)
         "odi": ROOT / "data" / "features" / "bowler_innings_odi.json"}       # men's ODIs
PATH = PATHS["t20"]  # legacy name
BALL_THRESHOLDS = {"t20": (6, 12, 18, 24), "odi": (12, 24, 36, 48, 60)}  # 1-4 overs / 2-10 overs
MIN_INNINGS = 6
MAX_FLUKE = 0.01
ASK_MIN_WICKETS = 2  # shown only when he still needs this many wickets to keep the record (same "big ask" idea as batters)


@lru_cache(maxsize=2)
def _records(fmt: str = "t20") -> dict[str, list[tuple[int, int, int, float, str]]]:
    """bowler -> [(wickets k, balls N, innings n, league rate, opponent)]: the best (smallest fluke chance) claim overall
    and the best against each single opponent ("" = overall; an opponent record holds only against that team)."""
    try:
        rows = json.loads(PATHS[fmt].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    per: dict[str, dict[str, list[tuple[int, int]]]] = {}
    thresholds = BALL_THRESHOLDS[fmt]
    league = {(n, k): [0, 0] for n in thresholds for k in (1, 2)}
    for _date, name, balls, _runs, wickets, _src, opponent, *_city in rows:
        per.setdefault(name, {}).setdefault("", []).append((balls, wickets))
        if opponent:
            per[name].setdefault(opponent, []).append((balls, wickets))
        for n in thresholds:
            if balls >= n:
                for k in (1, 2):
                    league[(n, k)][0] += 1
                    league[(n, k)][1] += int(wickets >= k)
    rate = {key: (hit / seen if seen else 0.0) for key, (seen, hit) in league.items()}
    out: dict[str, list[tuple[int, int, int, float, str]]] = {}
    for name, by_opponent in per.items():
        for opponent, innings in by_opponent.items():
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
                out.setdefault(name, []).append((*best[1:], opponent))
    return out


def all_records(fmt: str = "t20") -> dict[str, list[tuple[int, int, int, float, str]]]:
    return {name: list(claims) for name, claims in _records(fmt).items()}


def overs_to_balls(overs) -> int | None:
    """"3.2" / 3.2 -> 20 legal balls; None if it cannot be read."""
    try:
        whole, _, part = str(overs).partition(".")
        return int(whole) * 6 + int(part or 0)
    except ValueError:
        return None


def bowler_line(full_name: str, team: str, balls: int, wickets: int, fmt: str = "t20", opponent: str = "") -> str:
    """The Fact-o-Meter card for a bowler with a 100% record that applies now: he has bowled its ball count and still
    needs ASK_MIN_WICKETS+ wickets to reach its wicket count (an opponent record applies only against that team), else ""."""
    from app.live.prematch_xi import resolve_player

    name = resolve_player(full_name, team)
    live = [c for c in (_records(fmt).get(name, []) if name else [])
            if c[4] in ("", opponent) and balls >= c[1] and c[0] - wickets >= ASK_MIN_WICKETS]
    if not live:
        return ""
    k, n, innings, _baseline, opp = max(live, key=lambda c: (c[0], c[4] != ""))
    what = "a wicket" if k == 1 else f"{k}+ wickets"
    spell = f"{n // 6}+ overs" if n % 6 == 0 else f"{n}+ balls"
    vs = f" vs {opp}" if opp else ""
    return card(fmt, f"{full_name} {balls // 6}.{balls % 6} ov, {wickets} wkt{'' if wickets == 1 else 's'}",
                f"Has taken {what} in every one of his innings{vs} of {spell}", innings)
