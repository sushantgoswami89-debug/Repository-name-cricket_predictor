"""Bowler "100% record" stat: he took a wicket (or two) in EVERY innings in which he bowled N or more balls.

From every men's T20I + IPL bowler innings (data/features/bowler_innings.json, build_bowler_innings.py). A claim is
"wickets >= k in every innings with N+ legal balls" (k = 1 or 2, N = 6/12/18/24 balls = 1-4 overs), needing at least
MIN_INNINGS such innings and a fluke chance <= MAX_FLUKE (the league's own rate of taking k+ wickets from N+ balls,
to the power n). The claim with the smallest fluke chance is kept. Out-of-sample 2026-10-11 (record built from
earlier innings, tested on his next qualifying innings): held 70% of 138 tests vs 61% for the league -- a modest edge,
so the line states the record ("N of N innings"), not a prediction. Nine bowlers qualified on 2026-10-11.
"""

from __future__ import annotations

import json
from functools import lru_cache

from app.live.prematch_model import ROOT

PATH = ROOT / "data" / "features" / "bowler_innings.json"
BALL_THRESHOLDS = (6, 12, 18, 24)
MIN_INNINGS = 6
MAX_FLUKE = 0.01


@lru_cache(maxsize=1)
def _records() -> dict[str, tuple[int, int, int]]:
    """bowler -> (wickets k, balls N, innings n) for the claim with the smallest fluke chance."""
    try:
        rows = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    per: dict[str, list[tuple[int, int]]] = {}
    league = {(n, k): [0, 0] for n in BALL_THRESHOLDS for k in (1, 2)}
    for _date, name, balls, _runs, wickets, *_rest in rows:
        per.setdefault(name, []).append((balls, wickets))
        for n in BALL_THRESHOLDS:
            if balls >= n:
                for k in (1, 2):
                    league[(n, k)][0] += 1
                    league[(n, k)][1] += int(wickets >= k)
    rate = {key: (hit / seen if seen else 0.0) for key, (seen, hit) in league.items()}
    out: dict[str, tuple[int, int, int]] = {}
    for name, innings in per.items():
        best = None
        for k in (1, 2):
            for n in BALL_THRESHOLDS:
                picked = [w for balls, w in innings if balls >= n]
                if len(picked) < MIN_INNINGS or min(picked) < k:
                    continue
                fluke = rate[(n, k)] ** len(picked)
                if fluke <= MAX_FLUKE and (best is None or fluke < best[0]):
                    best = (fluke, k, n, len(picked))
        if best:
            out[name] = best[1:]
    return out


def all_records() -> dict[str, tuple[int, int, int]]:
    return dict(_records())


def overs_to_balls(overs) -> int | None:
    """"3.2" / 3.2 -> 20 legal balls; None if it cannot be read."""
    try:
        whole, _, part = str(overs).partition(".")
        return int(whole) * 6 + int(part or 0)
    except ValueError:
        return None


def bowler_line(full_name: str, team: str, balls: int, wickets: int) -> str:
    """"STAT: J Coetzee 3.0-0-22-0 ..." text for a bowler with a 100% record who has bowled his record's ball count
    without yet reaching its wicket count, else ""."""
    from app.live.prematch_xi import resolve_player

    name = resolve_player(full_name, team)
    record = _records().get(name) if name else None
    if not record:
        return ""
    k, n, innings = record
    if balls < n or wickets >= k:
        return ""
    what = "a wicket" if k == 1 else f"{k}+ wickets"
    spell = f"{n // 6}+ overs" if n % 6 == 0 else f"{n}+ balls"
    return f"STAT: {full_name} ({balls // 6}.{balls % 6} ov, {wickets} wkts) -- took {what} in {innings} of {innings} innings where he bowled {spell}"
