"""Fact-o-Meter at a ground: "his favourite ground -- every time he has batted first here he has made 50+" (and bowlers).

Per format (T20 = every men's T20 competition, ODI = men's ODIs; never mixed), for ONE ground (the city key the ground
records use) and one innings (1st, 2nd, or either):
  * batters: in ALL n innings he scored X+ (T20 30/50/75/100, ODI 50/75/100/150);
  * bowlers: in ALL n innings (at least an over) he took k+ wickets (2/3/4).
A record needs MIN_INNINGS (4) and a fluke chance of at most 0.1% (the league's per-innings rate of doing that, to the
power n -- stricter than the other facts because thousands of batter-ground pairs are searched). The card carries the
detail the user asked for: how many, his average runs / wickets there and his best.
Honest numbers (checked 2026-10-11, record from earlier innings, tested on his next at that ground): batter ground
records held 40-45% of the time in T20 (league 9-21%), ODI too few to say; bowlers 21-27% (T20, 14 tests). A batter's
average at a ground adds almost nothing beyond his career average (weight 0.12 in T20, 0.00 in ODI). So these are true
records of the past, not predictions -- the card says "n of n".
"""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache

from app.live.fact_o_meter import card, ratio_fill
from app.live.prematch_model import ROOT

BAT_PATHS = {"t20": ROOT / "data" / "features" / "batter_innings.json", "odi": ROOT / "data" / "features" / "batter_innings_odi.json"}
BOWL_PATHS = {"t20": ROOT / "data" / "features" / "bowler_innings.json", "odi": ROOT / "data" / "features" / "bowler_innings_odi.json"}
BAT_TARGETS = {"t20": (30, 50, 75, 100), "odi": (50, 75, 100, 150)}
BOWL_TARGETS = (2, 3, 4)
MIN_INNINGS = 4
MAX_FLUKE = 0.001
ASK_MIN_RUNS = 20
ASK_MIN_WICKETS = 2


def _load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


@lru_cache(maxsize=2)
def _batter_claims(fmt: str = "t20") -> dict[tuple[str, str, int], dict]:
    """(batter, city, innings 0/1/2) -> {x, n, avg, best, fluke} for his strongest record there."""
    rows = _load(BAT_PATHS[fmt])
    targets = BAT_TARGETS[fmt]
    league = {x: [0, 0] for x in targets}
    per = defaultdict(list)
    for _date, name, _balls, runs, _out, _at15, _to50, _src, inn, _opp, city in rows:
        per[(name, city, 0)].append(runs)
        per[(name, city, inn)].append(runs)
        for x in targets:
            league[x][0] += 1
            league[x][1] += int(runs >= x)
    rate = {x: league[x][1] / league[x][0] for x in targets if league[x][0]}
    out = {}
    for key, runs in per.items():
        if len(runs) < MIN_INNINGS:
            continue
        best = None
        for x in targets:
            if min(runs) >= x and rate[x] ** len(runs) <= MAX_FLUKE and (best is None or x > best["x"]):
                best = {"x": x, "n": len(runs), "avg": sum(runs) / len(runs), "best": max(runs), "fluke": rate[x] ** len(runs)}
        if best:
            out[key] = best
    return out


@lru_cache(maxsize=2)
def _bowler_claims(fmt: str = "t20") -> dict[tuple[str, str, int], dict]:
    rows = _load(BOWL_PATHS[fmt])
    league = {k: [0, 0] for k in BOWL_TARGETS}
    per = defaultdict(list)
    for _date, name, balls, _runs, wickets, _src, _opp, city, inn in rows:
        if balls < 6:
            continue
        per[(name, city, 0)].append(wickets)
        per[(name, city, inn)].append(wickets)
        for k in BOWL_TARGETS:
            league[k][0] += 1
            league[k][1] += int(wickets >= k)
    rate = {k: league[k][1] / league[k][0] for k in BOWL_TARGETS if league[k][0]}
    out = {}
    for key, wickets in per.items():
        if len(wickets) < MIN_INNINGS:
            continue
        best = None
        for k in BOWL_TARGETS:
            if min(wickets) >= k and rate[k] ** len(wickets) <= MAX_FLUKE and (best is None or k > best["x"]):
                best = {"x": k, "n": len(wickets), "avg": sum(wickets) / len(wickets), "best": max(wickets), "fluke": rate[k] ** len(wickets)}
        if best:
            out[key] = best
    return out


def all_batter_claims(fmt: str = "t20") -> dict:
    return {k: dict(v) for k, v in _batter_claims(fmt).items()}


def all_bowler_claims(fmt: str = "t20") -> dict:
    return {k: dict(v) for k, v in _bowler_claims(fmt).items()}


def _innings_words(inn: int, kind: str) -> str:
    if kind == "bat":
        return {0: "", 1: " first innings", 2: " second innings"}[inn]
    return {0: "", 1: " bowling first", 2: " bowling second"}[inn]


def batter_ground_line(full_name: str, resolved: str | None, city: str, innings: int, runs: int, balls: int,
                       fmt: str = "t20") -> str:
    """The card for his strongest ground record that applies in this innings while he still needs ASK_MIN_RUNS+."""
    if not resolved or not city:
        return ""
    live = [{**claim, "inn": inn} for inn in (0, innings)
            if (claim := _batter_claims(fmt).get((resolved, city, inn))) and claim["x"] - runs >= ASK_MIN_RUNS]
    if not live:
        return ""
    c = max(live, key=lambda c: (c["x"], c["inn"] != 0))
    where = city.title()
    when = {0: "every innings", 1: "every first innings", 2: "every second innings"}[c["inn"]]
    fact = (f"His favourite ground: {c['x']}+ in {when} at {where}, {c['n']} of {c['n']} "
            f"(avg {c['avg']:.0f}, best {c['best']})")
    return card(fmt, f"{full_name} {runs}*({balls})", fact, c["n"])


def bowler_ground_line(full_name: str, resolved: str | None, city: str, innings: int, wickets: int, balls: int,
                       fmt: str = "t20") -> str:
    """The card for his strongest ground record that applies now (he still needs ASK_MIN_WICKETS+ wickets)."""
    if not resolved or not city:
        return ""
    live = [{**c, "inn": inn} for inn in (0, innings) if (c := _bowler_claims(fmt).get((resolved, city, inn)))
            and c["x"] - wickets >= ASK_MIN_WICKETS]
    if not live:
        return ""
    c = max(live, key=lambda c: (c["x"], c["inn"] != 0))
    when = {0: "every spell", 1: "every time he bowls first", 2: "every time he bowls second"}[c["inn"]]
    fact = f"Loves this ground: {c['x']}+ wickets {when} at {city.title()}, {c['n']} of {c['n']} (avg {c['avg']:.1f} wkts)"
    return card(fmt, f"{full_name} {balls // 6}.{balls % 6} ov, {wickets} wkt{'' if wickets == 1 else 's'}", fact, c["n"])


# ---------------------------------------------------------------------------------------------------------------
# Average-gap facts: not a 100% record, but a big gap between his average at this ground and his own career average.
# A batter / bowler needs at least AVG_MIN_INNINGS there, an average at least AVG_RATIO times his career average and a
# z-score of AVG_Z or more (gap / (his career spread / sqrt(n))). Checked 2026-10-11 out of sample: the next innings
# comes back most of the way to his career average (T20 batters: claimed 46.9, next innings 27.5 vs career 23.2), so the
# card gives the numbers and never a forecast.
AVG_MIN_INNINGS = 5
AVG_RATIO = 1.5
AVG_Z = 3.0
AVG_MIN_CAREER = 20


@lru_cache(maxsize=2)
def _average_claims(fmt: str = "t20", kind: str = "bat") -> dict[tuple[str, str], dict]:
    """(player, city) -> {n, avg, career, best, ratio} for the grounds where he is far above his own level."""
    import math

    if kind == "bat":
        rows = [(r[1], r[10], r[3]) for r in _load(BAT_PATHS[fmt])]
    else:
        rows = [(r[1], r[7], r[4]) for r in _load(BOWL_PATHS[fmt]) if r[2] >= 6]
    career = defaultdict(list)
    ground = defaultdict(list)
    for name, city, value in rows:
        career[name].append(value)
        ground[(name, city)].append(value)
    out = {}
    for (name, city), values in ground.items():
        base = career[name]
        if len(values) < AVG_MIN_INNINGS or len(base) < AVG_MIN_CAREER:
            continue
        mean_base = sum(base) / len(base)
        sd = math.sqrt(sum((v - mean_base) ** 2 for v in base) / len(base)) or 1e-9
        avg = sum(values) / len(values)
        z = (avg - mean_base) / (sd / math.sqrt(len(values)))
        if mean_base > 0 and avg >= AVG_RATIO * mean_base and z >= AVG_Z:
            out[(name, city)] = {"n": len(values), "avg": avg, "career": mean_base, "best": max(values), "ratio": avg / mean_base, "z": z}
    return out


def all_average_claims(fmt: str = "t20", kind: str = "bat") -> dict:
    return {k: dict(v) for k, v in _average_claims(fmt, kind).items()}


def batter_average_line(full_name: str, resolved: str | None, city: str, runs: int, balls: int, fmt: str = "t20") -> str:
    """Card for a batter far above his usual level at this ground (shown while he is still batting)."""
    c = _average_claims(fmt, "bat").get((resolved, city)) if resolved and city else None
    if not c:
        return ""
    fact = (f"Loves {city.title()}: averages {c['avg']:.0f} here in {c['n']} innings vs {c['career']:.0f} overall "
            f"({c['ratio']:.1f}x), best {c['best']}")
    return card(fmt, f"{full_name} {runs}*({balls})", fact, c["n"], ratio_fill(c["ratio"]))


def bowler_average_line(full_name: str, resolved: str | None, city: str, wickets: int, balls: int, fmt: str = "t20") -> str:
    c = _average_claims(fmt, "bowl").get((resolved, city)) if resolved and city else None
    if not c:
        return ""
    fact = (f"Loves {city.title()}: {c['avg']:.1f} wickets per innings here in {c['n']} innings vs {c['career']:.1f} overall "
            f"({c['ratio']:.1f}x), best {c['best']}")
    return card(fmt, f"{full_name} {balls // 6}.{balls % 6} ov, {wickets} wkt{'' if wickets == 1 else 's'}", fact, c["n"], ratio_fill(c["ratio"]))
