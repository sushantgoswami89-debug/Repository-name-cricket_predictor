"""Fact-o-Meter: "when he does X, his team ALWAYS wins / loses" -- 100% result records for batters and bowlers.

Per format (T20 = every men's T20 competition, ODI = men's ODIs; never mixed), from every decided match
(data/features/player_results[_odi].json, build_player_results.py [odi]). A record is "in all n matches where he scored
X+ (batting first / chasing / either) his team WON" -- or LOST -- and it needs
  * at least MIN_MATCHES such matches, and
  * a fluke chance of at most 1%: the league's own rate of that outcome when any player does that, to the power n
    (a team wins ~65% when a batter makes 50+, so a 22-for-22 run is a fact; 5-for-5 is not).
Targets: batters 30/50/75/100 runs (T20), 50/75/100/150 (ODI); bowlers 2/3/4/5 wickets (bowling at least an over).
It is shown live when he is on the way: a batter on 60%+ of the target and short of it, a bowler one wicket short.
Checked out of sample 2026-10-11 (record from earlier matches, tested on his next one): win records held 72% (T20) and
82% (ODI) against 63% / 64% for the league; loss records repeated 49% (T20) / 61% (ODI) against 38% / 37%. So these are
real but modest -- the card states the record, never a prediction. Kohli has no such record: India won 24 of 29 ODIs
when he made 100+ chasing.
"""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache

from app.live.fact_o_meter import card
from app.live.prematch_model import ROOT

PATHS = {"t20": ROOT / "data" / "features" / "player_results.json",
         "odi": ROOT / "data" / "features" / "player_results_odi.json"}
BAT_TARGETS = {"t20": (30, 50, 75, 100), "odi": (50, 75, 100, 150)}
BOWL_TARGETS = (2, 3, 4, 5)
MIN_MATCHES = 5
MAX_FLUKE = 0.01
APPROACH = 0.6  # a batter is "on the way" from 60% of the target


@lru_cache(maxsize=2)
def _claims(fmt: str = "t20") -> dict[str, list[dict]]:
    """player -> list of {kind, x, inn, outcome ("WON"/"LOST"), n, fluke}."""
    try:
        rows = json.loads(PATHS[fmt].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    league = defaultdict(lambda: [0, 0])
    per = defaultdict(lambda: defaultdict(list))
    for _date, name, _team, bat_inn, runs, _balls, bowl_inn, wkts, bowled, won, *_rest in rows:
        events = []
        if bat_inn:
            for x in BAT_TARGETS[fmt]:
                if runs >= x:
                    events += [("bat", x, 0), ("bat", x, bat_inn)]
        if bowl_inn and bowled >= 6:
            for k in BOWL_TARGETS:
                if wkts >= k:
                    events += [("bowl", k, 0), ("bowl", k, bowl_inn)]
        for key in events:
            league[key][0] += 1
            league[key][1] += won
            per[name][key].append(won)
    out: dict[str, list[dict]] = {}
    for name, conditions in per.items():
        for (kind, x, inn), results in conditions.items():
            n = len(results)
            if n < MIN_MATCHES or len(set(results)) != 1:
                continue
            win_rate = league[(kind, x, inn)][1] / league[(kind, x, inn)][0]
            won = bool(results[0])
            fluke = (win_rate if won else 1 - win_rate) ** n
            if fluke <= MAX_FLUKE:
                out.setdefault(name, []).append({"kind": kind, "x": x, "inn": inn, "outcome": "WON" if won else "LOST",
                                                 "n": n, "fluke": fluke})
    return out


def all_claims(fmt: str = "t20") -> dict[str, list[dict]]:
    return {name: [dict(c) for c in claims] for name, claims in _claims(fmt).items()}


def _words(claim: dict) -> str:
    inn = {0: "", 1: " batting first", 2: " chasing"} if claim["kind"] == "bat" else {0: "", 1: " bowling first", 2: " bowling second"}
    if claim["kind"] == "bat":
        return f"in which he scored {claim['x']}+{inn[claim['inn']]}"
    return f"in which he took {claim['x']}+ wickets{inn[claim['inn']]}"


def result_line(full_name: str, resolved: str | None, team: str, kind: str, value: int, balls: int, innings: int,
                fmt: str = "t20") -> str:
    """The Fact-o-Meter card for his best result record that applies right now, else "".
    kind "bat": value = runs so far, balls = balls faced; kind "bowl": value = wickets so far, balls = legal balls bowled.
    `resolved` is his Cricsheet name (resolve_player); `team` is the live team name for the wording."""
    if not resolved:
        return ""
    live = []
    for claim in _claims(fmt).get(resolved, []):
        if claim["kind"] != kind or claim["inn"] not in (0, innings):
            continue
        if kind == "bat" and not (APPROACH * claim["x"] <= value < claim["x"]):
            continue
        if kind == "bowl" and not (value == claim["x"] - 1 and balls >= 6):
            continue
        live.append(claim)
    if not live:
        return ""
    claim = min(live, key=lambda c: (c["fluke"], -c["n"]))
    if kind == "bat":
        head = f"{full_name} {value}*({balls})"
    else:
        head = f"{full_name} {balls // 6}.{balls % 6} ov, {value} wkt{'' if value == 1 else 's'}"
    n = claim["n"]
    fact = f"{team} have {claim['outcome']} all {n} matches {_words(claim)}"
    return card(fmt, head, fact, n)
