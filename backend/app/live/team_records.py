"""Team Fact-o-Meter: 100% records for a TEAM in a situation -- winning the toss, batting first, chasing.

Per format (T20 = every men's T20 competition, ODI = men's ODIs; never mixed), from every decided match in the ground
facts. For each team and each situation
  * winning the toss, winning the toss and bowling / batting, batting first, chasing
the team's results are checked in three scopes: overall, at one ground (city key) and against one opponent. A record is
"won ALL n" or "lost ALL n" with at least MIN_MATCHES (7) and a fluke chance of at most 1% (the league's own win rate for
that situation, ~48-52%, to the power n -- so about 7 matches). Per situation the least-likely scope is kept.
Pre-toss the card lists winning the toss / batting first / chasing; once the toss is known it keeps only what applies
(the toss winner's records, who bats first). Records of the past, not forecasts.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

from app.live.fact_o_meter import meter
from app.live.prematch_conditions import load_facts_odi, load_facts_t20

MIN_MATCHES = 7
MAX_FLUKE = 0.01
SITUATIONS = {
    "won the toss": lambda r: r["toss"] is True,
    "batting first": lambda r: r["first"] is True,
    "chasing": lambda r: r["first"] is False,
    "won the toss and bowled": lambda r: r["toss"] is True and r["dec"] == "field",
    "won the toss and batted": lambda r: r["toss"] is True and r["dec"] == "bat",
}
PHRASE = {"won the toss": "Winning the toss", "batting first": "Batting first", "chasing": "Chasing",
          "won the toss and bowled": "Winning the toss and bowling", "won the toss and batted": "Winning the toss and batting"}


def _team_rows(fmt: str) -> list[dict]:
    out = []
    for f in (load_facts_odi() if fmt == "odi" else load_facts_t20()):
        if len(f["teams"]) != 2 or not f["winner"] or f["winner"] not in f["teams"]:
            continue
        for team in f["teams"]:
            out.append({
                "team": team, "opp": next(t for t in f["teams"] if t != team), "city": f["city"], "won": int(f["winner"] == team),
                "toss": (f.get("toss_winner") == team) if f.get("toss_winner") in f["teams"] else None, "dec": f.get("toss_decision"),
                "first": (f["first_team"] == team) if f["first_team"] in f["teams"] else None,
            })
    return out


@lru_cache(maxsize=2)
def _claims(fmt: str = "t20") -> dict[str, list[dict]]:
    """team -> [{situation, scope ("overall" | "ground" | "opponent"), where, outcome ("WON"/"LOST"), n, fluke}] (best scope per
    situation, per team)."""
    rows = _team_rows(fmt)
    rate = {s: (sum(r["won"] for r in rows if fn(r)) / max(1, sum(1 for r in rows if fn(r)))) for s, fn in SITUATIONS.items()}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["team"], "overall", "")].append(r)
        groups[(r["team"], "ground", r["city"])].append(r)
        groups[(r["team"], "opponent", r["opp"])].append(r)
    best: dict[tuple[str, str], dict] = {}
    for (team, scope, where), lst in groups.items():
        for situation, fn in SITUATIONS.items():
            results = [r["won"] for r in lst if fn(r)]
            n = len(results)
            if n < MIN_MATCHES or len(set(results)) != 1:
                continue
            won = bool(results[0])
            fluke = (rate[situation] if won else 1 - rate[situation]) ** n
            if fluke > MAX_FLUKE:
                continue
            claim = {"situation": situation, "scope": scope, "where": where, "outcome": "WON" if won else "LOST", "n": n, "fluke": fluke}
            current = best.get((team, situation, scope, where))
            key = (team, situation, scope, where)
            if current is None or fluke < current["fluke"]:
                best[key] = claim
    out: dict[str, list[dict]] = defaultdict(list)
    for (team, *_), claim in best.items():
        out[team].append(claim)
    return dict(out)


def all_claims(fmt: str = "t20") -> dict[str, list[dict]]:
    return {t: [dict(c) for c in v] for t, v in _claims(fmt).items()}


def _applies(situation: str, team: str, toss_won_by: str, toss_decision: str, bat_first: str) -> bool:
    if not toss_won_by and not bat_first:  # pre-toss: the three general situations
        return situation in ("won the toss", "batting first", "chasing")
    if situation == "won the toss":
        return team == toss_won_by
    if situation == "won the toss and bowled":
        return team == toss_won_by and toss_decision.lower() in ("bowl", "field")
    if situation == "won the toss and batted":
        return team == toss_won_by and toss_decision.lower() == "bat"
    if situation == "batting first":
        return bool(bat_first) and team == bat_first
    return bool(bat_first) and team != bat_first  # chasing


def _line(claim: dict, city: str, fmt: str) -> str:
    where = {"overall": "", "ground": f" at {claim['where'].title()}", "opponent": f" vs {claim['where']}"}[claim["scope"]]
    result = f"won {claim['n']} of {claim['n']}" if claim["outcome"] == "WON" else f"lost {claim['n']} of {claim['n']}"
    return f"{PHRASE[claim['situation']]}{where}: {result}  {meter(claim['n'])}"


def team_card(team: str, opponent: str, city: str, toss_won_by: str = "", toss_decision: str = "", bat_first: str = "",
              fmt: str = "t20", limit: int = 3) -> str:
    """The team's Fact-o-Meter card for this match: its 100% toss / batting-first / chasing records at this ground, against
    this opponent or overall (the least likely scope per situation), or "" when it has none that apply."""
    chosen: dict[str, dict] = {}
    for claim in _claims(fmt).get(team, []):
        if claim["scope"] == "ground" and claim["where"] != city:
            continue
        if claim["scope"] == "opponent" and claim["where"] != opponent:
            continue
        if not _applies(claim["situation"], team, toss_won_by, toss_decision, bat_first):
            continue
        if claim["situation"] not in chosen or claim["fluke"] < chosen[claim["situation"]]["fluke"]:
            chosen[claim["situation"]] = claim
    if not chosen:
        return ""
    top = sorted(chosen.values(), key=lambda c: c["fluke"])[:limit]
    return f"🎯 {'FACT-O-METER'} ({fmt.upper()}) · {team}\n" + "\n".join("   " + _line(c, city, fmt) for c in top)
