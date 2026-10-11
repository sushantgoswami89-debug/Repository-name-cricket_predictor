"""Fact-o-Meter lines for the pre-match briefing: what the records say about the players in THIS XI, at THIS ground, against THIS team.

Needs both confirmed XIs. For every player (T20 data: every men's T20 competition; the T20 briefing only) it looks up
  * his 100% ground record (batters: X+ in every innings there; bowlers: k+ wickets in every innings there),
  * his average-gap ground fact (far above his own career average at this ground),
  * his team's 100% result record against this opponent / overall ("India have WON all 14 matches vs Sri Lanka in which
    he scored 75+"),
ranks them by how unlikely luck would be, keeps one per player and the best `limit` overall. Innings-specific records
are used only when the toss says who bats first. Records of the past, never forecasts.
"""

from __future__ import annotations

import math

from app.live.fact_o_meter import label, meter, ratio_fill
from app.live.prematch_model import venue_city
from app.live.result_facts import _claims as _result_claims
from app.live.venue_form import _average_claims, _batter_claims, _bowler_claims


def _luck(p: float) -> float:
    return -math.log10(max(p, 1e-12))


def _innings_for(team: str, role: str, bat_first: str) -> int:
    """1 / 2 for the innings this team plays in this role, 0 when the toss is not known."""
    if not bat_first:
        return 0
    batting_innings = 1 if team == bat_first else 2
    return batting_innings if role == "bat" else 3 - batting_innings


def xi_fact_lines(team_a: str, team_b: str, xi: dict[str, list[str]] | None, venue: str, bat_first: str = "",
                  fmt: str = "t20", limit: int = 4) -> list[str]:
    """Up to `limit` one-line Fact-o-Meter facts about the players in the confirmed XIs ([] when none or no XI)."""
    if not xi:
        return []
    from app.live.prematch_xi import resolve_player

    city = venue_city(venue or "")
    found: list[tuple[float, str, str, int, int]] = []   # (luck score, player, text, n, bar fill)
    for team, opponent in ((team_a, team_b), (team_b, team_a)):
        for full in xi.get(team, []):
            resolved = resolve_player(full, team)
            if not resolved:
                continue
            options: list[tuple[float, str, int, int]] = []  # (luck, text, n, bar fill)
            for role, claims in (("bat", _batter_claims(fmt)), ("bowl", _bowler_claims(fmt))):
                for inn in {0, _innings_for(team, role, bat_first)}:
                    claim = claims.get((resolved, city, inn)) if city else None
                    if not claim:
                        continue
                    when = {0: "every innings", 1: "every first innings", 2: "every second innings"}[inn] if role == "bat" else \
                           {0: "every spell", 1: "every time he bowls first", 2: "every time he bowls second"}[inn]
                    what = f"{claim['x']}+ in {when}" if role == "bat" else f"{claim['x']}+ wickets {when}"
                    options.append((_luck(claim["fluke"]), f"{what} at {city.title()}, {claim['n']} of {claim['n']} (avg {claim['avg']:.1f})", claim["n"], 10))
            for role in ("bat", "bowl"):
                avg = _average_claims(fmt, role).get((resolved, city)) if city else None
                if avg:
                    unit = "runs" if role == "bat" else "wickets"
                    options.append((_luck(0.5 * math.erfc(avg["z"] / math.sqrt(2))),
                                    f"averages {avg['avg']:.0f} {unit} at {city.title()} ({avg['n']} inns) vs {avg['career']:.0f} overall"
                                    if role == "bat" else
                                    f"{avg['avg']:.1f} wickets an innings at {city.title()} ({avg['n']} inns) vs {avg['career']:.1f} overall", avg["n"],
                                    ratio_fill(avg["ratio"])))
            for claim in _result_claims(fmt).get(resolved, []):
                if claim["opp"] not in ("", opponent):
                    continue
                if claim["inn"] not in (0, _innings_for(team, claim["kind"], bat_first)):
                    continue
                vs = f" vs {opponent}" if claim["opp"] else ""
                did = f"scored {claim['x']}+" if claim["kind"] == "bat" else f"took {claim['x']}+ wickets"
                options.append((_luck(claim["fluke"]), f"{team} have {claim['outcome']} all {claim['n']} matches{vs} in which he {did}", claim["n"], 10))
            if options:
                score, text, n, fill = max(options)
                found.append((score, full, text, n, fill))
    found.sort(reverse=True)
    return [f"🎯 {label(fmt)}: {full} -- {text}  {meter(n, fill)}" for _, full, text, n, fill in found[:limit]]
