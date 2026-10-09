"""Pre-match analysis from raw Cricsheet men's T20I history.

Descriptive history only (head-to-head, recent form, venue record) -- NOT a
model forecast. The live match-winner model's H2H snapshot
(data/live/team_h2h_pairs.json) is IPL-only and has no international
pairings, so this reads the raw T20I files directly. Added 2026-10-06 after
the briefing was found to be facts-only (toss/venue/XI counts) with no
analysis at all.

prematch_text() never raises: a missing data directory or unknown team
returns None so a failure here can never block the live prediction flow.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
T20I_DIR = ROOT / "data" / "raw" / "cricsheet" / "t20i"


def _first_innings_total(match: dict) -> tuple[str, int] | None:
    innings = match.get("innings") or []
    if not innings:
        return None
    first = innings[0]
    total = sum(
        delivery["runs"]["total"]
        for over in first.get("overs", [])
        for delivery in over.get("deliveries", [])
    )
    return first.get("team", ""), total


@lru_cache(maxsize=1)
def _all_rows() -> tuple[dict, ...]:
    rows = []
    for path in T20I_DIR.glob("*.json"):
        match = json.loads(path.read_text(encoding="utf-8"))
        info = match["info"]
        if info.get("gender") not in (None, "male"):
            continue
        first = _first_innings_total(match)
        rows.append(
            {
                "date": info["dates"][0],
                "teams": frozenset(info.get("teams", [])),
                "venue": f"{info.get('venue', '')} {info.get('city', '')}".lower(),
                "winner": info.get("outcome", {}).get("winner"),
                "first_team": first[0] if first else None,
                "first_total": first[1] if first else None,
            }
        )
    rows.sort(key=lambda row: row["date"])
    return tuple(rows)


def _pct(part: int, whole: int) -> str:
    return f"{part / whole * 100:.0f}%" if whole else "n/a"


def _venue_needles(venue: str) -> list[str]:
    # TOI/Cricbuzz venue strings look like "<long stadium name>, <city>";
    # Cricsheet's differ in the stadium part, so the city is the reliable key.
    city = venue.rsplit(",", 1)[-1].strip().lower()
    return [city] if city else []


def _bat_first_team(team_a: str, team_b: str, toss_won_by: str, toss_decision: str) -> str:
    """The side that bats first, from the toss, or "" when unknown."""
    winner = toss_won_by.strip()
    if winner not in (team_a, team_b):
        return ""
    other = team_b if winner == team_a else team_a
    decision = toss_decision.strip().lower()
    if decision.startswith("bat"):
        return winner
    if decision.startswith(("bowl", "field")):
        return other
    return ""


def confirmed_xi(team_players: dict) -> dict[str, list[str]] | None:
    """{team: [names]} of confirmed players when both sides have at least 9; else None."""
    out = {
        team: [p.name for p in players if p.confirm_xi]
        for team, players in (team_players or {}).items()
    }
    return out if len(out) == 2 and all(len(v) >= 9 for v in out.values()) else None


def _xi_lines(team_a, team_b, venue, bat_first, weather, xi):
    """(win % line, extra lines) from the XI-aware model, or None."""
    from app.live.prematch_model import venue_city
    from app.live.prematch_xi import predict_with_xi

    if not xi or team_a not in xi or team_b not in xi:
        return None
    result = predict_with_xi(team_a, team_b, bat_first, weather, xi[team_a], xi[team_b], venue_city(venue))
    if result is None:
        return None
    mix = result["mix"]
    describe = lambda t: "/".join(f"{g} {mix[t][g] * 100:.0f}%" for g in ("pace", "finger", "wrist"))
    shift = result["xi_shift"] * 100
    lines = [
        f"XI check: {team_a} attack {describe(team_a)}; {team_b} attack {describe(team_b)} (share of overs by pace/finger-spin/wrist-spin)."
        f" Players matched to history: {team_a} {result['matched'][team_a]}, {team_b} {result['matched'][team_b]}."
        f" For {team_a}: XI quality {result['quality_shift'] * 100:+.0f} pts, spin/pace matchup {result['matchup_shift'] * 100:+.0f} pts."
    ]
    return result, lines


def _conditions_line(city: str, day: dict) -> str:
    gap = None
    if day.get("eve_temp") is not None and day.get("eve_dew_point") is not None:
        gap = day["eve_temp"] - day["eve_dew_point"]
    parts = []
    if day.get("eve_temp") is not None:
        parts.append(f"evening {day['eve_temp']:.0f}C")
    if gap is not None:
        parts.append(f"dew point {day['eve_dew_point']:.0f}C (gap {gap:.1f}C)")
    if day.get("eve_humidity") is not None:
        parts.append(f"humidity {day['eve_humidity']:.0f}%")
    if day.get("aft_cloud") is not None:
        parts.append(f"afternoon cloud {day['aft_cloud']:.0f}%")
    if day.get("rain_mm"):
        parts.append(f"rain {day['rain_mm']:.1f} mm")
    verdict = ""
    if gap is not None:
        verdict = " -- DEW LIKELY (helps the chasing side)" if gap <= 2.5 else " -- dew not expected"
    return f"Conditions (forecast, {city.title()}): " + ", ".join(parts) + verdict


def _win_percent_block(team_a: str, team_b: str, venue: str, bat_first: str, xi: dict | None = None) -> list[str]:
    """Model win % plus the evidence next to it. Empty when no saved model or a
    team has no history in the local data."""
    from app.live.match_weather import match_day_weather
    from app.live.prematch_conditions import predict_with_dew
    from app.live.prematch_model import current_state, predict, venue_city

    result = predict(team_a, team_b, venue, bat_first)
    if result is None:
        return []
    city = venue_city(venue)
    weather = None
    try:
        weather = match_day_weather(city) if city else None
    except Exception:
        weather = None
    dew = predict_with_dew(team_a, team_b, bat_first, weather)
    xi_result = None
    try:
        xi_result = _xi_lines(team_a, team_b, venue, bat_first, weather, xi)
    except Exception:
        xi_result = None
    if xi_result is not None:
        model, xi_lines = xi_result
        holdout = model["holdout"].get("full_member") or {}
        lines = [f"Win %: {team_a} {model['p_a'] * 100:.0f}% | {team_b} {model['p_b'] * 100:.0f}% (with XIs)"]
        lines.append(
            "XI effect is small and not yet proven for full-member games alone "
            "(2021-25: clear gain across all T20Is, +0.3 pts Brier on full members, within noise)."
        )
    elif dew is not None:
        holdout = dew["holdout"].get("full_member") or {}
        shift = dew["dew_shift"] * 100
        lines = [
            f"Win %: {team_a} {dew['p_a'] * 100:.0f}% | {team_b} {dew['p_b'] * 100:.0f}%"
            + (f" (dew moves {abs(shift):.0f} pts)" if abs(shift) >= 0.5 else "")
        ]
    else:
        holdout = result["holdout"].get("full_member") or {}
        lines = [
            f"Win %: {team_a} {result['p_a'] * 100:.0f}% | {team_b} {result['p_b'] * 100:.0f}%"
            + ("" if result["post_toss"] else " (toss not known: venue and dew effects off)")
        ]
    if xi_result is not None:
        lines.extend(xi_lines)
    if holdout:
        lines.append(
            f"Model check: on {holdout['n']} unseen 2025+ games between two full members it picked the "
            f"winner {holdout['acc'] * 100:.0f}% of the time (a coin flip is 50%). Treat as a lean, not a forecast."
        )
    if weather:
        lines.append(_conditions_line(city, weather))
        lines.append(
            "Tested and left out: cloud cover, and team-level spin/pace matchups (no gain)."
            if xi_result is not None
            else "Tested and left out: cloud cover and team-level spin/pace matchups (no gain); XIs not yet announced."
        )
    state = current_state()
    if city:
        pieces = []
        for label, table in (("T20Is", state.venue_t20i), ("IPL", state.venue_ipl)):
            record = table.get(city)
            if record and record.matches:
                pieces.append(
                    f"{label}: {record.matches} matches, bat first won "
                    f"{record.bat_first_wins / record.matches * 100:.0f}%"
                )
        if pieces:
            lines.append(f"At {city.title()} -- " + "; ".join(pieces) + " (shown separately, not blended)")
    if bat_first:
        lines.append(f"Bats first: {bat_first}")
    lines.append("Inputs: team strength (Elo), head-to-head, and (after the toss) dew and the announced XIs.")
    return lines


def prematch_text(
    team_a: str,
    team_b: str,
    venue: str = "",
    toss_won_by: str = "",
    toss_decision: str = "",
    xi: dict | None = None,
) -> str | None:
    try:
        rows = _all_rows()
    except Exception:
        return None
    a, b = team_a.strip(), team_b.strip()
    known = {team for row in rows for team in row["teams"]}
    if a not in known or b not in known:
        return None

    lines = [f"📊 Pre-match history ({a} vs {b}, men's T20Is)"]
    h2h = [r for r in rows if r["teams"] == {a, b} and r["winner"]]
    if h2h:
        wins_a = sum(r["winner"] == a for r in h2h)
        last5 = h2h[-5:]
        lines.append(
            f"H2H: {a} {wins_a}-{len(h2h) - wins_a} {b} ({len(h2h)} matches, "
            f"{a} {_pct(wins_a, len(h2h))}); last {len(last5)}: "
            f"{a} {sum(r['winner'] == a for r in last5)}-"
            f"{sum(r['winner'] == b for r in last5)}"
        )
    else:
        lines.append("H2H: no previous men's T20I meetings")

    forms = []
    for team in (a, b):
        games = [r for r in rows if team in r["teams"] and r["winner"]][-5:]
        forms.append(f"{team} {' '.join('W' if r['winner'] == team else 'L' for r in games)}")
    lines.append("Form (last 5): " + " | ".join(forms))

    needles = _venue_needles(venue)
    if needles:
        at_venue = [r for r in rows if any(n in r["venue"] for n in needles)]
        scored = [r for r in at_venue if r["first_total"] is not None and r["winner"]]
        if scored:
            avg = sum(r["first_total"] for r in scored) / len(scored)
            bat_first = sum(r["winner"] == r["first_team"] for r in scored)
            line = (
                f"Venue ({venue.rsplit(',', 1)[-1].strip()}): {len(scored)} T20Is, "
                f"avg 1st inns {avg:.0f}, bat-first won {_pct(bat_first, len(scored))}"
            )
            if len(scored) < 10:
                line += " (small sample)"
            lines.append(line)
        else:
            lines.append(f"Venue ({venue.rsplit(',', 1)[-1].strip()}): no men's T20I history")
    try:
        block = _win_percent_block(a, b, venue, _bat_first_team(a, b, toss_won_by, toss_decision), xi)
    except Exception:
        block = []
    if block:
        lines.append("")
        lines.extend(block)
    else:
        lines.append("History only, not a model forecast.")
    return "\n".join(lines)
