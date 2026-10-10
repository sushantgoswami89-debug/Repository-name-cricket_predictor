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


def toss_line(toss_won_by: str, toss_decision: str, team_a: str = "", team_b: str = "") -> str:
    """"Toss: West Indies won, chose to bowl -> India bat first" (or "not yet available")."""
    if not toss_won_by:
        return "Toss: not yet available"
    decision = toss_decision.strip().lower()
    verb = "bat" if decision.startswith("bat") else "bowl" if decision.startswith(("bowl", "field")) else decision or "unknown"
    text = f"Toss: {toss_won_by} won, chose to {verb}"
    first = _bat_first_team(team_a, team_b, toss_won_by, toss_decision) if team_a and team_b else ""
    return text + (f" -> {first} bat first" if first else "")


def xi_line(team_players: dict) -> str:
    """Line-up status: "XIs: ✅ FULL LINE-UPS CONFIRMED (India 11, West Indies 11)", a partial
    "XIs: partly announced -- India 11 | West Indies 7 confirmed", or "XIs: not announced yet"."""
    counts = {team: sum(1 for p in players if p.confirm_xi) for team, players in (team_players or {}).items()}
    if not counts or not any(counts.values()):
        return "XIs: not announced yet"
    if len(counts) == 2 and all(n >= 11 for n in counts.values()):
        return "XIs: ✅ FULL LINE-UPS CONFIRMED (" + ", ".join(f"{team} {n}" for team, n in counts.items()) + ")"
    return "XIs: partly announced -- " + " | ".join(f"{team} {n}" for team, n in counts.items()) + " confirmed"


def confirmed_xi(team_players: dict) -> dict[str, list[str]] | None:
    """{team: [names]} of confirmed players when both sides have at least 9; else None."""
    out = {
        team: [p.name for p in players if p.confirm_xi]
        for team, players in (team_players or {}).items()
    }
    return out if len(out) == 2 and all(len(v) >= 9 for v in out.values()) else None


def _venue_line(city: str) -> str:
    """One line on the ground: bat-first win rate in T20Is and IPL, with sample sizes."""
    from app.live.prematch_model import current_state

    state = current_state()
    pieces = []
    for label, table in (("T20Is", state.venue_t20i), ("IPL", state.venue_ipl)):
        record = table.get(city)
        if record and record.matches:
            pieces.append(f"{record.bat_first_wins / record.matches * 100:.0f}% in {record.matches} {label}")
    if not pieces:
        return ""
    small = all((table.get(city).matches if table.get(city) else 0) < 10 for table in (state.venue_t20i,))
    return f"Ground: {city.title()} -- bat first won " + ", ".join(pieces) + (" (small sample)" if small else "")


def _dew_line(day: dict) -> str:
    gap = None
    if day.get("eve_temp") is not None and day.get("eve_dew_point") is not None:
        gap = day["eve_temp"] - day["eve_dew_point"]
    if gap is None:
        return ""
    detail = f"humidity {day['eve_humidity']:.0f}%, gap {gap:.1f}C" if day.get("eve_humidity") is not None else f"gap {gap:.1f}C"
    if gap <= 2.5:
        return f"Dew: LIKELY tonight ({detail}) -- helps the chasing side"
    return f"Dew: not expected ({detail})"


def _compact_block(team_a: str, team_b: str, venue: str, bat_first: str, xi: dict | None, h2h_text: str, form_text: str) -> list[str]:
    """The pre-match section: win %, the reason in one line, form, dew, ground."""
    from app.live.match_weather import match_day_weather
    from app.live.prematch_conditions import predict_with_dew
    from app.live.prematch_model import predict, venue_city
    from app.live.prematch_xi import predict_with_xi

    base = predict(team_a, team_b, venue, bat_first)
    if base is None:
        return []
    city = venue_city(venue)
    try:
        weather = match_day_weather(city) if city else None
    except Exception:
        weather = None
    chosen, label, holdout, xi_result = base, "pre-toss", base["holdout"].get("full_member") or {}, None
    try:
        if xi and team_a in xi and team_b in xi:
            xi_result = predict_with_xi(team_a, team_b, bat_first, weather, xi[team_a], xi[team_b], city)
    except Exception:
        xi_result = None
    dew = predict_with_dew(team_a, team_b, bat_first, weather)
    if xi_result is not None:
        chosen, label, holdout = xi_result, "with XIs", xi_result["holdout"].get("full_member") or {}
    elif dew is not None:
        chosen, label, holdout = dew, "after toss", dew["holdout"].get("full_member") or {}
    elif base["post_toss"]:
        label = "after toss"

    lines = [f"Win %: {team_a} {chosen['p_a'] * 100:.0f}% | {team_b} {chosen['p_b'] * 100:.0f}% ({label})"]
    if holdout:
        lines.append(
            f"A lean, not a forecast: the model was right {holdout['acc'] * 100:.0f}% on {holdout['n']} unseen full-member games."
        )
    reasons = [h2h_text] if h2h_text else []
    if xi_result is not None:
        reasons.append(f"XIs {xi_result['xi_shift'] * 100:+.0f} pts for {team_a}")
    if dew is not None and abs(dew["dew_shift"]) >= 0.005:
        chaser = team_b if bat_first == team_a else team_a
        reasons.append(f"dew favours {chaser} ({abs(dew['dew_shift']) * 100:.0f} pts)")
    if reasons:
        lines.append("Why: " + "; ".join(reasons))
    if form_text:
        lines.append(form_text)
    if weather:
        dew_line = _dew_line(weather)
        if dew_line:
            lines.append(dew_line)
    if city:
        ground = _venue_line(city)
        if ground:
            lines.append(ground)
    return lines


def prematch_text(
    team_a: str,
    team_b: str,
    venue: str = "",
    toss_won_by: str = "",
    toss_decision: str = "",
    xi: dict | None = None,
) -> str | None:
    """Compact pre-match section (about 6 lines): win %, why, form, dew, ground."""
    try:
        rows = _all_rows()
    except Exception:
        return None
    a, b = team_a.strip(), team_b.strip()
    known = {team for row in rows for team in row["teams"]}
    if a not in known or b not in known:
        return None

    h2h = [r for r in rows if r["teams"] == {a, b} and r["winner"]]
    h2h_text = ""
    if h2h:
        wins_a = sum(r["winner"] == a for r in h2h)
        last5 = h2h[-5:]
        h2h_text = (
            f"H2H {a} {wins_a}-{len(h2h) - wins_a} (last 5: {sum(r['winner'] == a for r in last5)}-"
            f"{sum(r['winner'] == b for r in last5)})"
        )
    forms = []
    for team in (a, b):
        games = [r for r in rows if team in r["teams"] and r["winner"]][-5:]
        forms.append(f"{team} {''.join('W' if r['winner'] == team else 'L' for r in games)}")
    form_text = "Form (last 5): " + " | ".join(forms)

    try:
        lines = _compact_block(a, b, venue, _bat_first_team(a, b, toss_won_by, toss_decision), xi, h2h_text, form_text)
    except Exception:
        lines = []
    if not lines:
        # No saved model or a team without history: facts only.
        lines = [x for x in (h2h_text, form_text) if x] + ["(no win % for this fixture: not enough history)"]
    return "\n".join(lines)
