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
    """Line-up status: "XIs: ✅ FULL LINE-UPS CONFIRMED (India 11, West Indies 11)" followed by one
    line of names per team in batting order, a partial
    "XIs: partly announced -- India 11 | West Indies 7 confirmed", or "XIs: not announced yet"."""
    counts = {team: sum(1 for p in players if p.confirm_xi) for team, players in (team_players or {}).items()}
    if not counts or not any(counts.values()):
        return "XIs: not announced yet"
    if len(counts) == 2 and all(n >= 11 for n in counts.values()):
        header = "XIs: ✅ FULL LINE-UPS CONFIRMED (" + ", ".join(f"{team} {n}" for team, n in counts.items()) + ")"
        names = [
            f"{team}: " + ", ".join(p.name for p in sorted(players, key=lambda p: p.position) if p.confirm_xi)
            for team, players in team_players.items()
        ]
        return "\n".join([header, *names])
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


_TTL: dict = {}


def _cached(key, seconds, fn):
    """Tiny time-limited cache so the header and the body share one weather / TOI call."""
    import time as _time

    hit = _TTL.get(key)
    if hit and _time.monotonic() - hit[0] < seconds:
        return hit[1]
    value = fn()
    _TTL[key] = (_time.monotonic(), value)
    return value


def toi_pitch_note(team_a: str, team_b: str) -> str:
    """TOI's pitch note for this fixture ("Batting friendly; Dry surface | Likely ...") or "".
    Best effort: any failure returns "" so a briefing never depends on it."""

    def fetch():
        try:
            from app.live.toi_discovery import ToiMatchDiscovery

            wanted = {team_a.lower(), team_b.lower()}
            for match in ToiMatchDiscovery().matches():
                if {t.strip().lower() for t in match.label.split(" vs ")} == wanted:
                    pitch = "" if "unknown; unknown" in match.pitch.lower() else match.pitch
                    track = "" if "no reliable" in match.likely_track.lower() else match.likely_track
                    return " | ".join(x for x in (pitch, track) if x)
        except Exception:
            pass
        return ""

    return _cached(("toi-pitch", team_a, team_b), 600, fetch)


def _track_call(note: str, city: str, cloud: float | None) -> str:
    """Flat / green / turning, only from evidence that holds up.

    * ground scoring level (prior T20I + IPL first-innings totals at the ground vs all
      grounds) -- validated out of sample: correlation 0.37 with the next match's total;
    * TOI's pitch note, the only direct source for "green" or "turning";
    * the ground's spin-vs-pace wicket history is NOT used: its correlation with the next
      match's spin share was 0.05.
    """
    import re

    from app.live.prematch_conditions import load_facts

    totals = _cached(("ground-totals",), 3600, lambda: _ground_totals())
    ground = totals.get(city, [])
    league = totals.get("__all__", [150.0])
    league_avg = sum(league) / len(league)
    scoring = ""
    if len(ground) >= 5:
        avg = sum(ground) / len(ground)
        shrunk = (sum(ground) + 10 * league_avg) / (len(ground) + 10)
        label = "high" if shrunk >= league_avg + 8 else "low" if shrunk <= league_avg - 10 else "average"
        scoring = f"ground scoring {label}, avg {avg:.0f} in {len(ground)} games vs {league_avg:.0f} overall"
    # Only TOI's own pitch descriptor decides the call. Its "likely track" sentence ("good grip
    # for slower balls/spin as the surface wears") is boilerplate it prints for every dry
    # surface (seen on four different grounds in one day), so it is not evidence.
    pitch_field = note.split("|")[0].strip()
    text = pitch_field.lower()
    kind = ""
    if re.search(r"\b(green|grass|seam|pace[- ]friendly|bowling[- ]friendly|bounce)\b", text):
        kind = "green / seam-friendly"
    elif re.search(r"\b(spin|turn|turning|dusty)\b", text):
        kind = "turning"
    elif re.search(r"batting[- ]friendly|true batting|\bflat\b", text):
        kind = "flat"
    parts = []
    if kind:
        parts.append(f"{kind} (TOI: {pitch_field})")
    elif pitch_field:
        parts.append(f"TOI: {pitch_field}")
    if "dry" in text and kind != "turning":
        parts.append("may take turn as it wears")
    if scoring:
        parts.append(scoring)
    if cloud is not None and cloud >= 70:
        parts.append(f"overcast ({cloud:.0f}% cloud) -- some help for seamers")
    if not kind:
        parts.append("green / turning: no reliable source for this ground")
    return "Pitch: " + "; ".join(parts)


def _ground_totals() -> dict:
    from app.live.prematch_conditions import load_facts

    out: dict = {"__all__": []}
    for fact in load_facts():
        if fact["totals"]:
            out.setdefault(fact["city"], []).append(fact["totals"][0])
            out["__all__"].append(fact["totals"][0])
    return out


def _weather_line(day: dict | None) -> str:
    if not day:
        return "Weather: forecast unavailable"
    parts = []
    if day.get("eve_temp") is not None:
        parts.append(f"{day['eve_temp']:.0f}C in the evening")
    if day.get("eve_humidity") is not None:
        parts.append(f"humidity {day['eve_humidity']:.0f}%")
    if day.get("aft_cloud") is not None:
        parts.append(f"cloud {day['aft_cloud']:.0f}%")
    parts.append(f"rain {day['rain_mm']:.1f} mm" if day.get("rain_mm") else "no rain")
    gap = None
    if day.get("eve_temp") is not None and day.get("eve_dew_point") is not None:
        gap = day["eve_temp"] - day["eve_dew_point"]
    dew = "" if gap is None else " | DEW LIKELY (helps the chasing side)" if gap <= 2.5 else " | no dew expected"
    return "Weather: " + ", ".join(parts) + dew


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
        weather = _cached(("weather", city), 600, lambda: match_day_weather(city)) if city else None
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
    if reasons:
        lines.append("Why: " + "; ".join(reasons))
    if form_text:
        lines.append(form_text)
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


def prematch_header(
    team_a: str,
    team_b: str,
    venue: str = "",
    toss_won_by: str = "",
    toss_decision: str = "",
    xi: dict | None = None,
    pitch_note: str | None = None,
) -> list[str]:
    """Three lines for the top of the briefing: Weather, Pitch (flat / green / turning with its
    evidence) and Advantage (the team the conditions favour, in points of win probability).

    Advantage = dew effect + spin/pace XI matchup effect for each side -- the parts of the win %
    that come from conditions, not from team strength or history. Before the toss dew cannot be
    assigned to a team (it helps whoever chases), so that is said instead.
    """
    from app.live.match_weather import match_day_weather
    from app.live.prematch_conditions import predict_with_dew
    from app.live.prematch_model import venue_city
    from app.live.prematch_xi import predict_with_xi

    city = venue_city(venue)
    try:
        weather = _cached(("weather", city), 600, lambda: match_day_weather(city)) if city else None
    except Exception:
        weather = None
    note = pitch_note if pitch_note is not None else toi_pitch_note(team_a, team_b)
    cloud = weather.get("aft_cloud") if weather else None
    lines = [_weather_line(weather), _track_call(note, city, cloud)]

    a, b = team_a.strip(), team_b.strip()
    bat_first = _bat_first_team(a, b, toss_won_by, toss_decision)
    dew_likely = bool(
        weather and weather.get("eve_temp") is not None and weather.get("eve_dew_point") is not None
        and weather["eve_temp"] - weather["eve_dew_point"] <= 2.5
    )
    if not bat_first:
        lines.append(
            "Advantage: toss pending -- dew would help whoever chases" if dew_likely else "Advantage: toss pending -- no clear conditions edge yet"
        )
        return lines
    edge_a = 0.0
    notes = []
    try:
        dew = predict_with_dew(a, b, bat_first, weather)
        if dew is not None and abs(dew["dew_shift"]) >= 0.005:
            edge_a += dew["dew_shift"]
            chaser = b if bat_first == a else a
            notes.append(f"dew helps {chaser} ({abs(dew['dew_shift']) * 100:.0f} pts)")
        if xi and a in xi and b in xi:
            result = predict_with_xi(a, b, bat_first, weather, xi[a], xi[b], city)
            if result is not None and abs(result["matchup_shift"]) >= 0.005:
                edge_a += result["matchup_shift"]
                side = a if result["matchup_shift"] > 0 else b
                notes.append(f"spin/pace matchup favours {side} ({abs(result['matchup_shift']) * 100:.0f} pts)")
    except Exception:
        pass
    if abs(edge_a) < 0.01:
        lines.append("Advantage: even on conditions" + (f" ({'; '.join(notes)})" if notes else ""))
    else:
        lines.append(f"Advantage: {a if edge_a > 0 else b} on conditions ({'; '.join(notes)})")
    return lines
