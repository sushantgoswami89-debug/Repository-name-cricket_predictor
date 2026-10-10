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

import re

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
    """Line-up status, one line: "XIs: ✅ FULL LINE-UPS CONFIRMED (India 11, West Indies 11)", a partial
    "XIs: partly announced -- India 11 | West Indies 7 confirmed", or "XIs: not announced yet"."""
    counts = {team: sum(1 for p in players if p.confirm_xi) for team, players in (team_players or {}).items()}
    if not counts or not any(counts.values()):
        return "XIs: not announced yet"
    if len(counts) == 2 and all(n >= 11 for n in counts.values()):
        return "XIs: ✅ FULL LINE-UPS CONFIRMED (" + ", ".join(f"{team} {n}" for team, n in counts.items()) + ")"
    return "XIs: partly announced -- " + " | ".join(f"{team} {n}" for team, n in counts.items()) + " confirmed"


def xi_names(team_players: dict) -> list[str]:
    """The player names, one line per team in batting order -- only once BOTH full line-ups are
    confirmed. Goes at the very end of the briefing."""
    counts = {team: sum(1 for p in players if p.confirm_xi) for team, players in (team_players or {}).items()}
    if not (len(counts) == 2 and all(n >= 11 for n in counts.values())):
        return []
    return [
        f"{team}: " + ", ".join(p.name for p in sorted(players, key=lambda p: p.position) if p.confirm_xi)
        for team, players in team_players.items()
    ]


def confirmed_xi(team_players: dict) -> dict[str, list[str]] | None:
    """{team: [names]} of confirmed players when both sides have at least 9; else None."""
    out = {
        team: [p.name for p in players if p.confirm_xi]
        for team, players in (team_players or {}).items()
    }
    return out if len(out) == 2 and all(len(v) >= 9 for v in out.values()) else None


def _ground_record(city: str) -> dict:
    """Per source and combined: matches, mean first-innings total, bat-first wins (decided matches)."""
    from app.live.prematch_conditions import load_facts

    def build():
        table: dict = {}
        for fact in load_facts():
            row = table.setdefault(fact["city"], {})
            for key in (fact["src"], "all"):
                cell = row.setdefault(
                    key,
                    {"n": 0, "totals": [], "all_totals": [], "wkts": [], "best_pace": 0, "best_spin": 0,
                     "decided": 0, "bat_first_wins": 0},
                )
                cell["n"] += 1
                if fact["totals"]:
                    cell["totals"].append(fact["totals"][0])
                cell["all_totals"].extend(fact["totals"])
                cell["wkts"].extend(fact.get("innings_wkts", []))
                best = fact.get("best_by_group") or {}
                cell["best_pace"] = max(cell["best_pace"], best.get("pace", 0))
                cell["best_spin"] = max(cell["best_spin"], best.get("spin", 0))
                if fact["winner"] and fact["first_team"] in fact["teams"]:
                    cell["decided"] += 1
                    cell["bat_first_wins"] += int(fact["winner"] == fact["first_team"])
        return table

    return _cached(("ground-record",), 3600, build).get(city, {})


def venue_summary(city: str) -> str:
    """The title-line ground record, pooled over every T20I and IPL match at the ground, in the
    format the user specified: "T20 - 11 1st inns avg 153, avg wkts - 6, pace - 3/spin - 4"
    (11 = matches; pace / spin = most wickets by one bowler of that kind in a match; bowlers without
    a known style are not counted in those two)."""
    record = _ground_record(city)
    total = record.get("all")
    if not total:
        return "T20 - no history here"

    def mean(values):
        return f"{sum(values) / len(values):.0f}" if values else "n/a"

    text = (
        f"T20 - {total['n']} 1st inns avg {mean(total['totals'])}, "
        f"avg wkts - {mean(total['wkts'])}, pace - {total['best_pace']}/spin - {total['best_spin']}"
    )
    if total["n"] < 10:
        text += " [small sample]"
    return text


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
        scoring = f"ground scoring {label}"
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


DEW_GAP_C = 2.5          # temperature minus dew point at/below which dew is likely (backtested threshold)
DEW_ONSET_GAP_C = 3.0    # dew starts to form a little above that
DEW_SIGMA_C = 1.5        # ASSUMED forecast error of the temperature - dew point gap, used for the "chance"
FIRST_INNINGS_HOURS = 1.8  # a T20 chase begins roughly 1h50 after the first ball


def _clock(hours: float) -> str:
    total = int(round(hours * 60 / 5) * 5)
    return f"{(total // 60) % 24:02d}:{total % 60:02d}"


def dew_outlook(day: dict | None, start_local_h: float | None = None) -> dict | None:
    """Dew weight, chance and onset from the hour-by-hour forecast.

    weight: heavy (gap <= 1.5C in the second half of the chase), medium (<= 2.5C), small (<= 4C), little (more).
            Thresholds follow the backtest: the chasing side only gained above the 2.5C line.
    chance: probability the gap is at or below 2.5C during the chase if the forecast gap has a
            +-1.5C standard error -- that error size is an assumption, not measured.
    onset:  first time after sunset the gap falls to 3C or less (linear between forecast hours).
    """
    import math

    hours = (day or {}).get("hours") or {}
    if len(hours) < 12:
        return None
    gap = {h: v["temp"] - v["dew_point"] for h, v in hours.items()}

    def gap_at(t: float) -> float | None:
        lo, hi = int(math.floor(t)), int(math.floor(t)) + 1
        if lo not in gap:
            return None
        if hi not in gap:
            return gap[lo]
        return gap[lo] + (gap[hi] - gap[lo]) * (t - lo)

    start = start_local_h if start_local_h is not None else 19.0
    chase_from, chase_to = start + FIRST_INNINGS_HOURS, start + 3.5
    window = [g for g in (gap_at(chase_from + 0.5 * i) for i in range(int((chase_to - chase_from) / 0.5) + 1)) if g is not None]
    if not window:
        return None
    late = window[len(window) // 2:]  # dew bites in the second half of the chase
    mean_gap = sum(late) / len(late)
    weight = "HEAVY" if mean_gap <= 1.5 else "MEDIUM" if mean_gap <= DEW_GAP_C else "SMALL" if mean_gap <= 4.0 else "LITTLE"
    chance = 0.5 * (1.0 + math.erf((DEW_GAP_C - mean_gap) / (DEW_SIGMA_C * math.sqrt(2))))
    sunset = day.get("sunset_h")
    begin = max(sunset if sunset is not None else 17.5, 12.0)
    onset = None
    t = begin
    while t <= 23.5:
        g = gap_at(t)
        if g is not None and g <= DEW_ONSET_GAP_C:
            onset = t
            break
        t += 0.25
    return {
        "weight": weight,
        "chance": round(chance * 20) * 5,  # nearest 5%
        "mean_gap": mean_gap,
        "onset_h": onset,
        "chase_h": chase_from,
        "start_assumed": start_local_h is None,
    }


def _dew_line(outlook: dict | None) -> str:
    if outlook is None:
        return "Dew: no hourly forecast available"
    if outlook["weight"] == "LITTLE":
        return f"Dew: LITTLE -- {outlook['chance']:.0f}% chance, little effect expected on the chase"
    onset, chase = outlook["onset_h"], outlook["chase_h"]
    when = "no onset in the forecast"
    if onset is not None:
        timing = "before the chase" if onset <= chase - 0.25 else "around the start of the chase" if onset <= chase + 0.25 else "mid-chase"
        when = f"sets in ~{_clock(onset)}, {timing} (~{_clock(chase)}{', start time assumed 19:00' if outlook['start_assumed'] else ''})"
    helps = " -- helps the chasing side" if outlook["weight"] in ("MEDIUM", "HEAVY") else ""
    return f"Dew: {outlook['weight']} -- {outlook['chance']:.0f}% chance, {when}{helps}"


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
    return "Weather: " + ", ".join(parts)


def _compact_block(team_a: str, team_b: str, venue: str, bat_first: str, xi: dict | None, h2h_text: str, form_text: str) -> list[str]:
    """The pre-match section, two lines: the win % (with its stage) and the last-5 form.
    The build-up of the number and the model's measured accuracy are documented in the notes, not sent."""
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
    if form_text:
        lines.append(form_text)
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


def _dew_short(outlook: dict | None) -> str:
    """"Dew: HEAVY -- 90%, sets in ~18:40" (weight, chance, onset)."""
    if outlook is None:
        return "Dew: n/a"
    text = f"Dew: {outlook['weight']} -- {outlook['chance']:.0f}%"
    if outlook["weight"] != "LITTLE" and outlook["onset_h"] is not None:
        text += f", sets in ~{_clock(outlook['onset_h'])}"
    return text


def _pitch_short(note: str) -> str:
    """"Pitch: flat (Batting friendly; Dry surface)" -- the call, with TOI's own words in brackets.
    green / turning only when TOI's pitch field says so; otherwise "unknown"."""
    pitch_field = note.split("|")[0].strip()
    text = pitch_field.lower()
    if re.search(r"\b(green|grass|seam|pace[- ]friendly|bowling[- ]friendly|bounce)\b", text):
        kind = "green"
    elif re.search(r"\b(spin|turn|turning|dusty)\b", text):
        kind = "turning"
    elif re.search(r"batting[- ]friendly|true batting|\bflat\b", text):
        kind = "flat"
    else:
        kind = "unknown"
    return f"Pitch: {kind}" + (f" ({pitch_field})" if pitch_field else "")


def prematch_header(
    team_a: str,
    team_b: str,
    venue: str = "",
    toss_won_by: str = "",
    toss_decision: str = "",
    xi: dict | None = None,
    pitch_note: str | None = None,
    start_utc_ms: int | None = None,
) -> list[str]:
    """The facts under the title: Weather, then one line "Dew: HEAVY -- 90%, sets in ~18:40, Pitch: flat
    (Batting friendly; Dry surface)". The estimate itself -- win % and how it builds up from the base
    rating through dew and the XI effects -- is ONE compiled block at the bottom (prematch_text),
    so the briefing never gives two estimates that point different ways."""
    from app.live.match_weather import match_day_weather
    from app.live.prematch_model import venue_city

    city = venue_city(venue)
    try:
        weather = _cached(("weather", city), 600, lambda: match_day_weather(city)) if city else None
    except Exception:
        weather = None
    note = pitch_note if pitch_note is not None else toi_pitch_note(team_a, team_b)
    cloud = weather.get("aft_cloud") if weather else None
    start_local_h = None
    if start_utc_ms and weather and weather.get("utc_offset_s") is not None:
        local_s = start_utc_ms / 1000.0 + weather["utc_offset_s"]
        start_local_h = (local_s % 86400) / 3600.0
    return [
        _weather_line(weather),
        _dew_short(dew_outlook(weather, start_local_h)) + ", " + _pitch_short(note),
    ]
