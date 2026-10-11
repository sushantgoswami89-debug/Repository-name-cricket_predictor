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
from app.live.fact_o_meter import MAX_BASELINE, label as fact_label, meter

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


_SHORT_TEAMS = {
    "West Indies": "WI", "South Africa": "SA", "New Zealand": "NZ", "Sri Lanka": "SL",
    "United Arab Emirates": "UAE", "United States of America": "USA", "Papua New Guinea": "PNG",
}


def _short_team(name: str) -> str:
    return _SHORT_TEAMS.get(name, name)


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


def _ground_record(city: str, fmt: str = "t20") -> dict:
    """Per source and combined: matches, first / second innings totals, wickets, bat-first wins and
    toss-winner wins (decided matches). fmt "t20" pools every men's T20 competition (internationals + IPL, BBL, PSL, ...); "odi" is men's ODIs (facts built by
    `build_match_facts.py odi`)."""
    from app.live.prematch_conditions import load_facts_odi, load_facts_t20

    def build():
        table: dict = {}
        for fact in (load_facts_odi() if fmt == "odi" else load_facts_t20()):
            row = table.setdefault(fact["city"], {})
            for key in (fact["src"], "all"):
                cell = row.setdefault(
                    key,
                    {"n": 0, "totals": [], "second_totals": [], "all_totals": [], "wkts": [], "best_pace": 0,
                     "best_spin": 0, "pace_wkts": [], "spin_wkts": [], "decided": 0, "bat_first_wins": 0, "toss_decided": 0, "toss_wins": 0, "toss_bat": 0, "toss_field": 0},
                )
                cell["n"] += 1
                if fact["totals"]:
                    cell["totals"].append(fact["totals"][0])
                if len(fact["totals"]) > 1:
                    cell["second_totals"].append(fact["totals"][1])
                cell["all_totals"].extend(fact["totals"])
                cell["wkts"].extend(fact.get("innings_wkts", []))
                cell["pace_wkts"].extend(fact.get("innings_pace_wkts", []))
                cell["spin_wkts"].extend(fact.get("innings_spin_wkts", []))
                best = fact.get("best_by_group") or {}
                cell["best_pace"] = max(cell["best_pace"], best.get("pace", 0))
                cell["best_spin"] = max(cell["best_spin"], best.get("spin", 0))
                if fact["winner"] and fact["first_team"] in fact["teams"]:
                    cell["decided"] += 1
                    cell["bat_first_wins"] += int(fact["winner"] == fact["first_team"])
                if fact["winner"] and fact.get("toss_winner") in fact["teams"]:
                    cell["toss_decided"] += 1
                    cell["toss_wins"] += int(fact["winner"] == fact["toss_winner"])
                if fact.get("toss_decision") in ("bat", "field"):
                    cell["toss_bat" if fact["toss_decision"] == "bat" else "toss_field"] += 1
        return table

    return _cached(("ground-record", fmt), 3600, build).get(city, {})


VENUE_RECORD_MIN = 6
VENUE_RECORD_MAX_FLUKE = 0.01


def _venue_event_table(fmt: str = "t20") -> tuple[dict, dict]:
    """(league win rate per event, {city: {event: [won, ...]}}) over decided matches of ONE format (T20I + IPL, or
    ODI). Events: the toss winner who chose to bowl / to bat won; the team batting first / chasing won."""
    from app.live.prematch_conditions import load_facts_odi, load_facts_t20

    def build():
        per_city: dict = {}
        league: dict = {}
        for fact in (load_facts_odi() if fmt == "odi" else load_facts_t20()):
            if not fact["winner"] or fact["winner"] not in fact["teams"]:
                continue
            events = {}
            if fact.get("toss_winner") in fact["teams"]:
                won_toss_side = fact["winner"] == fact["toss_winner"]
                if fact.get("toss_decision") == "field":
                    events["the toss winner who chose to bowl"] = won_toss_side
                elif fact.get("toss_decision") == "bat":
                    events["the toss winner who chose to bat"] = won_toss_side
            if fact["first_team"] in fact["teams"]:
                events["the team batting first"] = fact["winner"] == fact["first_team"]
                events["the team chasing"] = fact["winner"] != fact["first_team"]
            for name, won in events.items():
                per_city.setdefault(fact["city"], {}).setdefault(name, []).append(won)
                cell = league.setdefault(name, [0, 0])
                cell[0] += 1
                cell[1] += int(won)
        return {k: v[1] / v[0] for k, v in league.items()}, per_city

    return _cached(("venue-events", fmt), 3600, build)


def venue_record(city: str, fmt: str = "t20") -> str:
    """"the team chasing has won 8 of 8 here" -- a 100% record at the ground in that format only, or "".
    Needs at least 6 such matches and a fluke chance (league rate for that event ** n) of at most 1%; the
    least likely one is shown. A fact about the past, not a prediction."""
    rates, per_city = _venue_event_table(fmt)
    best = None
    for name, results in per_city.get(city, {}).items():
        n = len(results)
        if n < VENUE_RECORD_MIN or not all(results):
            continue
        fluke = rates[name] ** n
        if rates[name] <= MAX_BASELINE and fluke <= VENUE_RECORD_MAX_FLUKE and (best is None or fluke < best[0]):
            best = (fluke, f"{name} has won {n} of {n} here", n)
    return best[1] if best else ""


def venue_summary(city: str, fmt: str = "t20") -> str:
    """The title-line ground record.

    Scoring numbers come from the match's OWN format only, never mixed: a T20 match uses every men's T20
    competition at the ground (internationals + IPL, BBL, PSL, CPL, ...), an ODI match uses men's ODIs:
      "T20 - 11 1st inns avg 153, avg wkts - 6, pace - 3.2/spin - 2.1"  /  "ODI - 8 1st inns avg 332, ..."
    (avg wkts = wickets falling per innings, run-outs included; pace / spin = average wickets per innings
    credited to pace / spin bowlers, so they add up to a bit less than avg wkts: run-outs and bowlers
    without a known style are in neither). The results record pools T20 and ODI matches at the ground into one figure:
      "bat 1st wins 38% / chase wins 62% / toss winner wins 56%, toss winner chooses bowl 70%"
    (toss choice shows the more common one only; "[small sample]" is added when under 10 matches pooled.)
    `fmt` chooses the format of the scoring numbers and of the 100% venue record that is appended.
    """
    t20 = _ground_record(city, "t20").get("all")
    odi = _ground_record(city, "odi").get("all")
    if not t20 and not odi:
        return "T20 - no history here"

    def mean(values):
        return f"{sum(values) / len(values):.0f}" if values else "n/a"

    def mean1(values):
        return f"{sum(values) / len(values):.1f}" if values else "n/a"

    own, own_label = (odi, "ODI") if fmt == "odi" else (t20, "T20")
    if own:
        text = (f"{own_label} - {own['n']} 1st inns avg {mean(own['totals'])}, avg wkts - {mean(own['wkts'])}, "
                f"pace - {mean1(own['pace_wkts'])}/spin - {mean1(own['spin_wkts'])}")
        if own["n"] < 10:
            text += " [small sample]"
    else:
        text = f"{own_label} - no history here"
    cells = [c for c in (t20, odi) if c]
    decided = sum(c["decided"] for c in cells)
    if decided:
        first = sum(c["bat_first_wins"] for c in cells) / decided
        text += f" | bat 1st wins {first:.0%} / chase wins {1 - first:.0%}"
        toss = sum(c["toss_decided"] for c in cells)
        if toss:
            text += f" / toss winner wins {sum(c['toss_wins'] for c in cells) / toss:.0%}"
        choices = sum(c["toss_bat"] + c["toss_field"] for c in cells)
        if choices:
            bat = sum(c["toss_bat"] for c in cells) / choices
            text += ", toss winner chooses " + (f"bat {bat:.0%}" if bat > 0.5 else f"bowl {1 - bat:.0%}" if bat < 0.5 else "bat 50% / bowl 50%")
        if decided < 10:
            text += " [small sample]"
    record = venue_record(city, "odi" if fmt == "odi" else "t20")  # records are format-specific, never pooled
    if record:
        n = int(re.search(r"won (\d+) of", record).group(1))
        text += f" | 🎯 {fact_label('odi' if fmt == 'odi' else 't20')}: {record} {meter(n)}"
    return text


_TTL: dict = {}


def _cached(key, seconds, fn):
    """Tiny time-limited cache so the header and the body share one weather / TOI call."""
    import time as _time

    hit = _TTL.get(key)
    if hit and _time.monotonic() - hit[0] < seconds:
        return hit[1]
    value = fn()
    if value:  # never cache a failure (None / empty): the next call should try again
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
    from app.live.rain_risk import rain_risk

    risk = rain_risk(day)
    parts.append(f"rain {day['rain_mm']:.1f} mm" if day.get("rain_mm") else "no rain")
    if risk:
        parts.append(f"rain risk {risk['p_interrupted'] * 100:.0f}% (washout {risk['p_no_result'] * 100:.0f}%)")
    return "Weather: " + ", ".join(parts)


def _compact_block(
    team_a: str, team_b: str, venue: str, bat_first: str, xi: dict | None, h2h_text: str, form_text: str,
    log_context: dict | None = None,
) -> list[str]:
    """The pre-match section, three lines: the win % (with its stage), the last-5 form and the head to head.
    One staged model produces the number (pre-toss -> after toss -> with XIs); each stage adds a block
    on top of the same base, so the figure only moves because new information arrived."""
    from app.live.match_weather import match_day_weather
    from app.live.prematch_model import venue_city
    from app.live.prematch_xi import predict_stages

    city = venue_city(venue)
    weather = None
    if city:  # the forecast gives rain risk before the toss and dew once we know who chases
        try:
            weather = _cached(("weather", city), 600, lambda: match_day_weather(city))
        except Exception:
            weather = None
    result = None
    try:
        names_a = (xi or {}).get(team_a, [])
        names_b = (xi or {}).get(team_b, [])
        result = predict_stages(team_a, team_b, bat_first, weather, names_a, names_b, city)
    except Exception:
        result = None
    if result is None:
        return []
    label = result["stage"]
    if bat_first and not result["dew_known"]:
        label += ", dew unknown"  # the forecast could not be fetched: the number has no dew term
    from app.live.rain_risk import rain_risk

    risk = rain_risk(weather)
    p_nr = risk["p_no_result"] if risk else 0.0
    if p_nr < 0.02:  # a washout below 2% is noise: show the plain two-way split
        p_nr = 0.0
    # Rain enters the number as a washout share: each side can only win if the match is completed.
    shares = [result["p_a"] * (1 - p_nr), result["p_b"] * (1 - p_nr), p_nr]
    pct = [int(round(v * 100)) for v in shares]
    favourite = 0 if shares[0] >= shares[1] else 1
    pct[favourite] += 100 - sum(pct)  # keep the printed numbers summing to exactly 100
    win_line = f"Win %: {team_a} {pct[0]}% | {team_b} {pct[1]}%"
    if p_nr > 0:
        win_line += f" | No result {pct[2]}%"
    lines = [f"{win_line} ({label})"]
    if log_context and log_context.get("match_id"):
        from app.live.prematch_log import log_prediction

        log_prediction(str(log_context["match_id"]), {
            "source": log_context.get("source"), "team_a": team_a, "team_b": team_b, "city": city,
            "stage": result["stage"], "dew_known": result["dew_known"], "bat_first": bat_first or None,
            "p_a": round(shares[0], 4), "p_b": round(shares[1], 4), "p_nr": round(shares[2], 4),
            "pct_a": pct[0], "pct_b": pct[1], "pct_nr": pct[2],
            "p_base": round(result["p_base"], 4), "dew_shift": round(result["dew_shift"], 4),
            "quality_shift": round(result.get("quality_shift", 0.0), 4),
            "matchup_shift": round(result.get("matchup_shift", 0.0), 4),
            "rain_interrupted": round(risk["p_interrupted"], 4) if risk else None,
            "model_through": result.get("model_through"),
        })
    if form_text:
        lines.append(form_text)
    if h2h_text:
        lines.append(h2h_text)
    lines.extend(_matchup_lines(result))
    return lines


_GROUP_WORDS = {"pace": "pace", "finger": "finger spin", "wrist": "wrist spin"}


def _matchup_lines(result: dict) -> list[str]:
    """Two lines, one per direction, once both XIs are known:
    "India bat v WI bowl: +55 | WI attack: pace 49% / spin 51%". The number is net runs per 20 overs
    against an average T20 innings (a wicket costs 12), after the opposing attack's mix is applied."""
    directions = result.get("directions")
    if not directions:
        return []
    lines = []
    for bat, d in directions.items():
        bowl = d["bowling_team"]
        mix = d["attack_mix"]
        spin = mix["finger"] + mix["wrist"]
        line = (
            f"{_short_team(bat)} bat v {_short_team(bowl)} bowl: {d['net_runs']:+.0f} runs vs avg | "
            f"{_short_team(bowl)} attack pace {mix['pace'] * 100:.0f}% / spin {spin * 100:.0f}%"
        )
        if d["weak_vs"]:
            line += f" | {_short_team(bat)} weak v {_GROUP_WORDS[d['weak_vs']]} ({d['weak_share'] * 100:.0f}% of overs)"
        lines.append(line)
    return lines


def prematch_text(
    team_a: str,
    team_b: str,
    venue: str = "",
    toss_won_by: str = "",
    toss_decision: str = "",
    xi: dict | None = None,
    log_context: dict | None = None,
) -> str | None:
    """Compact pre-match section (about 6 lines): win %, why, form, dew, ground."""
    try:
        rows = _all_rows()
    except Exception:
        return None
    a, b = team_a.strip(), team_b.strip()
    known = {team for row in rows for team in row["teams"]}
    missing = [t for t in (a, b) if t not in known]
    if missing:
        # Cricsheet has no Afghanistan matches at all (0 of 5,775 international T20s), so say so
        # instead of silently dropping the win % (checked 2026-10-10: not fixable by a download).
        return f"Win %: not available -- no match history for {' / '.join(missing)} in our data"

    h2h = [r for r in rows if r["teams"] == {a, b} and r["winner"]]
    # Head to head over the last 5 decided meetings: "Head to head: India - 3 vs WI - 2 in last 5 matches".
    if h2h:
        last5 = h2h[-5:]
        h2h_text = (
            f"Head to head: {_short_team(a)} - {sum(r['winner'] == a for r in last5)} vs "
            f"{_short_team(b)} - {sum(r['winner'] == b for r in last5)} in last {len(last5)} matches"
        )
    else:
        h2h_text = "Head to head: no previous men's T20I meetings"
    forms = []
    for team in (a, b):
        games = [r for r in rows if team in r["teams"] and r["winner"]][-5:]
        forms.append(f"{team} {''.join('W' if r['winner'] == team else 'L' for r in games)}")
    form_text = "Form (last 5): " + " | ".join(forms)

    try:
        lines = _compact_block(a, b, venue, _bat_first_team(a, b, toss_won_by, toss_decision), xi, h2h_text, form_text, log_context)
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
    dew_text = _dew_short(dew_outlook(weather, start_local_h))
    a, b = team_a.strip(), team_b.strip()
    bat_first = _bat_first_team(a, b, toss_won_by, toss_decision)
    if bat_first and weather:
        try:
            from app.live.prematch_model import venue_city as _vc
            from app.live.prematch_xi import predict_stages

            staged = predict_stages(a, b, bat_first, weather, [], [], _vc(venue))
        except Exception:
            staged = None
        if staged and abs(staged["dew_shift"]) >= 0.005:
            chaser = b if bat_first == a else a
            dew_text += f" ({_short_team(chaser)} +{abs(staged['dew_shift']) * 100:.0f} pts)"
    return [
        _weather_line(weather),
        dew_text + ", " + _pitch_short(note),
    ]


def fact_o_meter_block(team_a: str, team_b: str, xi: dict | None, venue: str, toss_won_by: str = "",
                       toss_decision: str = "") -> list[str]:
    """Fact-o-Meter lines about the players in the confirmed XIs at this ground / against this team (see prematch_facts).
    Empty until both XIs are confirmed; never raises (a side feature must not block the briefing)."""
    if not xi:
        return []
    try:
        from app.live.prematch_facts import xi_fact_lines

        return xi_fact_lines(team_a, team_b, xi, venue, _bat_first_team(team_a, team_b, toss_won_by, toss_decision))
    except Exception as error:
        print(f"[briefing] fact-o-meter unavailable: {error!r}")
        return []


def team_card_block(team_a: str, team_b: str, venue: str, toss_won_by: str = "", toss_decision: str = "",
                    fmt: str = "t20") -> list[str]:
    """Team Fact-o-Meter cards (toss winning, batting first, chasing records) for both sides; [] when neither has one."""
    try:
        from app.live.prematch_model import venue_city
        from app.live.team_records import team_card

        city = venue_city(venue or "")
        bat_first = _bat_first_team(team_a, team_b, toss_won_by, toss_decision)
        cards = [team_card(team, other, city, toss_won_by, toss_decision, bat_first, fmt)
                 for team, other in ((team_a, team_b), (team_b, team_a))]
        return [c for c in cards if c]
    except Exception as error:
        print(f"[briefing] team fact-o-meter unavailable: {error!r}")
        return []
