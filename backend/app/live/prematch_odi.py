"""Pre-match win probability for men's ODIs: the T20I base block (Elo + head-to-head) rebuilt on ODI history.

Same point-in-time construction as prematch_model.py (State is shared); only the history differs.
No dew / XI blocks for ODIs yet: Cricsheet has no start times (day vs day-night) and the XI player
records are T20-only. See train_prematch_odi.py for the backtest that decides what is shown.
"""

from __future__ import annotations

import json
from functools import lru_cache

from app.live.prematch_model import ROOT, State, _load, sigmoid

ODI_DIR = ROOT / "data" / "raw" / "cricsheet" / "odi"
MODEL_PATH = ROOT / "models" / "prematch_odi_base.json"
FEATURES = ("elo", "h2h", "h2h5")


@lru_cache(maxsize=1)
def odi_rows() -> tuple[dict, ...]:
    return _load(ODI_DIR)


def decided(row: dict) -> bool:
    return bool(row["winner"]) and len(row["teams"]) == 2 and row["winner"] in row["teams"]


def build_dataset() -> list[dict]:
    """One row per decided men's ODI, features from strictly earlier ODIs. Orientation: team_a < team_b."""
    state = State()
    out = []
    for row in odi_rows():
        if not decided(row):
            continue
        a, b = sorted(row["teams"])
        x = state.features(a, b, row["city"], 0)
        out.append({
            "date": row["date"], "a": a, "b": b, "city": row["city"],
            "y": 1 if row["winner"] == a else 0,
            "x": {"elo": x[0], "form": x[1], "h2h": x[2], "h2h5": x[3]},
            "n_a": len(state.residuals.get(a, [])), "n_b": len(state.residuals.get(b, [])),
            "elo_p": state.expected(a, b),
        })
        state.apply_t20i({**row, "teams": (a, b)})
    return out


@lru_cache(maxsize=1)
def current_state() -> State:
    state = State()
    for row in odi_rows():
        if decided(row):
            a, b = sorted(row["teams"])
            state.apply_t20i({**row, "teams": (a, b)})
    return state


def predict_base(team_a: str, team_b: str) -> dict | None:
    """ODI win probability from Elo + head-to-head, or None without a saved model / for a team with no ODIs."""
    try:
        saved = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    state = current_state()
    if team_a not in state.ratings or team_b not in state.ratings:
        return None
    a, b = sorted((team_a, team_b))
    row = state.features(a, b, "", 0)
    values = {"elo": row[0], "h2h": row[2], "h2h5": row[3]}
    block = saved["base"]
    z = sum(w * values[f] / sc for f, w, sc in zip(block["features"], block["weights"], block["scale"]))
    p_a = sigmoid(z)
    p_first = p_a if team_a == a else 1.0 - p_a
    return {"p_a": p_first, "p_b": 1.0 - p_first, "stage": "pre-toss",
            "trained_through": saved.get("trained_through"), "backtest": saved.get("backtest", {})}


def odi_text(team_a: str, team_b: str, log_context: dict | None = None) -> str:
    """Compact ODI pre-match block, same shape as the T20 one: Win %, Form, Head to head (all from men's ODIs).
    Pre-toss only: there is no ODI dew / XI block yet (see module docstring)."""
    from app.live.prematch_analysis import _short_team

    a, b = team_a.strip(), team_b.strip()
    rows = [r for r in odi_rows() if decided(r)]
    known = {t for r in rows for t in r["teams"]}
    missing = [t for t in (a, b) if t not in known]
    if missing:
        return f"Win %: not available -- no ODI history for {' / '.join(missing)} in our data"
    prediction = predict_base(a, b)
    lines = []
    if prediction:
        pa, pb = round(prediction["p_a"] * 100), round(prediction["p_b"] * 100)
        lines.append(f"Win %: {_short_team(a)} {pa}% | {_short_team(b)} {pb}% (pre-toss, ODI)")
        if log_context and log_context.get("match_id"):
            from app.live import prematch_log

            prematch_log.log_prediction(log_context["match_id"], {
                "source": log_context.get("source", ""), "format": "odi", "team_a": a, "team_b": b,
                "city": log_context.get("city", ""), "stage": "pre-toss", "bat_first": "", "dew_known": False,
                "p_a": prediction["p_a"], "p_b": prediction["p_b"], "p_nr": 0.0,
                "pct_a": pa, "pct_b": pb, "pct_nr": 0,
            })
    forms = []
    for team in (a, b):
        games = [r for r in rows if team in r["teams"]][-5:]
        forms.append(f"{team} {''.join('W' if r['winner'] == team else 'L' for r in games)}")
    lines.append("Form (last 5): " + " | ".join(forms))
    meetings = [r for r in rows if set(r["teams"]) == {a, b}][-5:]
    if meetings:
        lines.append(
            f"Head to head: {_short_team(a)} - {sum(r['winner'] == a for r in meetings)} vs "
            f"{_short_team(b)} - {sum(r['winner'] == b for r in meetings)} in last {len(meetings)} matches"
        )
    else:
        lines.append("Head to head: no previous men's ODI meetings")
    return "\n".join(lines)


def odi_briefing(label: str, venue: str = "", toss_won_by: str = "", toss_decision: str = "",
                 match_id: str = "", source: str = "") -> str:
    """The ODI briefing: title with the ODI ground record, weather + pitch, toss, then odi_text()."""
    from app.live.match_weather import match_day_weather
    from app.live.prematch_analysis import _cached, _pitch_short, _weather_line, toi_pitch_note, toss_line, venue_summary
    from app.live.prematch_model import venue_city

    teams = [t.strip() for t in label.split(" vs ")]
    city = venue_city(venue or "")
    lines = [f"🏏 {label}" + (f" -- {city.title()} | {venue_summary(city, 'odi')}" if city else "")]
    if city:
        try:
            lines.append(_weather_line(_cached(("weather", city), 600, lambda: match_day_weather(city))))
        except Exception:
            pass
    if len(teams) == 2:
        note = toi_pitch_note(teams[0], teams[1])
        lines.append(_pitch_short(note) if note else "Pitch: unknown")
        lines.append(toss_line(toss_won_by, toss_decision, teams[0], teams[1]))
        from app.live.prematch_analysis import team_card_block

        cards = team_card_block(teams[0], teams[1], venue or "", toss_won_by, toss_decision, "odi")
        if cards:
            lines.append("")
            lines.extend(cards)
        lines.append("")
        lines.append(odi_text(teams[0], teams[1], {"match_id": match_id, "source": source, "city": city} if match_id else None))
    return "\n".join(lines)
