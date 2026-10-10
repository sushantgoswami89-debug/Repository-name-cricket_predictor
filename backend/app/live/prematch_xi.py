"""Player-level spin/pace matchups from the actual XIs (point-in-time).

Follow-up to the team-level test (2026-10-09, no gain): a team's last 20 games
are a crude stand-in for who actually plays. Here every historical match uses
its real XI (Cricsheet `info.players`) and each player's own record against
pace, finger spin and wrist spin, built only from earlier matches.

Per match, for team A against team B's XI:
  batting value of A's XI vs bowler group g = weighted mean over A's batters of
      shrunk (runs - 12*dismissals)/ball vs g, relative to the league
  B's attack mix = share of B's expected balls bowled by pace/finger/wrist (from
      each bowler's balls-per-appearance so far); optionally tilted by the pitch
  matchup(A) = sum_g mix_B[g] * (A's value vs g - A's overall value)
  feature   = matchup(A) - matchup(B)       (interaction beyond plain quality)
  quality   = (A bat + A bowl) - (B bat + B bowl), the XI-quality baseline
Unknown-style bowlers carry no group (skipped, not guessed).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from app.live.prematch_conditions import ConditionsState, GROUPS, dew_index, load_weather
from app.ml.player_style_registry import PlayerStyleRegistry

ROOT = Path(__file__).resolve().parents[3]
LAMBDA = 12.0
PSEUDO_ALL = 60.0
PSEUDO_GROUP = 120.0
DEFAULT_WEIGHT = 8.0
STYLE_GROUP = {
    "left_arm_pace": "pace", "right_arm_pace": "pace",
    "right_arm_offbreak": "finger", "left_arm_orthodox": "finger",
    "wrist_spin": "wrist", "left_arm_wristspin": "wrist",
}
NOT_BOWLER = {"run out", "retired hurt", "retired out", "obstructing the field", "timed out"}
XI_FEATURES = ("xi_quality", "xi_matchup", "xi_matchup_pitch")


def _new_cells():
    return {g: [0, 0, 0] for g in GROUPS}


class PlayerBook:
    """Cumulative player records. bat[p] = per-group [balls, runs, outs] + appearances;
    bowl[p] = [balls, runs, wkts] + appearances."""

    def __init__(self) -> None:
        self.bat = defaultdict(lambda: {"cells": _new_cells(), "apps": 0, "balls": 0})
        self.bowl = defaultdict(lambda: {"cells": [0, 0, 0], "apps": 0})
        self.g_bat = _new_cells()   # league batting vs group
        self.g_bowl = [0, 0, 0]

    # league reference rates -------------------------------------------------
    def _rate(self, cell) -> float:
        balls, runs, outs = cell
        return (runs - LAMBDA * outs) / balls if balls else 0.0

    def league_all(self) -> float:
        tot = [sum(self.g_bat[g][i] for g in GROUPS) for i in range(3)]
        return self._rate(tot)

    # batter value ---------------------------------------------------------------
    def batter_overall(self, name: str) -> float:
        rec = self.bat.get(name)
        league = self.league_all()
        if rec is None:
            return 0.0
        tot = [sum(rec["cells"][g][i] for g in GROUPS) for i in range(3)]
        if tot[0] == 0:
            return 0.0
        return (self._rate(tot) * tot[0] + PSEUDO_ALL * league) / (tot[0] + PSEUDO_ALL) - league

    def batter_vs(self, name: str, group: str) -> float:
        """Value vs group relative to the league's value vs that group."""
        league_g = self._rate(self.g_bat[group])
        prior = self.batter_overall(name)  # skill carries over across groups
        rec = self.bat.get(name)
        cell = rec["cells"][group] if rec else [0, 0, 0]
        n = cell[0]
        rate_dev = (self._rate(cell) - league_g) if n else 0.0
        return (n * rate_dev + PSEUDO_GROUP * prior) / (n + PSEUDO_GROUP)

    def batter_weight(self, name: str) -> float:
        rec = self.bat.get(name)
        if rec is None or rec["apps"] == 0:
            return DEFAULT_WEIGHT
        return max(1.0, rec["balls"] / rec["apps"])

    # bowler -----------------------------------------------------------------------
    def bowler_quality(self, name: str) -> float:
        """Negative runs-per-ball net of wickets allowed vs league: higher = better."""
        rec = self.bowl.get(name)
        league = self._rate(self.g_bowl)
        if rec is None or rec["cells"][0] == 0:
            return 0.0
        b = rec["cells"]
        # bowler "net" = runs conceded minus wicket value; lower is better
        r = (b[1] - LAMBDA * b[2]) / b[0]
        shrunk = (r * b[0] + PSEUDO_ALL * league) / (b[0] + PSEUDO_ALL)
        return league - shrunk

    def bowler_weight(self, name: str) -> float:
        rec = self.bowl.get(name)
        if rec is None or rec["apps"] == 0:
            return 0.0
        return rec["cells"][0] / rec["apps"]


def _bowler_group(registry, cache, name):
    if name not in cache:
        cache[name] = STYLE_GROUP.get(registry.lookup(name)["bowling_style"])
    return cache[name]


def team_view(book: PlayerBook, xi: list[str], registry, cache) -> dict:
    """Summary of one XI: batting values per group, overall quality, attack mix."""
    weights = [book.batter_weight(p) for p in xi]
    wsum = sum(weights) or 1.0
    bat_overall = sum(w * book.batter_overall(p) for w, p in zip(weights, xi)) / wsum
    bat_group = {
        g: sum(w * book.batter_vs(p, g) for w, p in zip(weights, xi)) / wsum for g in GROUPS
    }
    bowl_w, mix, bowl_quality, bsum = {}, {g: 0.0 for g in GROUPS}, 0.0, 0.0
    for p in xi:
        group = _bowler_group(registry, cache, p)
        w = book.bowler_weight(p)
        if group is None or w <= 0:
            continue
        mix[group] += w
        bowl_quality += w * book.bowler_quality(p)
        bsum += w
    if bsum:
        mix = {g: v / bsum for g, v in mix.items()}
        bowl_quality /= bsum
    else:
        mix = {g: 1.0 / len(GROUPS) for g in GROUPS}
    return {"bat_overall": bat_overall, "bat_group": bat_group, "mix": mix, "bowl_quality": bowl_quality}


def matchup(bat: dict, opp: dict, tilt: dict | None = None) -> float:
    """How much `bat`'s XI gains or loses vs its own average from the opponent's attack mix."""
    mix = dict(opp["mix"])
    if tilt:
        mix = {g: mix[g] * tilt[g] for g in GROUPS}
        total = sum(mix.values()) or 1.0
        mix = {g: v / total for g, v in mix.items()}
    return sum(mix[g] * (bat["bat_group"][g] - bat["bat_overall"]) for g in GROUPS)


def pitch_tilt(spin: float) -> dict:
    """spin-lean > 0 means spin takes wickets more often than usual: weight spin overs up."""
    f = math.exp(max(-1.0, min(1.0, spin)))
    return {"pace": 1.0 / f, "finger": f, "wrist": f}


def _parse(path: Path, source: str, registry, cache):
    match = json.loads(path.read_text(encoding="utf-8"))
    info = match["info"]
    if info.get("gender") not in (None, "male") or len(info.get("teams", [])) != 2:
        return None
    teams = info["teams"]
    players = info.get("players", {})
    if any(t not in players or len(players[t]) < 9 for t in teams):
        return None
    city = (str(info.get("city") or "").strip().lower()
            or str(info.get("venue") or "").rsplit(",", 1)[-1].strip().lower())
    bat_updates = defaultdict(lambda: _new_cells())  # player -> group cells (this match)
    faced = defaultdict(int)
    bowl_updates = defaultdict(lambda: [0, 0, 0])
    team_cells = {t: {"bat": _new_cells(), "bowl": _new_cells()} for t in teams}
    for innings in match.get("innings", []):
        team = innings.get("team")
        if team not in teams:
            continue
        other = next(t for t in teams if t != team)
        for over in innings.get("overs", []):
            for d in over.get("deliveries", []):
                extras = d.get("extras", {})
                runs = d["runs"]
                wide = "wides" in extras
                noball = "noballs" in extras
                group = _bowler_group(registry, cache, d["bowler"])
                bowler_wkt = any(w.get("kind") not in NOT_BOWLER for w in d.get("wickets", []))
                out = any(w.get("player_out") == d["batter"] and w.get("kind") not in NOT_BOWLER
                          for w in d.get("wickets", []))
                if not wide:
                    faced[d["batter"]] += 1
                if group is not None and not wide:
                    cell = bat_updates[d["batter"]][group]
                    cell[0] += 1; cell[1] += runs["batter"]; cell[2] += int(out)
                    for table, side in ((team_cells[team]["bat"], None), (team_cells[other]["bowl"], None)):
                        c = table[group]
                        c[0] += 0 if noball else 1
                        c[1] += runs["batter"]
                        c[2] += int(bowler_wkt)
                if not wide and not noball:
                    bu = bowl_updates[d["bowler"]]
                    bu[0] += 1
                    bu[1] += runs["total"] - extras.get("byes", 0) - extras.get("legbyes", 0)
                    bu[2] += int(bowler_wkt)
                elif noball or wide:
                    bu = bowl_updates[d["bowler"]]
                    bu[1] += runs["total"] - extras.get("byes", 0) - extras.get("legbyes", 0)
    outcome = info.get("outcome", {})
    first = (match.get("innings") or [{}])[0].get("team")
    return {
        "src": source, "date": info["dates"][0], "city": city, "teams": teams,
        "winner": outcome.get("winner"), "first_team": first,
        "xi": {t: list(players[t]) for t in teams},
        "bat_updates": {p: c for p, c in bat_updates.items()},
        "faced": dict(faced), "bowl_updates": dict(bowl_updates),
        "team_cells": team_cells,
    }


def _match_row(book, cond, registry, cache, fact, day) -> dict:
    """Features for one match from the state so far (before folding the match in)."""
    teams = fact["teams"]
    a, b = sorted(teams)
    sign = 0
    if fact["first_team"] in (a, b):
        sign = 1 if fact["first_team"] == a else -1
    base = cond.base.features(a, b, fact["city"], 0)
    view_a = team_view(book, fact["xi"][a], registry, cache)
    view_b = team_view(book, fact["xi"][b], registry, cache)
    tilt = pitch_tilt(cond.pitch_spin(fact["city"]))
    quality = (view_a["bat_overall"] + view_a["bowl_quality"]) - (view_b["bat_overall"] + view_b["bowl_quality"])
    mu = matchup(view_a, view_b) - matchup(view_b, view_a)
    mu_p = matchup(view_a, view_b, tilt) - matchup(view_b, view_a, tilt)
    return {
        "date": fact["date"], "a": a, "b": b, "city": fact["city"],
        "y": 1 if fact["winner"] == a else 0, "sign": sign,
        "x": {
            "elo": base[0], "h2h": base[2], "h2h5": base[3],
            "dew_chase": dew_index(day) * -sign,
            "xi_quality": quality * 20.0,
            "xi_matchup": mu * 20.0,
            "xi_matchup_pitch": mu_p * 20.0,
        },
    }


def _parsed_matches(registry, cache) -> list[dict]:
    parsed = []
    for source in ("t20i", "ipl"):
        for path in sorted((ROOT / "data" / "raw" / "cricsheet" / source).glob("*.json")):
            fact = _parse(path, source, registry, cache)
            if fact:
                parsed.append(fact)
    parsed.sort(key=lambda f: f["date"])
    return parsed


def build_dataset_xi(collect: bool = True):
    """One row per decided men's T20I, XI matchup features from earlier matches only.
    Returns the rows; `current_xi_state()` returns the end state for live use."""
    return _replay(collect)[0]


def _replay(collect: bool):
    registry = PlayerStyleRegistry()
    cache: dict = {}
    parsed = _parsed_matches(registry, cache)

    weather = load_weather()
    cond = ConditionsState()  # Elo / H2H and the city pitch profile
    book = PlayerBook()
    out = []
    for fact in parsed:
        teams = fact["teams"]
        decided = fact["winner"] and fact["winner"] in teams
        if collect and fact["src"] == "t20i" and decided:
            out.append(_match_row(book, cond, registry, cache, fact, weather.get(fact["city"], {}).get(fact["date"])))
        # fold the match in
        cond.apply({
            "src": fact["src"], "teams": teams, "city": fact["city"], "winner": fact["winner"],
            "first_team": fact["first_team"],
            "bat": {t: fact["team_cells"][t]["bat"] for t in teams},
            "bowl": {t: fact["team_cells"][t]["bowl"] for t in teams},
        })
        for team in teams:
            for p in fact["xi"][team]:
                book.bat[p]["apps"] += 1
                book.bowl[p]["apps"] += 1
        for p, cells in fact["bat_updates"].items():
            rec = book.bat[p]
            for g in GROUPS:
                for i in range(3):
                    rec["cells"][g][i] += cells[g][i]
                    book.g_bat[g][i] += cells[g][i]
        for p, n in fact["faced"].items():
            book.bat[p]["balls"] += n
        for p, cells in fact["bowl_updates"].items():
            rec = book.bowl[p]
            for i in range(3):
                rec["cells"][i] += cells[i]
                book.g_bowl[i] += cells[i]
    return out, book, cond, registry, cache


_STATE = None


def current_xi_state():
    """(PlayerBook, ConditionsState, registry, cache) after every local match; cached."""
    global _STATE
    if _STATE is None:
        _, book, cond, registry, cache = _replay(False)
        _STATE = (book, cond, registry, cache)
    return _STATE


_TEAM_PLAYERS = None


def _team_pools():
    """team -> Cricsheet player names seen for that team since 2022; plus all names."""
    global _TEAM_PLAYERS
    if _TEAM_PLAYERS is None:
        pools: dict[str, set] = defaultdict(set)
        every: set = set()
        for source in ("t20i", "ipl"):
            for path in (ROOT / "data" / "raw" / "cricsheet" / source).glob("*.json"):
                info = json.loads(path.read_text(encoding="utf-8"))["info"]
                if info.get("gender") not in (None, "male") or info["dates"][0] < "2022-01-01":
                    continue
                for team, names in info.get("players", {}).items():
                    every.update(names)
                    if source == "t20i":
                        pools[team].update(names)
        _TEAM_PLAYERS = (pools, every)
    return _TEAM_PLAYERS


def resolve_player(full_name: str, team: str) -> str | None:
    """Live full name ("Shai Hope") -> Cricsheet unique name ("SD Hope"): same surname and
    first initial, preferring the team's own recent players; None if absent or ambiguous."""
    tokens = full_name.split()
    if len(tokens) < 2:
        return None
    pools, every = _team_pools()

    def hits(pool):
        return [n for n in pool if n.split()[-1].lower() == tokens[-1].lower()
                and n.split()[0][0].lower() == tokens[0][0].lower()]

    found = hits(pools.get(team, ()))
    if len(found) != 1:
        found = found or hits(every)
    return found[0] if len(found) == 1 else None


XI_MODEL_PATH = ROOT / "models" / "prematch_t20i_xi.json"


def predict_with_xi(team_a, team_b, bat_first, weather_day, xi_a, xi_b, venue_city_name):
    """Win probability from Elo + H2H + dew + XI quality + XI spin/pace matchup.
    xi_a / xi_b are the announced XIs as full names. Returns None without a saved
    model, toss, or at least 9 resolvable players per side."""
    from app.live.prematch_model import sigmoid

    try:
        saved = json.loads(XI_MODEL_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    book, cond, registry, cache = current_xi_state()
    if team_a not in cond.base.ratings or team_b not in cond.base.ratings or not bat_first:
        return None
    resolved = {team: [resolve_player(n, team) for n in names] for team, names in ((team_a, xi_a), (team_b, xi_b))}
    if any(sum(p is not None for p in names) < 9 for names in resolved.values()):
        return None
    a, b = sorted((team_a, team_b))
    if bat_first not in (a, b):
        return None
    sign = 1 if bat_first == a else -1
    xi = {team: [p for p in names if p] for team, names in resolved.items()}
    view = {t: team_view(book, xi[t], registry, cache) for t in (a, b)}
    tilt = pitch_tilt(cond.pitch_spin(venue_city_name))
    base = cond.base.features(a, b, venue_city_name, 0)

    def features(use_quality: bool, use_matchup: bool, use_dew: bool = True):
        quality = (view[a]["bat_overall"] + view[a]["bowl_quality"]) - (view[b]["bat_overall"] + view[b]["bowl_quality"])
        mu_p = matchup(view[a], view[b], tilt) - matchup(view[b], view[a], tilt)
        x = {"elo": base[0], "h2h": base[2], "h2h5": base[3], "dew_chase": dew_index(weather_day) * -sign if use_dew else 0.0,
             "xi_quality": quality * 20.0 if use_quality else 0.0,
             "xi_matchup_pitch": mu_p * 20.0 if use_matchup else 0.0}
        return x

    def prob(x):
        z = sum(w * x[n] / sc for n, w, sc in zip(saved["features"], saved["weights"], saved["scale"]))
        p_a = sigmoid(z)
        return p_a if team_a == a else 1.0 - p_a

    p = prob(features(True, True))
    p_none = prob(features(False, False))
    p_quality = prob(features(True, False))
    p_base = prob(features(False, False, use_dew=False))  # strength + H2H only, this model's weights
    return {
        "p_a": p, "p_b": 1.0 - p, "xi_shift": p - p_none,
        "p_base": p_base, "dew_shift": p_none - p_base,
        "quality_shift": p_quality - p_none, "matchup_shift": p - p_quality,
        "matched": {t: len(xi[t]) for t in xi}, "holdout": saved.get("holdout", {}),
        "spin_lean": cond.pitch_spin(venue_city_name),
        "mix": {t: view[t]["mix"] for t in view},
    }


STAGES_PATH = ROOT / "models" / "prematch_t20i_stages.json"


def predict_stages(team_a, team_b, bat_first, weather_day, xi_a, xi_b, venue_city_name):
    """The single live win-probability model (see train_prematch_stages.py).

    Built in blocks on one scale: base (Elo + head-to-head) is always applied; dew needs the toss
    (who chases) and a forecast; the XI block needs both XIs (>= 9 resolvable players each) and the
    toss. Because each block adds to the previous logit, the base never changes between stages.
    Returns None without a saved model or for a team with no history.
    """
    from app.live.prematch_model import sigmoid

    try:
        saved = json.loads(STAGES_PATH.read_text(encoding="utf-8"))["blocks"]
    except (OSError, ValueError, KeyError):
        return None
    book, cond, registry, cache = current_xi_state()
    if team_a not in cond.base.ratings or team_b not in cond.base.ratings:
        return None
    a, b = sorted((team_a, team_b))
    sign = 0 if bat_first not in (a, b) else (1 if bat_first == a else -1)
    base_row = cond.base.features(a, b, venue_city_name, 0)
    values = {"elo": base_row[0], "h2h": base_row[2], "h2h5": base_row[3]}

    def block_logit(name, vals):
        block = saved[name]
        return sum(w * vals[f] / sc for f, w, sc in zip(block["features"], block["weights"], block["scale"]))

    def to_team_a(p_a):
        return p_a if team_a == a else 1.0 - p_a

    z_base = block_logit("base", values)
    z = z_base
    out = {"stage": "pre-toss", "dew_known": False, "matched": None, "mix": None}
    z_dew = z_base
    if sign:
        out["stage"] = "after toss"
        if weather_day is not None:
            out["dew_known"] = True
            z_dew = z_base + block_logit("dew", {"dew_chase": dew_index(weather_day) * -sign})
        z = z_dew
    z_xi = z_dew
    if sign and xi_a and xi_b:
        resolved = {t: [resolve_player(n, t) for n in names] for t, names in ((team_a, xi_a), (team_b, xi_b))}
        if all(sum(p is not None for p in names) >= 9 for names in resolved.values()):
            xi = {t: [p for p in names if p] for t, names in resolved.items()}
            view = {t: team_view(book, xi[t], registry, cache) for t in (a, b)}
            tilt = pitch_tilt(cond.pitch_spin(venue_city_name))
            quality = (view[a]["bat_overall"] + view[a]["bowl_quality"]) - (view[b]["bat_overall"] + view[b]["bowl_quality"])
            mu_p = matchup(view[a], view[b], tilt) - matchup(view[b], view[a], tilt)
            z_quality = z_dew + block_logit("xi", {"xi_quality": quality * 20.0, "xi_matchup_pitch": 0.0})
            z_xi = z_dew + block_logit("xi", {"xi_quality": quality * 20.0, "xi_matchup_pitch": mu_p * 20.0})
            z = z_xi
            out.update(
                stage="with XIs",
                matched={t: len(xi[t]) for t in xi},
                mix={t: view[t]["mix"] for t in view},
                quality_shift=to_team_a(sigmoid(z_quality)) - to_team_a(sigmoid(z_dew)),
                matchup_shift=to_team_a(sigmoid(z_xi)) - to_team_a(sigmoid(z_quality)),
            )
    p_final = to_team_a(sigmoid(z))
    out.update(
        p_a=p_final, p_b=1.0 - p_final,
        p_base=to_team_a(sigmoid(z_base)),
        dew_shift=to_team_a(sigmoid(z_dew)) - to_team_a(sigmoid(z_base)),
    )
    return out
