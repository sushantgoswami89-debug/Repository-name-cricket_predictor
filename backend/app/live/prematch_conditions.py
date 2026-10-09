"""Conditions-based pre-match features (dew, overcast, pitch type x team matchups).

Not tied to a ground's win record: a ground only enters as a *pitch-type
measurement* (how much more often spin takes wickets than pace there, from
earlier matches), because the user's point (2026-10-09) is that dew, cloud and
spin/seam conditions act the same everywhere. Built on top of the v1 point-in-time
state (Elo, form, head-to-head) in app/live/prematch_model.py.

Extra features, all antisymmetric (team A minus team B) and computed only from
matches before the one being predicted:

  dew_chase     dew index x (+1 if A chases, -1 if A bats first)
  cloud_chase   afternoon cloud x (+1 if A chases)   (overcast helps the side bowling first)
  finger_x      pitch spin-lean x (A minus B finger-spin matchup value)
  wrist_x       pitch spin-lean x (A minus B wrist-spin matchup value)
  pace_x        (cloud - pitch spin-lean) x (A minus B pace matchup value)

Matchup value for a team against a bowler group = how well its batters score
against that group, net of wickets lost, plus how well its own bowlers of that
group do, each relative to the all-matches average and shrunk by sample size.
Weather comes from Open-Meteo (fetch_match_weather.py); missing weather = 0
(neutral). Cricsheet has no start times, so every match is assumed an evening
match for dew, which makes day games noisy.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from pathlib import Path

from app.live.prematch_model import State, venue_city

ROOT = Path(__file__).resolve().parents[3]
FACTS_PATH = ROOT / "data" / "features" / "prematch_match_facts.json"
WEATHER_PATH = ROOT / "data" / "external" / "weather" / "city_daily.json"
DEW_MODEL_PATH = ROOT / "models" / "prematch_t20i_dew.json"
GROUPS = ("pace", "finger", "wrist")
WINDOW = 20  # each team's last N T20Is feed its matchup profile
PSEUDO_BALLS = 150.0
PITCH_PSEUDO_BALLS = 300.0
WICKET_COST = 12.0  # runs a wicket is worth, to net scoring against dismissals

FEATURES_V2 = (
    "elo", "form", "h2h", "h2h5",
    "dew_chase", "cloud_chase", "finger_x", "wrist_x", "pace_x",
)
SIGN_FEATURES = {"dew_chase", "cloud_chase", "finger_x", "wrist_x", "pace_x"}


def load_weather() -> dict:
    try:
        return json.loads(WEATHER_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


DEW_GAP_C = 2.5  # evening temperature minus dew point at/below which dew is likely
DEW_WIDTH_C = 0.5


def dew_index(day: dict | None) -> float:
    """~1 when dew is likely, ~0 when not. A threshold, not a gradient: in pre-2025
    men's T20Is the chasing side won 56% when the evening temperature was within
    2.5C of the dew point (496 games) against about 49% otherwise, with no
    gradient above that. Threshold chosen on pre-2025 data only."""
    if not day or day.get("eve_temp") is None or day.get("eve_dew_point") is None:
        return 0.0
    gap = day["eve_temp"] - day["eve_dew_point"]
    return 1.0 / (1.0 + math.exp((gap - DEW_GAP_C) / DEW_WIDTH_C))


def cloud_index(day: dict | None) -> float:
    """-1 clear .. +1 fully overcast, from afternoon cloud cover."""
    if not day or day.get("aft_cloud") is None:
        return 0.0
    return (day["aft_cloud"] - 50.0) / 50.0


class Cells:
    """balls/runs/wickets per bowler group."""

    __slots__ = ("c",)

    def __init__(self) -> None:
        self.c = {g: [0, 0, 0] for g in GROUPS}

    def add(self, other: dict) -> None:
        for g in GROUPS:
            for i in range(3):
                self.c[g][i] += other[g][i]


class ConditionsState:
    def __init__(self) -> None:
        self.base = State()
        self.global_cells = Cells()
        self.bat_window: dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW))
        self.bowl_window: dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW))
        self.city_cells: dict[str, Cells] = defaultdict(Cells)

    # -- matchup values -------------------------------------------------
    def _net(self, cells: list, group: str) -> float:
        balls, runs, wkts = cells
        gb, gr, gw = self.global_cells.c[group]
        if gb == 0:
            return 0.0
        g_rpb, g_wpb = gr / gb, gw / gb
        rpb = (runs + PSEUDO_BALLS * g_rpb) / (balls + PSEUDO_BALLS)
        wpb = (wkts + PSEUDO_BALLS * g_wpb) / (balls + PSEUDO_BALLS)
        return (rpb - WICKET_COST * wpb) - (g_rpb - WICKET_COST * g_wpb)

    def _sum(self, window: deque, group: str) -> list:
        total = [0, 0, 0]
        for entry in window:
            for i in range(3):
                total[i] += entry[group][i]
        return total

    def team_value(self, team: str, group: str) -> float:
        """Batting value against the group plus own bowlers' value of that group."""
        bat = self._net(self._sum(self.bat_window[team], group), group)
        bowl = -self._net(self._sum(self.bowl_window[team], group), group)
        return bat + bowl

    # -- pitch type -------------------------------------------------------
    def pitch_spin(self, city: str) -> float:
        cells = self.city_cells.get(city)
        if cells is None:
            return 0.0

        def wpb(group_names, source):
            b = sum(source.c[g][0] for g in group_names)
            w = sum(source.c[g][2] for g in group_names)
            return b, w

        gb_s, gw_s = wpb(("finger", "wrist"), self.global_cells)
        gb_p, gw_p = wpb(("pace",), self.global_cells)
        if min(gb_s, gb_p, gw_s, gw_p) == 0:
            return 0.0
        b_s, w_s = wpb(("finger", "wrist"), cells)
        b_p, w_p = wpb(("pace",), cells)
        rate_s = (w_s + PITCH_PSEUDO_BALLS * gw_s / gb_s) / (b_s + PITCH_PSEUDO_BALLS)
        rate_p = (w_p + PITCH_PSEUDO_BALLS * gw_p / gb_p) / (b_p + PITCH_PSEUDO_BALLS)
        return math.log(rate_s / rate_p) - math.log((gw_s / gb_s) / (gw_p / gb_p))

    # -- features -----------------------------------------------------------
    def features(self, a: str, b: str, city: str, sign: int, day: dict | None) -> list[float]:
        v1 = self.base.features(a, b, city, 0)
        elo, form, h2h, h2h5 = v1[0], v1[1], v1[2], v1[3]
        a_chases = -sign  # sign = +1 when A bats first; 0 when the toss is unknown
        spin = self.pitch_spin(city)
        cloud = cloud_index(day)
        diff = {g: self.team_value(a, g) - self.team_value(b, g) for g in GROUPS}
        return [
            elo, form, h2h, h2h5,
            dew_index(day) * a_chases,
            cloud * a_chases,
            spin * diff["finger"] * 20.0,
            spin * diff["wrist"] * 20.0,
            (cloud - spin) * diff["pace"] * 20.0,
        ]

    def apply(self, fact: dict) -> None:
        """Fold a finished match into the state."""
        teams = fact["teams"]
        cells_by_city = self.city_cells[fact["city"]]
        for team in teams:
            self.global_cells.add(fact["bat"][team])
            cells_by_city.add(fact["bat"][team])
        if fact["src"] != "t20i":
            return
        a, b = sorted(teams)
        if fact["winner"] and fact["winner"] in teams:
            self.base.apply_t20i(
                {"teams": (a, b), "winner": fact["winner"], "first_team": fact["first_team"], "city": fact["city"]}
            )
        for team in teams:
            self.bat_window[team].append(fact["bat"][team])
            self.bowl_window[team].append(fact["bowl"][team])


def load_facts() -> list[dict]:
    return json.loads(FACTS_PATH.read_text(encoding="utf-8"))


def build_dataset_v2() -> list[dict]:
    """One row per decided men's T20I; every feature uses earlier matches only."""
    facts = load_facts()
    weather = load_weather()
    state = ConditionsState()
    out = []
    for fact in facts:
        teams = fact["teams"]
        decided = fact["winner"] and fact["winner"] in teams and len(teams) == 2
        if fact["src"] == "t20i" and decided:
            a, b = sorted(teams)
            sign = 0
            if fact["first_team"] in (a, b):
                sign = 1 if fact["first_team"] == a else -1
            day = weather.get(fact["city"], {}).get(fact["date"])
            out.append(
                {
                    "date": fact["date"],
                    "a": a,
                    "b": b,
                    "city": fact["city"],
                    "y": 1 if fact["winner"] == a else 0,
                    "sign": sign,
                    "has_weather": day is not None,
                    "x": state.features(a, b, fact["city"], sign, day),
                    "x_pretoss": state.features(a, b, fact["city"], 0, day),
                    "elo_p": state.base.expected(a, b),
                }
            )
        state.apply(fact)
    return out


def predict_with_dew(team_a: str, team_b: str, bat_first: str, weather_day: dict | None) -> dict | None:
    """Win probability that adds the dew effect (chasing side favoured when dew is
    likely) to Elo + head-to-head. Needs the toss (who chases) and a weather
    summary for the match day; returns None otherwise or if no model is saved."""
    from app.live.prematch_model import current_state, sigmoid

    try:
        saved = json.loads(DEW_MODEL_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    state = current_state()
    if not bat_first or weather_day is None or team_a not in state.ratings or team_b not in state.ratings:
        return None
    a, b = sorted((team_a, team_b))
    if bat_first not in (a, b):
        return None
    sign = 1 if bat_first == a else -1
    v1 = state.features(a, b, "", 0)
    dew = dew_index(weather_day)
    x = [v1[0], v1[2], v1[3], dew * -sign]
    z = sum(w * xi / sc for w, xi, sc in zip(saved["weights"], x, saved["scale"]))
    p_a = sigmoid(z)
    p_first = p_a if team_a == a else 1.0 - p_a
    z0 = sum(w * xi / sc for w, xi, sc in zip(saved["weights"][:3], x[:3], saved["scale"][:3]))
    p0_a = sigmoid(z0)
    p0_first = p0_a if team_a == a else 1.0 - p0_a
    return {
        "p_a": p_first,
        "p_b": 1.0 - p_first,
        "dew": dew,
        "dew_shift": p_first - p0_first,
        "holdout": saved.get("holdout", {}),
    }
