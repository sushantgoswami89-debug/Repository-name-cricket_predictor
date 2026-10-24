"""Pre-match win probability for men's T20Is, built point-in-time from Cricsheet.

Every feature for a match is computed only from matches played BEFORE it, so
the training rows are honest (no look-ahead). Features are antisymmetric in
(team A - team B) and the logistic model has no intercept, so the alphabetical
A/B orientation cannot leak into the prediction.

Features (all "A minus B" unless noted):
  elo        Elo rating difference / 400
  form       mean over each team's last 5 games of (result - Elo-expected result),
             i.e. recent over/under-performance, opponent-adjusted
  h2h        (wins_A - wins_B) / (meetings + 3), all previous meetings, shrunk
  h2h5       same for the last 5 meetings only, shrunk by +2
  venue_t20i bat-first edge at the ground from T20Is, shrunk toward the global
             rate; multiplied by +1 if A bats first, -1 if A chases (post-toss only)
  venue_ipl  same from IPL matches at the same city (separate signal, user spec
             2026-10-06: small T20I venue samples are supplemented by IPL)

See evaluate_prematch_model.py for the backtest that decides whether this is
allowed into the briefing: it must beat plain Elo on the 2025+ holdout.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
T20I_DIR = ROOT / "data" / "raw" / "cricsheet" / "t20i"
IPL_DIR = ROOT / "data" / "raw" / "cricsheet" / "ipl"

# Elo: K shrinks with a team's experience (high while a rating is still settling, low once it is
# established) and each result is scaled by the margin of victory. Chosen 2026-10-10 from rolling-origin
# tests on 2021-25 (fit on earlier years only): Brier +0.0030 over the old fixed K=96 on all games
# (CI [+0.0012,+0.0048]; positive in both 2021-23 and 2024-25), +0.0018 on full-member games (n.s.).
ELO_K_MIN = 16.0
ELO_K_MAX = 96.0
ELO_K_GAMES = 30.0
ELO_K = ELO_K_MAX  # legacy name used by older scripts
ELO_INITIAL = 1500.0
WICKET_MARGIN_RUNS = 8.0  # a win "by N wickets" counts as about 8N runs


def margin_runs(outcome: dict) -> float:
    """Winning margin in runs-equivalent (10 when unknown: ties, super overs, no margin recorded)."""
    by = outcome.get("by", {}) or {}
    return float(by.get("runs") or (by.get("wickets", 0) * WICKET_MARGIN_RUNS) or 10.0)


def margin_multiplier(margin: float) -> float:
    return math.log(1.0 + margin / 10.0) + 0.5
VENUE_PRIOR = 12.0  # pseudo-matches pulling a venue's bat-first rate to the global rate
FEATURES = ("elo", "form", "h2h", "h2h5", "venue_t20i", "venue_ipl")


def city_key(info: dict) -> str:
    city = str(info.get("city") or "").strip().lower()
    if city:
        return city
    venue = str(info.get("venue") or "")
    return venue.rsplit(",", 1)[-1].strip().lower()


def venue_city(venue: str) -> str:
    """City from a TOI/Cricbuzz venue string ("<stadium>, <city>")."""
    return venue.rsplit(",", 1)[-1].strip().lower()


def _first_innings(match: dict) -> tuple[str | None, int | None]:
    innings = match.get("innings") or []
    if not innings:
        return None, None
    first = innings[0]
    total = sum(
        d["runs"]["total"] for over in first.get("overs", []) for d in over.get("deliveries", [])
    )
    return first.get("team"), total


def _load(directory: Path) -> tuple[dict, ...]:
    rows = []
    for path in directory.glob("*.json"):
        match = json.loads(path.read_text(encoding="utf-8"))
        info = match["info"]
        if info.get("gender") not in (None, "male"):
            continue
        first_team, first_total = _first_innings(match)
        rows.append(
            {
                "date": info["dates"][0],
                "teams": tuple(info.get("teams", [])),
                "city": city_key(info),
                "winner": info.get("outcome", {}).get("winner"),
                "margin": margin_runs(info.get("outcome", {})),
                "first_team": first_team,
                "first_total": first_total,
            }
        )
    rows.sort(key=lambda row: row["date"])
    return tuple(rows)


@lru_cache(maxsize=1)
def t20i_rows() -> tuple[dict, ...]:
    return _load(T20I_DIR)


@lru_cache(maxsize=1)
def ipl_rows() -> tuple[dict, ...]:
    return _load(IPL_DIR)


@dataclass
class VenueRecord:
    bat_first_wins: int = 0
    matches: int = 0

    def edge(self, global_rate: float) -> float:
        """Shrunk bat-first win rate minus 0.5, for the venue."""
        rate = (self.bat_first_wins + VENUE_PRIOR * global_rate) / (self.matches + VENUE_PRIOR)
        return rate - 0.5


@dataclass
class State:
    """Everything known up to a point in time."""

    ratings: dict[str, float] = field(default_factory=dict)
    residuals: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    h2h: dict[frozenset, list[str]] = field(default_factory=lambda: defaultdict(list))
    venue_t20i: dict[str, VenueRecord] = field(default_factory=lambda: defaultdict(VenueRecord))
    venue_ipl: dict[str, VenueRecord] = field(default_factory=lambda: defaultdict(VenueRecord))
    games: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    global_t20i: VenueRecord = field(default_factory=VenueRecord)
    global_ipl: VenueRecord = field(default_factory=VenueRecord)

    def rating(self, team: str) -> float:
        return self.ratings.get(team, ELO_INITIAL)

    def expected(self, a: str, b: str) -> float:
        return 1.0 / (1.0 + 10 ** (-(self.rating(a) - self.rating(b)) / 400.0))

    def features(self, a: str, b: str, city: str, a_bats_first: int) -> list[float]:
        """Feature vector for (a vs b) at `city`; a_bats_first is +1, -1, or 0 if
        the toss is not known (then the venue features are zero)."""
        meetings = self.h2h.get(frozenset((a, b)), [])
        wins_a = sum(w == a for w in meetings)
        wins_b = sum(w == b for w in meetings)
        last5 = meetings[-5:]
        l5a = sum(w == a for w in last5)
        l5b = sum(w == b for w in last5)
        form = lambda team: (
            sum(self.residuals[team][-5:]) / len(self.residuals[team][-5:])
            if self.residuals.get(team)
            else 0.0
        )
        g_t20i = (
            self.global_t20i.bat_first_wins / self.global_t20i.matches if self.global_t20i.matches else 0.5
        )
        g_ipl = self.global_ipl.bat_first_wins / self.global_ipl.matches if self.global_ipl.matches else 0.5
        return [
            (self.rating(a) - self.rating(b)) / 400.0,
            form(a) - form(b),
            (wins_a - wins_b) / (len(meetings) + 3.0),
            (l5a - l5b) / (len(last5) + 2.0),
            a_bats_first * self.venue_t20i[city].edge(g_t20i) if city in self.venue_t20i else 0.0,
            a_bats_first * self.venue_ipl[city].edge(g_ipl) if city in self.venue_ipl else 0.0,
        ]

    def apply_t20i(self, row: dict) -> None:
        a, b = row["teams"]
        winner = row["winner"]
        expected_a = self.expected(a, b)
        outcome_a = 1.0 if winner == a else 0.0
        self.residuals[a].append(outcome_a - expected_a)
        self.residuals[b].append((1 - outcome_a) - (1 - expected_a))
        mult = margin_multiplier(row.get("margin", 10.0))
        k_a = ELO_K_MIN + (ELO_K_MAX - ELO_K_MIN) * math.exp(-self.games[a] / ELO_K_GAMES)
        k_b = ELO_K_MIN + (ELO_K_MAX - ELO_K_MIN) * math.exp(-self.games[b] / ELO_K_GAMES)
        self.ratings[a] = self.rating(a) + k_a * mult * (outcome_a - expected_a)
        self.ratings[b] = self.rating(b) + k_b * mult * ((1 - outcome_a) - (1 - expected_a))
        self.games[a] += 1
        self.games[b] += 1
        self.h2h[frozenset((a, b))].append(winner)
        if row["first_team"] in (a, b):
            won_bat_first = int(winner == row["first_team"])
            self.venue_t20i[row["city"]].matches += 1
            self.venue_t20i[row["city"]].bat_first_wins += won_bat_first
            self.global_t20i.matches += 1
            self.global_t20i.bat_first_wins += won_bat_first

    def apply_ipl(self, row: dict) -> None:
        if not row["winner"] or row["first_team"] not in row["teams"]:
            return
        won_bat_first = int(row["winner"] == row["first_team"])
        self.venue_ipl[row["city"]].matches += 1
        self.venue_ipl[row["city"]].bat_first_wins += won_bat_first
        self.global_ipl.matches += 1
        self.global_ipl.bat_first_wins += won_bat_first


def build_dataset() -> list[dict]:
    """One row per decided men's T20I, features from strictly earlier matches.
    Alphabetical orientation: team_a < team_b."""
    state = State()
    ipl = [r for r in ipl_rows() if r["winner"]]
    ipl_at = 0
    out = []
    for row in t20i_rows():
        if not row["winner"] or len(row["teams"]) != 2 or row["winner"] not in row["teams"]:
            continue
        while ipl_at < len(ipl) and ipl[ipl_at]["date"] < row["date"]:
            state.apply_ipl(ipl[ipl_at])
            ipl_at += 1
        a, b = sorted(row["teams"])
        sign = 0
        if row["first_team"] in (a, b):
            sign = 1 if row["first_team"] == a else -1
        out.append(
            {
                "date": row["date"],
                "a": a,
                "b": b,
                "city": row["city"],
                "y": 1 if row["winner"] == a else 0,
                "sign": sign,
                "x": state.features(a, b, row["city"], sign),
                "x_pretoss": state.features(a, b, row["city"], 0),
                "n_a": len(state.residuals.get(a, [])),
                "n_b": len(state.residuals.get(b, [])),
                "elo_p": state.expected(a, b),
            }
        )
        state.apply_t20i({**row, "teams": (a, b)})
    return out


@lru_cache(maxsize=1)
def current_state() -> State:
    """State after every match in the local data (for live use)."""
    state = State()
    for row in ipl_rows():
        if row["winner"]:
            state.apply_ipl(row)
    for row in t20i_rows():
        if row["winner"] and len(row["teams"]) == 2 and row["winner"] in row["teams"]:
            a, b = sorted(row["teams"])
            state.apply_t20i({**row, "teams": (a, b)})
    return state


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


MODEL_PATH = ROOT / "models" / "prematch_t20i_logit.json"


def predict(team_a: str, team_b: str, venue: str = "", bat_first: str = "") -> dict | None:
    """Win probability for team_a vs team_b from the saved, backtested model, or
    None when no model is saved or a team has no history (e.g. Afghanistan in the
    local data). `bat_first` (a team name) switches on the venue features."""
    try:
        saved = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    state = current_state()
    if team_a not in state.ratings or team_b not in state.ratings:
        return None
    a, b = sorted((team_a, team_b))
    sign = 0 if not bat_first else (1 if bat_first == a else -1 if bat_first == b else 0)
    key = "weights_post_toss" if sign else "weights_pre_toss"
    names = saved["features"]
    weights = saved[key]
    x = state.features(a, b, venue_city(venue), sign)
    p_a = sigmoid(sum(w * x[FEATURES.index(n)] for n, w in zip(names, weights)))
    p_first = p_a if team_a == a else 1.0 - p_a
    return {
        "team_a": team_a,
        "p_a": p_first,
        "team_b": team_b,
        "p_b": 1.0 - p_first,
        "post_toss": bool(sign),
        "holdout": saved.get("holdout", {}),
    }
