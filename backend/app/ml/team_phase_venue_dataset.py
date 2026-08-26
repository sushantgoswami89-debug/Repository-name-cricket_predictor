"""Team's own average AND volatility of scoring in each (phase, venue)
bucket (2026-08-26, user-requested: "use RR, wicket in hands, Target, avg
of team in each phase and venue" -- then corrected: "not hardcoded, we
needed a random or volatile index"). RR, wickets-in-hand, and
target-derived state (current_run_rate, wickets_in_hand, runs_required,
required_run_rate) are already live inputs (BASE_FEATURES) and are not
rebuilt here -- the GBM combines them with the two new features below via
its own learned splits, not a hard-coded formula.

Genuinely new: no existing feature is TEAM x PHASE x VENUE. Existing venue
features are venue-wide (venue_par_score, venue_scoring_regime, not
team-specific) or team x venue as a home/away LABEL only
(batting_team_venue_context), with no scoring number attached.

Two outputs per pre-over row, both career-to-date and updated only
between matches (no leakage), hierarchically shrunk
(team, venue, phase) -> (team, phase) -> global (phase):
  - team_phase_venue_runs_per_over_shrunk: this team's own historical mean
    scoring rate in this phase, at this venue.
  - team_phase_venue_volatility_shrunk: the standard deviation of this
    team's own per-over run totals in this phase, at this venue -- a
    genuine "how volatile/unpredictable is this team's scoring here"
    index (not a fixed formula, not random noise -- a real statistic of
    spread), addressing the "random or volatile index" request literally
    as a volatility measure rather than a point estimate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.candidate_v3_dataset import _phase
from app.ml.ipl_venues import normalize_ipl_venue
from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS

SHRINKAGE_OVERS_VENUE = 20.0   # (team, venue, phase) -> (team, phase)
SHRINKAGE_OVERS_TEAM = 60.0    # (team, phase) -> global (phase)


def _empty_stat() -> dict[str, float]:
    return {"overs": 0.0, "runs": 0.0, "sumsq": 0.0}


def _mean(stat: dict[str, float]) -> float:
    return stat["runs"] / stat["overs"] if stat["overs"] else 0.0


def _std(stat: dict[str, float]) -> float:
    if stat["overs"] < 2:
        return 0.0
    mean = _mean(stat)
    variance = max(0.0, stat["sumsq"] / stat["overs"] - mean * mean)
    return math.sqrt(variance)


def build_team_phase_venue_dataset(
    project_root: Path, scopes: tuple[str, ...] = ("ipl", "t20i")
) -> pd.DataFrame:
    excluded = set(KNOWN_RULE_ANOMALY_EXCLUSIONS)
    paths: list[tuple[str, Path]] = []
    for scope in scopes:
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json"):
            if path.stem in excluded:
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            paths.append((str(raw["info"]["dates"][0]), path))
    paths.sort(key=lambda item: (item[0], item[1].name))

    global_phase: dict[str, dict[str, float]] = {}
    team_phase: dict[tuple[str, str], dict[str, float]] = {}
    team_venue_phase: dict[tuple[str, str, str], dict[str, float]] = {}

    rows: list[dict[str, Any]] = []

    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        venue = normalize_ipl_venue(str(raw.get("info", {}).get("venue", "")))
        match_events: list[tuple[str, str, int]] = []  # (team, phase, over_runs)

        for innings_number, innings in enumerate(raw.get("innings", []), start=1):
            team = str(innings.get("team", ""))

            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                phase = _phase(over_number)

                gp = global_phase.get(phase, _empty_stat())
                tp = team_phase.get((team, phase), _empty_stat())
                tvp = team_venue_phase.get((team, venue, phase), _empty_stat())

                tvp_overs = tvp["overs"]
                weight_venue = tvp_overs / (tvp_overs + SHRINKAGE_OVERS_VENUE)
                tp_overs = tp["overs"]
                weight_team = tp_overs / (tp_overs + SHRINKAGE_OVERS_TEAM)

                team_mean = weight_team * _mean(tp) + (1 - weight_team) * _mean(gp)
                team_std = weight_team * _std(tp) + (1 - weight_team) * _std(gp)
                venue_mean = weight_venue * _mean(tvp) + (1 - weight_venue) * team_mean
                venue_std = weight_venue * _std(tvp) + (1 - weight_venue) * team_std

                rows.append({
                    "source_file": path.name,
                    "match_date": match_date,
                    "innings": innings_number,
                    "over": over_number,
                    "team_phase_venue_runs_per_over_shrunk": venue_mean,
                    "team_phase_venue_volatility_shrunk": venue_std,
                    "team_phase_venue_overs": tvp_overs,
                })

                over_runs = sum(int(d.get("runs", {}).get("total", 0)) for d in deliveries)
                match_events.append((team, phase, over_runs))

        for team, phase, over_runs in match_events:
            for store, key in (
                (global_phase, phase),
                (team_phase, (team, phase)),
                (team_venue_phase, (team, venue, phase)),
            ):
                stat = store.setdefault(key, _empty_stat())
                stat["overs"] += 1
                stat["runs"] += over_runs
                stat["sumsq"] += over_runs * over_runs

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Team phase-venue dataset contains duplicate over keys.")
    return result
