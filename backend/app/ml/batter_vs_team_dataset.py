"""Batter's own historical scoring rate against a specific OPPONENT TEAM
(2026-08-26, user-requested). Genuinely different from what already
exists: `h2h_avg_runs`/`h2h_wicket_rate` are batter-vs-a-SPECIFIC-BOWLER;
`h2h_batting_team_win_rate_shrunk` (team_h2h_dataset.py) is TEAM-vs-TEAM
win rate. This is the batter's own record against every bowler from a
given opponent team collectively -- e.g. "how does THIS batter perform
whenever facing Australia specifically," not any one Australian bowler.

User's framing: recent (general) form and historical form-against-this-
team should be "weighted but not hard-coded" -- implemented as two
SEPARATE input features (this one + the existing recency-weighted
general-form features already live), letting the GBM's own learned
splits decide the combination rather than a hand-picked blend formula.

Player identity uses `canonical_player_id` (registry-based) -- the
2026-08-26 lesson from the pressure-scoring feature, where raw Cricsheet
names fragmented a player's history and produced a spurious result. Team
identity uses the raw team name string, matching this project's existing
convention (team_h2h_dataset.py) -- team names don't have the same
spelling-variance problem player names do.

Career-to-date, chronological (updated only between matches -- no
leakage), shrunk toward the batter's OWN overall prior rate (not a
global pooled rate), since a specific opponent-team sample is often much
smaller than the batter's whole career.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.candidate_v3_dataset import _is_legal, _is_wicket
from app.ml.ipl_identities import canonical_player_id
from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS

SHRINKAGE_BALLS = 60.0


def build_batter_vs_team_dataset(
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

    # career_prior: (player) -> {"balls": int, "runs": int} -- overall,
    # any opponent. career_vs_team: (player, opponent_team) -> same.
    career_prior: dict[str, dict[str, int]] = {}
    career_vs_team: dict[tuple[str, str], dict[str, int]] = {}

    rows: list[dict[str, Any]] = []

    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        registry = raw.get("info", {}).get("registry", {}).get("people", {})
        teams = raw.get("info", {}).get("teams", [])
        match_prior_events: list[tuple[str, int, int]] = []
        match_vs_team_events: list[tuple[str, str, int, int]] = []

        for innings_number, innings in enumerate(raw.get("innings", []), start=1):
            batting_team = str(innings.get("team", ""))
            opponent_team = next((t for t in teams if t != batting_team), teams[-1] if teams else "")

            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                striker = canonical_player_id(str(deliveries[0]["batter"]), registry)

                prior = career_prior.get(striker, {"balls": 0, "runs": 0})
                prior_rate = (prior["runs"] / prior["balls"]) if prior["balls"] else 0.0
                vs_team = career_vs_team.get((striker, opponent_team), {"balls": 0, "runs": 0})
                vs_team_balls = vs_team["balls"]
                weight = vs_team_balls / (vs_team_balls + SHRINKAGE_BALLS)
                vs_team_raw = (vs_team["runs"] / vs_team_balls) if vs_team_balls else prior_rate
                shrunk_rate = weight * vs_team_raw + (1 - weight) * prior_rate

                rows.append({
                    "source_file": path.name,
                    "match_date": match_date,
                    "innings": innings_number,
                    "over": over_number,
                    "striker_vs_opponent_team_runs_per_ball_shrunk": shrunk_rate,
                    "striker_vs_opponent_team_balls": vs_team_balls,
                })

                striker_legal_runs = sum(
                    int(d.get("runs", {}).get("batter", 0))
                    for d in deliveries
                    if _is_legal(d) and canonical_player_id(str(d.get("batter")), registry) == striker
                )
                striker_legal_balls = sum(
                    1 for d in deliveries
                    if _is_legal(d) and canonical_player_id(str(d.get("batter")), registry) == striker
                )
                if striker_legal_balls > 0:
                    match_prior_events.append((striker, striker_legal_balls, striker_legal_runs))
                    match_vs_team_events.append((striker, opponent_team, striker_legal_balls, striker_legal_runs))

        for striker, balls, runs in match_prior_events:
            profile = career_prior.setdefault(striker, {"balls": 0, "runs": 0})
            profile["balls"] += balls
            profile["runs"] += runs
        for striker, opponent_team, balls, runs in match_vs_team_events:
            profile = career_vs_team.setdefault((striker, opponent_team), {"balls": 0, "runs": 0})
            profile["balls"] += balls
            profile["runs"] += runs

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Batter vs opponent-team dataset contains duplicate over keys.")
    return result
