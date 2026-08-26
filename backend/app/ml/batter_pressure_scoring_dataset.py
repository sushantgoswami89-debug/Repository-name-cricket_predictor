"""Batter-specific pressure SCORING response (2026-08-26). Parallel to
app/ml/batter_pressure_response_dataset.py (which tracked dismissal rate
under pressure, for the wicket target) -- this tracks the same batter's
own historical RUN RATE specifically in overs immediately following a
pressure trigger, for the RUN-RANGE target. Never tested before: does a
batter slow down (survive-mode) or accelerate (counter-attack) under
pressure, and does that differ enough player-to-player to help predict
next-over runs? Matches the pattern that's actually worked for this
project (player-specific features), unlike generic pre-match context.

Same pressure trigger as the wicket-side test: the immediately preceding
over's runs were at least half below the run rate established up to that
point (prev_over_runs <= 0.5 * run_rate_before_that_over).

Player identity uses `canonical_player_id` (registry-based), matching
the established convention every other live-served player-prior feature
in this project uses (app/ml/ipl_phase_moe_dataset.py,
build_run_range_v3_live_snapshots.py) -- NOT raw Cricsheet delivery
names, which fragment across a player's different name spellings between
matches and would create a train/serve mismatch against the live
snapshot lookup (which must resolve the same way `_resolve_player_id`
does).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.candidate_v3_dataset import _is_legal
from app.ml.ipl_identities import canonical_player_id
from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS

SHRINKAGE_OVERS = 15.0


def build_batter_pressure_scoring_dataset(
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

    career_pressure_balls: dict[str, int] = {}
    career_pressure_runs: dict[str, int] = {}
    global_pressure_balls = 0
    global_pressure_runs = 0

    rows: list[dict[str, Any]] = []

    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        registry = raw.get("info", {}).get("registry", {}).get("people", {})
        match_events: list[tuple[str, int, int]] = []  # (striker, legal_balls, runs) in this pressure over

        for innings_number, innings in enumerate(raw.get("innings", []), start=1):
            cum_runs = 0
            cum_legal_balls = 0
            prev_over_runs: int | None = None
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                run_rate_before = (cum_runs / cum_legal_balls * 6) if cum_legal_balls else 0.0
                pressure_trigger = int(
                    prev_over_runs is not None
                    and run_rate_before > 0
                    and prev_over_runs <= 0.5 * run_rate_before
                )
                striker = canonical_player_id(str(deliveries[0]["batter"]), registry)

                sample_balls = career_pressure_balls.get(striker, 0)
                sample_runs = career_pressure_runs.get(striker, 0)
                prior = (global_pressure_runs / global_pressure_balls) if global_pressure_balls else 0.0
                shrunk_rate = (sample_runs + SHRINKAGE_OVERS * 6 * prior) / (sample_balls + SHRINKAGE_OVERS * 6)

                rows.append({
                    "source_file": path.name,
                    "match_date": match_date,
                    "innings": innings_number,
                    "over": over_number,
                    "striker_pressure_scoring_rate_shrunk": shrunk_rate,
                })

                striker_legal_balls = sum(
                    1 for d in deliveries if _is_legal(d) and canonical_player_id(str(d.get("batter")), registry) == striker
                )
                striker_runs = sum(
                    int(d.get("runs", {}).get("batter", 0))
                    for d in deliveries
                    if _is_legal(d) and canonical_player_id(str(d.get("batter")), registry) == striker
                )
                if pressure_trigger and striker_legal_balls > 0:
                    match_events.append((striker, striker_legal_balls, striker_runs))

                over_runs = sum(int(d.get("runs", {}).get("total", 0)) for d in deliveries)
                legal_balls = sum(1 for d in deliveries if _is_legal(d))
                cum_runs += over_runs
                cum_legal_balls += legal_balls
                prev_over_runs = over_runs

        for striker, balls, runs in match_events:
            career_pressure_balls[striker] = career_pressure_balls.get(striker, 0) + balls
            career_pressure_runs[striker] = career_pressure_runs.get(striker, 0) + runs
            global_pressure_balls += balls
            global_pressure_runs += runs

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Batter pressure-scoring dataset contains duplicate over keys.")
    return result
