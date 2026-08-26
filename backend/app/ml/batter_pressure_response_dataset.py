"""Batter-specific pressure-response dismissal rate (2026-08-25).

User's idea: does a specific batter's OWN historical dismissal rate,
right after an over-level "pressure trigger" (a sudden run-rate stall),
predict wicket risk better than the generic pressure/context features
already in the model? Structurally similar to this project's successful
player-specific features (recency form, partnership rate) rather than
the several generic-context features that failed this session
(volatility in three formulations, cascade features).

**Pressure trigger** (leakage-safe -- uses only this SAME innings' own
prior overs, no cross-match info): the immediately preceding over's runs
were at least half below the run rate established up to that point
(`prev_over_runs <= 0.5 * run_rate_before_that_over`). Matches the user's
own example (RR=8.0, an over yields 4 -- exactly the trigger threshold).

**Player-specific rate is career-to-date** (all matches strictly before
this match's date, chronological, no leakage -- updated only BETWEEN
matches so a match's own rows never see its own outcomes), shrunk toward
the global pressure-triggered dismissal rate (small per-player samples
expected -- same shrinkage convention used everywhere else in this
project, e.g. H2H/venue features).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.candidate_v3_dataset import _is_legal, _is_wicket
from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS

SHRINKAGE_OVERS = 15.0


def build_batter_pressure_response_dataset(
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

    career_pressure_overs: dict[str, int] = {}
    career_pressure_dismissals: dict[str, int] = {}
    global_pressure_overs = 0
    global_pressure_dismissals = 0

    rows: list[dict[str, Any]] = []

    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        match_events: list[tuple[str, bool]] = []  # (striker, was_dismissed_this_over)

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
                striker = str(deliveries[0]["batter"])

                sample = career_pressure_overs.get(striker, 0)
                dismissals = career_pressure_dismissals.get(striker, 0)
                prior = (global_pressure_dismissals / global_pressure_overs) if global_pressure_overs else 0.0
                shrunk_rate = (dismissals + SHRINKAGE_OVERS * prior) / (sample + SHRINKAGE_OVERS)

                rows.append({
                    "source_file": path.name,
                    "match_date": match_date,
                    "innings": innings_number,
                    "over": over_number,
                    "pressure_trigger": pressure_trigger,
                    "striker_pressure_dismissal_rate_shrunk": shrunk_rate,
                    "striker_pressure_sample": sample,
                })

                over_runs = sum(int(d.get("runs", {}).get("total", 0)) for d in deliveries)
                legal_balls = sum(1 for d in deliveries if _is_legal(d))
                striker_dismissed = any(
                    _is_wicket(d) and any(w.get("player_out") == striker for w in d.get("wickets", []))
                    for d in deliveries
                )
                if pressure_trigger:
                    match_events.append((striker, striker_dismissed))

                cum_runs += over_runs
                cum_legal_balls += legal_balls
                prev_over_runs = over_runs

        # Only now (match fully processed) fold this match's own pressure
        # events into the global career state -- guarantees no row in
        # THIS match could ever have seen its own outcome.
        for striker, dismissed in match_events:
            career_pressure_overs[striker] = career_pressure_overs.get(striker, 0) + 1
            if dismissed:
                career_pressure_dismissals[striker] = career_pressure_dismissals.get(striker, 0) + 1
            global_pressure_overs += 1
            global_pressure_dismissals += int(dismissed)

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Batter pressure-response dataset contains duplicate over keys.")
    return result
