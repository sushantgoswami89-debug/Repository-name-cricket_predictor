"""Batter pressure-survival duration (2026-08-25). User's refined idea:
not just "was the last over a stall" (already tested, mixed result) but
"how many consecutive overs has THIS batter already survived while the
team has been behind the required rate (current_run_rate < required_run_rate)
during a chase -- and does THIS batter's own history say they tend to get
out once they've been stuck there a while." A survival/duration framing,
not a one-shot trigger -- the "survival/hazard framing for wicket timing"
option flagged 2026-08-24 and never attempted, applied here as a concrete
feature rather than a full architecture change.

Two features:
  - `overs_under_pressure_this_spell`: LIVE, real-time, no history needed
    -- consecutive overs (this striker specifically) facing
    current_run_rate < required_run_rate during a chase, resets to 0 when
    pressure ends OR the striker changes.
  - `striker_extended_pressure_dismissal_rate_shrunk`: this batter's own
    career-to-date (chronological, no leakage) dismissal rate specifically
    on overs where they were already 2+ overs into such a spell -- shrunk
    toward the global rate for that same bucket.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.candidate_v3_dataset import _is_legal, _is_wicket
from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS

SHRINKAGE_OVERS = 15.0
EXTENDED_SPELL_THRESHOLD = 2  # overs already survived under pressure


def build_pressure_survival_dataset(
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

    career_extended_overs: dict[str, int] = {}
    career_extended_dismissals: dict[str, int] = {}
    global_extended_overs = 0
    global_extended_dismissals = 0

    rows: list[dict[str, Any]] = []

    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        regular_innings = [inn for inn in raw.get("innings", []) if not inn.get("super_over")]
        match_events: list[tuple[str, bool]] = []  # (striker, dismissed) for overs at/above the extended threshold

        first_innings_total: int | None = None
        for innings_number, innings in enumerate(regular_innings, start=1):
            target = first_innings_total + 1 if innings_number == 2 and first_innings_total is not None else 0
            cum_runs = 0
            cum_legal_balls = 0
            spell_striker = ""
            spell_length = 0

            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                is_chase = target > 0
                striker = str(deliveries[0]["batter"])

                current_run_rate = (cum_runs / cum_legal_balls * 6) if cum_legal_balls else 0.0
                balls_bowled_estimate = cum_legal_balls
                runs_required = max(0, target - cum_runs) if is_chase else 0
                balls_remaining_estimate = max(1, 120 - balls_bowled_estimate)
                required_run_rate = (runs_required * 6 / balls_remaining_estimate) if is_chase else 0.0
                under_pressure = int(is_chase and current_run_rate < required_run_rate)

                if under_pressure and striker == spell_striker:
                    spell_length += 1
                elif under_pressure:
                    spell_striker = striker
                    spell_length = 1
                else:
                    spell_striker = ""
                    spell_length = 0

                sample = career_extended_overs.get(striker, 0)
                dismissals = career_extended_dismissals.get(striker, 0)
                prior = (global_extended_dismissals / global_extended_overs) if global_extended_overs else 0.0
                shrunk_rate = (dismissals + SHRINKAGE_OVERS * prior) / (sample + SHRINKAGE_OVERS)

                rows.append({
                    "source_file": path.name,
                    "match_date": match_date,
                    "innings": innings_number,
                    "over": int(source_over["over"]) + 1,
                    "overs_under_pressure_this_spell": spell_length,
                    "striker_extended_pressure_dismissal_rate_shrunk": shrunk_rate,
                })

                striker_dismissed = any(
                    _is_wicket(d) and any(w.get("player_out") == striker for w in d.get("wickets", []))
                    for d in deliveries
                )
                if under_pressure and spell_length >= EXTENDED_SPELL_THRESHOLD:
                    match_events.append((striker, striker_dismissed))

                over_runs = sum(int(d.get("runs", {}).get("total", 0)) for d in deliveries)
                legal_balls = sum(1 for d in deliveries if _is_legal(d))
                cum_runs += over_runs
                cum_legal_balls += legal_balls

            if innings_number == 1:
                first_innings_total = cum_runs

        for striker, dismissed in match_events:
            career_extended_overs[striker] = career_extended_overs.get(striker, 0) + 1
            if dismissed:
                career_extended_dismissals[striker] = career_extended_dismissals.get(striker, 0) + 1
            global_extended_overs += 1
            global_extended_dismissals += int(dismissed)

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Pressure survival dataset contains duplicate over keys.")
    return result
