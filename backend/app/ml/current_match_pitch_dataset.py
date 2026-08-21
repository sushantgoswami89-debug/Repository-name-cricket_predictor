"""Leakage-safe observed pitch behaviour within the current match."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.candidate_v3_dataset import _is_legal, _is_wicket, _phase
from app.ml.innings_phase_player_dataset import bowling_category
from app.ml.venue_track_dataset import METRICS, _empty
from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS


def _add_rates(row: dict[str, Any], prefix: str, profile: dict[str, int]) -> None:
    balls = profile["balls"]
    row[f"{prefix}_samples"] = balls
    for name in METRICS:
        row[f"{prefix}_{name}_rate"] = profile[name] / balls if balls else 0.0


def build_current_match_pitch_dataset(project_root: Path) -> pd.DataFrame:
    styles = pd.read_csv(
        project_root / "data/candidates/v3.8/bowling_styles.csv"
    ).fillna("")
    style_by_id = {
        str(row.cricsheet_id): bowling_category(str(row.bowling_style))[0]
        for row in styles.itertuples()
    }
    paths: list[tuple[str, Path]] = []
    excluded = set(KNOWN_RULE_ANOMALY_EXCLUSIONS)
    for scope in ("t20i", "ipl"):
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json"):
            if path.stem in excluded:
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            paths.append((str(raw["info"]["dates"][0]), path))
    paths.sort(key=lambda item: (item[0], item[1].name))

    rows: list[dict[str, Any]] = []
    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        registry = raw["info"].get("registry", {}).get("people", {})
        match_profile = _empty()
        style_profiles: dict[str, dict[str, int]] = defaultdict(_empty)
        first_innings_profile = _empty()
        for innings_number, innings in enumerate(raw.get("innings", []), start=1):
            innings_profile = _empty()
            phase_profiles: dict[str, dict[str, int]] = defaultdict(_empty)
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                phase = _phase(over_number)
                row: dict[str, Any] = {
                    "source_file": path.name,
                    "match_date": match_date,
                    "innings": innings_number,
                    "over": over_number,
                }
                _add_rates(row, "match_pitch_innings", innings_profile)
                _add_rates(row, "match_pitch_phase", phase_profiles[phase])
                _add_rates(row, "match_pitch_match", match_profile)
                _add_rates(row, "match_pitch_first_innings", first_innings_profile)
                pace = _empty()
                for style in ("fast", "medium"):
                    for name, value in style_profiles[style].items():
                        pace[name] += value
                spin = _empty()
                for style in ("leg_spin", "off_spin"):
                    for name, value in style_profiles[style].items():
                        spin[name] += value
                _add_rates(row, "match_pitch_pace", pace)
                _add_rates(row, "match_pitch_spin", spin)
                rows.append(row)

                for delivery in deliveries:
                    if not _is_legal(delivery):
                        continue
                    bowler = str(delivery["bowler"])
                    style = style_by_id.get(str(registry.get(bowler, "")), "unknown")
                    batter_runs = int(delivery["runs"]["batter"])
                    total_runs = int(delivery["runs"]["total"])
                    event = {
                        "balls": 1,
                        "runs": total_runs,
                        "boundaries": int(batter_runs in {4, 6}),
                        "dots": int(total_runs == 0),
                        "wickets": int(_is_wicket(delivery)),
                    }
                    for profile in (
                        innings_profile,
                        phase_profiles[phase],
                        match_profile,
                        style_profiles[style],
                    ):
                        for name, value in event.items():
                            profile[name] += value
            if innings_number == 1:
                first_innings_profile = innings_profile.copy()

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Current-match pitch dataset contains duplicate over keys.")
    return result.sort_values(
        ["match_date", "source_file", "innings", "over"]
    ).reset_index(drop=True)
