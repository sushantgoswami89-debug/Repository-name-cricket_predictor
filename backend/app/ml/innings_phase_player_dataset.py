"""Pre-match player profiles by innings role, phase, and similar track."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.candidate_v3_dataset import _is_legal, _is_wicket, _phase
from app.ml.venue_track_dataset import METRICS, _empty, _rate, _track_type
from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS


def bowling_category(style: str) -> tuple[str, str]:
    normalized = style.lower().replace("-", " ")
    arm = (
        "left"
        if "left" in normalized
        else "right"
        if "right" in normalized
        else "unknown"
    )
    if any(
        value in normalized
        for value in ("legbreak", "leg break", "wrist spin", "chinaman")
    ):
        return "leg_spin", arm
    if any(value in normalized for value in ("offbreak", "off break", "orthodox")):
        return "off_spin", arm
    if "fast" in normalized:
        return "fast", arm
    if "medium" in normalized:
        return "medium", arm
    return "unknown", arm


def build_innings_phase_player_dataset(project_root: Path) -> pd.DataFrame:
    style_frame = pd.read_csv(
        project_root / "data/candidates/v3.8/bowling_styles.csv"
    ).fillna("")
    style_by_id = {
        str(row.cricsheet_id): bowling_category(str(row.bowling_style))
        for row in style_frame.itertuples()
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

    global_profile = _empty()
    venues: dict[str, dict[str, int]] = defaultdict(_empty)
    context_priors: dict[tuple[int, str, str, str, str], dict[str, int]] = defaultdict(
        _empty
    )
    batter_profiles: dict[tuple[str, int, str, str, str, str], dict[str, int]] = (
        defaultdict(_empty)
    )
    bowler_profiles: dict[tuple[str, int, str, str], dict[str, int]] = defaultdict(
        _empty
    )
    rows: list[dict[str, Any]] = []

    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        registry = raw["info"].get("registry", {}).get("people", {})
        venue = str(raw["info"].get("venue") or raw["info"].get("city") or "unknown")
        track = _track_type(venues[venue], global_profile)
        match_events: list[tuple[int, str, str, str, dict[str, int]]] = []
        for innings_number, innings in enumerate(raw.get("innings", []), start=1):
            is_chase = int(innings_number == 2)
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                phase = _phase(over_number)
                striker = str(deliveries[0]["batter"])
                bowler = str(deliveries[0]["bowler"])
                bowler_type, bowler_arm = style_by_id.get(
                    str(registry.get(bowler, "")), ("unknown", "unknown")
                )
                context_key = (is_chase, phase, track, bowler_type, bowler_arm)
                prior = context_priors[context_key]
                prior_balls = prior["balls"]
                prior_rates = {
                    name: prior[name] / prior_balls if prior_balls else 0.0
                    for name in METRICS
                }
                batter = batter_profiles[
                    (striker, is_chase, phase, track, bowler_type, bowler_arm)
                ]
                bowling_role = is_chase
                bowler_profile = bowler_profiles[(bowler, bowling_role, phase, track)]
                row: dict[str, Any] = {
                    "source_file": path.name,
                    "match_date": match_date,
                    "innings": innings_number,
                    "over": over_number,
                    "player_innings_is_chase": is_chase,
                    "player_innings_bowler_type": bowler_type,
                    "player_innings_bowler_arm": bowler_arm,
                    "player_innings_batter_samples": batter["balls"],
                    "player_innings_bowler_samples": bowler_profile["balls"],
                }
                for name in METRICS:
                    row[f"player_innings_batter_{name}_rate"] = _rate(
                        batter, name, prior_rates[name], 72.0
                    )
                    row[f"player_innings_bowler_{name}_rate"] = _rate(
                        bowler_profile, name, prior_rates[name], 96.0
                    )
                rows.append(row)

                for delivery in deliveries:
                    if not _is_legal(delivery):
                        continue
                    batter_name = str(delivery["batter"])
                    bowler_name = str(delivery["bowler"])
                    batter_runs = int(delivery["runs"]["batter"])
                    total_runs = int(delivery["runs"]["total"])
                    event = {
                        "balls": 1,
                        "runs": total_runs,
                        "boundaries": int(batter_runs in {4, 6}),
                        "dots": int(total_runs == 0),
                        "wickets": int(_is_wicket(delivery)),
                    }
                    event_type, event_arm = style_by_id.get(
                        str(registry.get(bowler_name, "")), ("unknown", "unknown")
                    )
                    match_events.append(
                        (
                            is_chase,
                            phase,
                            batter_name,
                            bowler_name,
                            event_type,
                            event_arm,
                            event,
                        )
                    )

        # Freeze profiles for the full match to prevent target bleed.
        for (
            is_chase,
            phase,
            batter,
            bowler,
            bowler_type,
            bowler_arm,
            event,
        ) in match_events:
            bowling_role = is_chase
            for profile in (
                global_profile,
                venues[venue],
                context_priors[(is_chase, phase, track, bowler_type, bowler_arm)],
                batter_profiles[
                    (batter, is_chase, phase, track, bowler_type, bowler_arm)
                ],
                bowler_profiles[(bowler, bowling_role, phase, track)],
            ):
                for name, value in event.items():
                    profile[name] += value

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Player innings-phase dataset contains duplicate over keys.")
    return result.sort_values(
        ["match_date", "source_file", "innings", "over"]
    ).reset_index(drop=True)
