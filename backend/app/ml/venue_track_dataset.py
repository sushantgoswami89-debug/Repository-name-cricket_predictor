"""Historical venue and similar-track features using pre-match information only."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.candidate_v3_dataset import _is_legal, _is_wicket, _phase
from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS

METRICS = ("runs", "boundaries", "dots", "wickets")


def _empty() -> dict[str, int]:
    return {"balls": 0, **{name: 0 for name in METRICS}}


def _rate(profile: dict[str, int], name: str, prior: float, strength: float) -> float:
    return (profile[name] + strength * prior) / (profile["balls"] + strength)


def _track_type(venue: dict[str, int], global_profile: dict[str, int]) -> str:
    """Classify the current historical venue without looking at this match."""

    if venue["balls"] < 120 or not global_profile["balls"]:
        return "unknown"
    venue_boundary = venue["boundaries"] / venue["balls"]
    global_boundary = global_profile["boundaries"] / global_profile["balls"]
    venue_runs = venue["runs"] / venue["balls"]
    global_runs = global_profile["runs"] / global_profile["balls"]
    if venue_boundary > global_boundary * 1.08 and venue_runs > global_runs * 1.05:
        return "high_scoring"
    if venue_boundary < global_boundary * 0.92 and venue_runs < global_runs * 0.95:
        return "slow"
    return "balanced"


def build_venue_track_dataset(project_root: Path) -> pd.DataFrame:
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
    global_phase: dict[str, dict[str, int]] = defaultdict(_empty)
    venues: dict[str, dict[str, int]] = defaultdict(_empty)
    venue_phases: dict[tuple[str, str], dict[str, int]] = defaultdict(_empty)
    track_profiles: dict[tuple[str, str], dict[str, int]] = defaultdict(_empty)
    batter_tracks: dict[tuple[str, str, str], dict[str, int]] = defaultdict(_empty)
    bowler_tracks: dict[tuple[str, str, str], dict[str, int]] = defaultdict(_empty)
    rows: list[dict[str, Any]] = []

    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        venue = str(raw["info"].get("venue") or raw["info"].get("city") or "unknown")
        track = _track_type(venues[venue], global_profile)
        match_events: list[tuple[str, str, str, dict[str, int]]] = []
        for innings_number, innings in enumerate(raw.get("innings", []), start=1):
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                phase = _phase(over_number)
                striker = str(deliveries[0]["batter"])
                bowler = str(deliveries[0]["bowler"])
                phase_prior = global_phase[phase]
                phase_balls = phase_prior["balls"]
                priors = {
                    name: phase_prior[name] / phase_balls if phase_balls else 0.0
                    for name in METRICS
                }
                venue_phase = venue_phases[(venue, phase)]
                track_prior = track_profiles[(track, phase)]
                track_balls = track_prior["balls"]
                track_rates = {
                    name: track_prior[name] / track_balls
                    if track_balls
                    else priors[name]
                    for name in METRICS
                }
                batter_profile = batter_tracks[(striker, track, phase)]
                bowler_profile = bowler_tracks[(bowler, track, phase)]
                row: dict[str, Any] = {
                    "source_file": path.name,
                    "match_date": match_date,
                    "innings": innings_number,
                    "over": over_number,
                    "venue_track_type": track,
                    "venue_track_venue_samples": venue_phase["balls"],
                    "venue_track_batter_samples": batter_profile["balls"],
                    "venue_track_bowler_samples": bowler_profile["balls"],
                }
                for name in METRICS:
                    row[f"venue_track_venue_{name}_rate"] = _rate(
                        venue_phase, name, priors[name], 180.0
                    )
                    row[f"venue_track_batter_{name}_rate"] = _rate(
                        batter_profile, name, track_rates[name], 60.0
                    )
                    row[f"venue_track_bowler_{name}_rate"] = _rate(
                        bowler_profile, name, track_rates[name], 90.0
                    )
                rows.append(row)

                for delivery in deliveries:
                    if not _is_legal(delivery):
                        continue
                    batter = str(delivery["batter"])
                    delivery_bowler = str(delivery["bowler"])
                    batter_runs = int(delivery["runs"]["batter"])
                    total_runs = int(delivery["runs"]["total"])
                    event = {
                        "balls": 1,
                        "runs": total_runs,
                        "boundaries": int(batter_runs in {4, 6}),
                        "dots": int(total_runs == 0),
                        "wickets": int(_is_wicket(delivery)),
                    }
                    match_events.append((phase, batter, delivery_bowler, event))

        # Freeze venue and player-track history until the match is complete.
        for phase, batter, bowler, event in match_events:
            for profile in (
                global_profile,
                global_phase[phase],
                venues[venue],
                venue_phases[(venue, phase)],
                track_profiles[(track, phase)],
                batter_tracks[(batter, track, phase)],
                bowler_tracks[(bowler, track, phase)],
            ):
                for name, value in event.items():
                    profile[name] += value

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Venue-track dataset contains duplicate over keys.")
    return result.sort_values(
        ["match_date", "source_file", "innings", "over"]
    ).reset_index(drop=True)
