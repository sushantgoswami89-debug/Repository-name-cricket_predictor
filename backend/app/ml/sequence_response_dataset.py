"""Leakage-safe delivery-sequence and player-response features for v3.3."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.candidate_v3_dataset import _is_legal, _is_wicket
from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS

OUTCOMES = ("boundary", "single", "wicket", "dot")


def _bucket(streak: int) -> int:
    return min(streak, 3)


def _rates(
    profiles: dict[tuple[str, int], dict[str, int]],
    global_profiles: dict[int, dict[str, int]],
    player: str,
    streak: int,
) -> dict[str, float | int]:
    bucket = _bucket(streak)
    player_values = profiles[(player, bucket)]
    global_values = global_profiles[bucket]
    sample = player_values["samples"]
    global_sample = global_values["samples"]
    result: dict[str, float | int] = {"samples": sample}
    for outcome in OUTCOMES:
        prior = global_values[outcome] / global_sample if global_sample else 0.0
        result[f"{outcome}_rate"] = (player_values[outcome] + 20.0 * prior) / (
            sample + 20.0
        )
    return result


def _empty_profile() -> dict[str, int]:
    return {"samples": 0, **{outcome: 0 for outcome in OUTCOMES}}


def _matchup_rates(
    profile: dict[str, int],
    batter_state: dict[str, float | int],
    bowler_state: dict[str, float | int],
    prior_strength: float,
) -> dict[str, float | int]:
    sample = profile["samples"]
    result: dict[str, float | int] = {"samples": sample}
    for outcome in OUTCOMES:
        prior = (
            float(batter_state[f"{outcome}_rate"])
            + float(bowler_state[f"{outcome}_rate"])
        ) / 2.0
        result[f"{outcome}_rate"] = (profile[outcome] + prior_strength * prior) / (
            sample + prior_strength
        )
    return result


def build_sequence_response_dataset(project_root: Path) -> pd.DataFrame:
    """Return one enriched row per verified target over in chronological order."""

    paths: list[tuple[str, Path]] = []
    excluded = set(KNOWN_RULE_ANOMALY_EXCLUSIONS)
    for path in (project_root / "data/raw/cricsheet/t20i").glob("*.json"):
        if path.stem in excluded:
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        paths.append((str(raw["info"]["dates"][0]), path))
    for path in (project_root / "data/raw/cricsheet/ipl").glob("*.json"):
        if path.stem in excluded:
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        paths.append((str(raw["info"]["dates"][0]), path))
    paths.sort(key=lambda item: (item[0], item[1].name))

    batter_profiles: dict[tuple[str, int], dict[str, int]] = defaultdict(_empty_profile)
    bowler_profiles: dict[tuple[str, int], dict[str, int]] = defaultdict(_empty_profile)
    global_batter: dict[int, dict[str, int]] = defaultdict(_empty_profile)
    global_bowler: dict[int, dict[str, int]] = defaultdict(_empty_profile)
    matchup_profiles: dict[tuple[str, str], dict[str, int]] = defaultdict(
        _empty_profile
    )
    pressure_matchups: dict[tuple[str, str, int], dict[str, int]] = defaultdict(
        _empty_profile
    )
    rows: list[dict[str, Any]] = []

    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        for innings_number, innings in enumerate(raw.get("innings", []), start=1):
            recent: deque[dict[str, int]] = deque(maxlen=18)
            batter_dots: dict[str, int] = defaultdict(int)
            bowler_dots: dict[str, int] = defaultdict(int)
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                first = deliveries[0]
                batter = str(first["batter"])
                bowler = str(first["bowler"])
                batter_state = _rates(
                    batter_profiles,
                    global_batter,
                    batter,
                    batter_dots[batter],
                )
                bowler_state = _rates(
                    bowler_profiles,
                    global_bowler,
                    bowler,
                    bowler_dots[bowler],
                )
                matchup_state = _matchup_rates(
                    matchup_profiles[(batter, bowler)],
                    batter_state,
                    bowler_state,
                    30.0,
                )
                pressure_bucket = max(
                    _bucket(batter_dots[batter]), _bucket(bowler_dots[bowler])
                )
                pressure_matchup_state = _matchup_rates(
                    pressure_matchups[(batter, bowler, pressure_bucket)],
                    batter_state,
                    bowler_state,
                    40.0,
                )
                recent_values = list(recent)
                trailing_dots = 0
                for event in reversed(recent_values):
                    if not event["dot"]:
                        break
                    trailing_dots += 1
                row: dict[str, Any] = {
                    "source_file": path.name,
                    "match_date": match_date,
                    "innings": innings_number,
                    "over": int(source_over["over"]) + 1,
                    "striker": batter,
                    "bowler": bowler,
                    "trailing_dot_streak": trailing_dots,
                    "striker_dot_streak": batter_dots[batter],
                    "bowler_dot_streak": bowler_dots[bowler],
                    "last_ball_runs": recent_values[-1]["runs"] if recent_values else 0,
                }
                for window in (3, 6, 12, 18):
                    events = recent_values[-window:]
                    denominator = max(1, len(events))
                    row[f"runs_last_{window}_balls"] = sum(
                        event["runs"] for event in events
                    )
                    for outcome in OUTCOMES:
                        row[f"{outcome}_rate_last_{window}"] = (
                            sum(event[outcome] for event in events) / denominator
                        )
                for prefix, values in (
                    ("batter_response", batter_state),
                    ("bowler_response", bowler_state),
                    ("matchup_response", matchup_state),
                    ("pressure_matchup_response", pressure_matchup_state),
                ):
                    for name, value in values.items():
                        row[f"{prefix}_{name}"] = value
                rows.append(row)

                for delivery in deliveries:
                    if not _is_legal(delivery):
                        continue
                    batter = str(delivery["batter"])
                    bowler = str(delivery["bowler"])
                    total = int(delivery["runs"]["total"])
                    batter_runs = int(delivery["runs"]["batter"])
                    event = {
                        "runs": total,
                        "boundary": int(batter_runs in {4, 6}),
                        "single": int(total == 1),
                        "wicket": int(_is_wicket(delivery)),
                        "dot": int(total == 0),
                    }
                    batter_key = (batter, _bucket(batter_dots[batter]))
                    bowler_key = (bowler, _bucket(bowler_dots[bowler]))
                    matchup_key = (batter, bowler)
                    pressure_key = (
                        batter,
                        bowler,
                        max(batter_key[1], bowler_key[1]),
                    )
                    for profile in (
                        batter_profiles[batter_key],
                        global_batter[batter_key[1]],
                        bowler_profiles[bowler_key],
                        global_bowler[bowler_key[1]],
                        matchup_profiles[matchup_key],
                        pressure_matchups[pressure_key],
                    ):
                        profile["samples"] += 1
                        for outcome in OUTCOMES:
                            profile[outcome] += event[outcome]
                    batter_dots[batter] = batter_dots[batter] + 1 if event["dot"] else 0
                    bowler_dots[bowler] = bowler_dots[bowler] + 1 if event["dot"] else 0
                    recent.append(event)

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Sequence response dataset contains duplicate over keys.")
    return result.sort_values(
        ["match_date", "source_file", "innings", "over"]
    ).reset_index(drop=True)
