"""Leakage-safe live state features for Candidate v3.6."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ml.candidate_v3_dataset import _is_legal, _is_wicket
from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS


def _empty_batter() -> dict[str, int]:
    return {"balls": 0, "runs": 0, "dots": 0, "singles": 0, "boundaries": 0}


def _empty_bowler() -> dict[str, int]:
    return {"balls": 0, "runs": 0, "dots": 0, "boundaries": 0, "wickets": 0}


def _rate(
    values: dict[str, int], name: str, prior: float, strength: float = 8.0
) -> float:
    return (values[name] + strength * prior) / (values["balls"] + strength)


def expected_striker_balls(striker_odd: float, partner_odd: float) -> float:
    """Propagate strike probability over six legal deliveries without sampling."""

    probability_striker = 1.0
    expected = 0.0
    for _ in range(6):
        expected += probability_striker
        probability_striker = (
            probability_striker * (1.0 - striker_odd)
            + (1.0 - probability_striker) * partner_odd
        )
    return expected


def build_state_strike_share_dataset(project_root: Path) -> pd.DataFrame:
    paths: list[tuple[str, Path]] = []
    excluded = set(KNOWN_RULE_ANOMALY_EXCLUSIONS)
    for scope in ("t20i", "ipl"):
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json"):
            if path.stem in excluded:
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            paths.append((str(raw["info"]["dates"][0]), path))
    paths.sort(key=lambda item: (item[0], item[1].name))

    historical_batters: dict[str, dict[str, int]] = defaultdict(_empty_batter)
    global_batter = _empty_batter()
    rows: list[dict[str, Any]] = []
    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        for innings_number, innings in enumerate(raw.get("innings", []), start=1):
            batters: dict[str, dict[str, int]] = defaultdict(_empty_batter)
            bowlers: dict[str, dict[str, int]] = defaultdict(_empty_bowler)
            since_boundary: dict[str, int] = defaultdict(int)
            recent: deque[dict[str, int]] = deque(maxlen=12)
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                first = deliveries[0]
                striker = str(first["batter"])
                partner = str(first["non_striker"])
                bowler = str(first["bowler"])
                striker_state = batters[striker]
                bowler_state = bowlers[bowler]
                global_single = (
                    global_batter["singles"] / global_batter["balls"]
                    if global_batter["balls"]
                    else 0.25
                )
                striker_odd = _rate(
                    historical_batters[striker], "singles", global_single, 24.0
                )
                partner_odd = _rate(
                    historical_batters[partner], "singles", global_single, 24.0
                )
                recent_values = list(recent)
                recent_runs = np.array(
                    [event["runs"] for event in recent_values], dtype=float
                )
                row = {
                    "source_file": path.name,
                    "match_date": match_date,
                    "innings": innings_number,
                    "over": int(source_over["over"]) + 1,
                    "settlement_balls_faced": striker_state["balls"],
                    "settlement_runs": striker_state["runs"],
                    "settlement_runs_per_ball": striker_state["runs"]
                    / max(1, striker_state["balls"]),
                    "settlement_dot_rate": striker_state["dots"]
                    / max(1, striker_state["balls"]),
                    "settlement_rotation_rate": striker_state["singles"]
                    / max(1, striker_state["balls"]),
                    "settlement_boundary_rate": striker_state["boundaries"]
                    / max(1, striker_state["balls"]),
                    "settlement_balls_since_boundary": since_boundary[striker],
                    "strike_share_expected_striker_balls": expected_striker_balls(
                        striker_odd, partner_odd
                    ),
                    "strike_share_striker_odd_rate": striker_odd,
                    "strike_share_partner_odd_rate": partner_odd,
                    "bowler_match_balls": bowler_state["balls"],
                    "bowler_match_runs_per_ball": bowler_state["runs"]
                    / max(1, bowler_state["balls"]),
                    "bowler_match_dot_rate": bowler_state["dots"]
                    / max(1, bowler_state["balls"]),
                    "bowler_match_boundary_rate": bowler_state["boundaries"]
                    / max(1, bowler_state["balls"]),
                    "bowler_match_wickets": bowler_state["wickets"],
                    "bowler_match_fourth_over": int(bowler_state["balls"] >= 18),
                    "volatility_runs_std_last_6": float(np.std(recent_runs[-6:]))
                    if len(recent_runs)
                    else 0.0,
                    "volatility_runs_std_last_12": float(np.std(recent_runs))
                    if len(recent_runs)
                    else 0.0,
                    "volatility_boundary_dot_mix": sum(
                        event["boundary"] + event["dot"] for event in recent_values
                    )
                    / max(1, len(recent_values)),
                    "volatility_recent_wicket": int(
                        any(event["wicket"] for event in recent_values[-6:])
                    ),
                }
                rows.append(row)

                for delivery in deliveries:
                    if not _is_legal(delivery):
                        continue
                    batter = str(delivery["batter"])
                    delivery_bowler = str(delivery["bowler"])
                    batter_runs = int(delivery["runs"]["batter"])
                    total_runs = int(delivery["runs"]["total"])
                    wicket = int(_is_wicket(delivery))
                    event = {
                        "runs": total_runs,
                        "dot": int(total_runs == 0),
                        "boundary": int(batter_runs in {4, 6}),
                        "wicket": wicket,
                    }
                    batter_event = {
                        "balls": 1,
                        "runs": batter_runs,
                        "dots": event["dot"],
                        "singles": int(batter_runs == 1),
                        "boundaries": event["boundary"],
                    }
                    for profile in (
                        batters[batter],
                        historical_batters[batter],
                        global_batter,
                    ):
                        for name, value in batter_event.items():
                            profile[name] += value
                    bowler_event = {
                        "balls": 1,
                        "runs": total_runs,
                        "dots": event["dot"],
                        "boundaries": event["boundary"],
                        "wickets": wicket,
                    }
                    for name, value in bowler_event.items():
                        bowlers[delivery_bowler][name] += value
                    since_boundary[batter] = (
                        0 if event["boundary"] else since_boundary[batter] + 1
                    )
                    recent.append(event)

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("v3.6 state dataset contains duplicate over keys.")
    return result.sort_values(
        ["match_date", "source_file", "innings", "over"]
    ).reset_index(drop=True)
