"""Verified over-mechanics targets for the Candidate v3.4 simulator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.candidate_v3_dataset import _is_legal, _is_wicket
from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS


def delivery_components(delivery: dict[str, Any]) -> dict[str, int]:
    """Classify one delivery into non-overlapping score components."""

    total = int(delivery["runs"]["total"])
    batter_runs = int(delivery["runs"]["batter"])
    extras_runs = int(delivery["runs"]["extras"])
    legal = int(_is_legal(delivery))
    return {
        "legal_balls": legal,
        "illegal_deliveries": 1 - legal,
        "dot_balls": int(legal and total == 0),
        "single_balls": int(legal and total == 1),
        "two_three_run_balls": int(legal and total in {2, 3}),
        "fours": int(batter_runs == 4),
        "sixes": int(batter_runs == 6),
        "boundary_runs": batter_runs if batter_runs in {4, 6} else 0,
        "running_batter_runs": (batter_runs if batter_runs not in {0, 4, 6} else 0),
        "extras_runs": extras_runs,
        "wickets": int(_is_wicket(delivery)),
        "total_runs": total,
    }


def build_over_component_dataset(project_root: Path) -> pd.DataFrame:
    excluded = set(KNOWN_RULE_ANOMALY_EXCLUSIONS)
    rows: list[dict[str, Any]] = []
    for scope in ("ipl", "t20i"):
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json"):
            if path.stem in excluded:
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            match_date = str(raw["info"]["dates"][0])
            for innings_number, innings in enumerate(raw.get("innings", []), start=1):
                for over in innings.get("overs", []):
                    deliveries = over.get("deliveries", [])
                    if not deliveries:
                        continue
                    totals = {
                        name: 0 for name in delivery_components(deliveries[0]).keys()
                    }
                    for delivery in deliveries:
                        for name, value in delivery_components(delivery).items():
                            totals[name] += value
                    rows.append(
                        {
                            "source_file": path.name,
                            "match_date": match_date,
                            "innings": innings_number,
                            "over": int(over["over"]) + 1,
                            "confirmed_striker": str(deliveries[0]["batter"]),
                            "confirmed_bowler": str(deliveries[0]["bowler"]),
                            **totals,
                        }
                    )
    frame = pd.DataFrame(rows)
    keys = ["source_file", "innings", "over"]
    if frame.duplicated(keys).any():
        raise ValueError("Over-component dataset contains duplicate keys.")
    if not (
        frame["total_runs"]
        == frame["boundary_runs"] + frame["running_batter_runs"] + frame["extras_runs"]
    ).all():
        raise ValueError("Over score does not reconcile with its components.")
    return frame.sort_values(
        ["match_date", "source_file", "innings", "over"]
    ).reset_index(drop=True)
