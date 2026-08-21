"""Leakage-safe batter response profiles segmented by innings entry phase."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.candidate_v3_dataset import _is_legal, _is_wicket, _phase
from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS

OUTCOMES = ("boundary", "single", "wicket", "dot")


def _empty_profile() -> dict[str, int]:
    return {"samples": 0, "runs": 0, **{name: 0 for name in OUTCOMES}}


def _smoothed_rates(
    player: dict[str, int], prior: dict[str, int], strength: float = 24.0
) -> dict[str, float | int]:
    samples = player["samples"]
    prior_samples = prior["samples"]
    result: dict[str, float | int] = {"samples": samples}
    for name in ("runs", *OUTCOMES):
        global_rate = prior[name] / prior_samples if prior_samples else 0.0
        result[f"{name}_rate"] = (player[name] + strength * global_rate) / (
            samples + strength
        )
    return result


def build_batter_entry_phase_dataset(project_root: Path) -> pd.DataFrame:
    """Return pre-over batter profiles split by the phase of innings entry."""

    paths: list[tuple[str, Path]] = []
    excluded = set(KNOWN_RULE_ANOMALY_EXCLUSIONS)
    for scope in ("t20i", "ipl"):
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json"):
            if path.stem in excluded:
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            paths.append((str(raw["info"]["dates"][0]), path))
    paths.sort(key=lambda item: (item[0], item[1].name))

    profiles: dict[tuple[str, str], dict[str, int]] = defaultdict(_empty_profile)
    phase_priors: dict[str, dict[str, int]] = defaultdict(_empty_profile)
    rows: list[dict[str, Any]] = []

    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        for innings_number, innings in enumerate(raw.get("innings", []), start=1):
            entry_phase: dict[str, str] = {}
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                current_phase = _phase(over_number)
                first = deliveries[0]
                for name in (str(first["batter"]), str(first["non_striker"])):
                    entry_phase.setdefault(name, current_phase)
                striker = str(first["batter"])
                striker_entry_phase = entry_phase[striker]
                values = _smoothed_rates(
                    profiles[(striker, striker_entry_phase)],
                    phase_priors[striker_entry_phase],
                )
                row: dict[str, Any] = {
                    "source_file": path.name,
                    "match_date": match_date,
                    "innings": innings_number,
                    "over": over_number,
                    "batter_entry_phase": striker_entry_phase,
                }
                for name, value in values.items():
                    row[f"entry_phase_batter_{name}"] = value
                rows.append(row)

                for delivery in deliveries:
                    phase = _phase(over_number)
                    batter = str(delivery["batter"])
                    non_striker = str(delivery["non_striker"])
                    entry_phase.setdefault(batter, phase)
                    entry_phase.setdefault(non_striker, phase)
                    if not _is_legal(delivery):
                        continue
                    batter_runs = int(delivery["runs"]["batter"])
                    total_runs = int(delivery["runs"]["total"])
                    event = {
                        "runs": batter_runs,
                        "boundary": int(batter_runs in {4, 6}),
                        "single": int(batter_runs == 1),
                        "wicket": int(_is_wicket(delivery)),
                        "dot": int(total_runs == 0),
                    }
                    for profile in (
                        profiles[(batter, entry_phase[batter])],
                        phase_priors[entry_phase[batter]],
                    ):
                        profile["samples"] += 1
                        for name, value in event.items():
                            profile[name] += value

    result = pd.DataFrame(rows).sort_values(
        ["match_date", "source_file", "innings", "over"]
    )
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Batter entry-phase dataset contains duplicate over keys.")
    return result.reset_index(drop=True)
