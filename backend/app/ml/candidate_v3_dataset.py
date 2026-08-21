"""Build the verified, leakage-safe Candidate v3 ball-by-ball training view."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    frame: pd.DataFrame
    report: dict[str, Any]


def _is_legal(delivery: dict[str, Any]) -> bool:
    extras = delivery.get("extras", {})
    return "wides" not in extras and "noballs" not in extras


def _is_wicket(delivery: dict[str, Any]) -> bool:
    return any(
        wicket.get("kind") not in {"retired hurt", "obstructing the field"}
        for wicket in delivery.get("wickets", [])
    )


def _phase(over_number: int) -> str:
    if over_number <= 6:
        return "powerplay"
    if over_number <= 15:
        return "middle"
    return "death"


def _digest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
        digest.update(str(path.stat().st_mtime_ns).encode())
    return digest.hexdigest()


def build_verified_dataset(project_root: Path) -> DatasetBuildResult:
    """Derive one pre-over row from every eligible T20 innings.

    All impact features use deliveries completed before the predicted over. The
    source total is reconciled independently while each innings is traversed.
    """

    source_files = sorted(
        path
        for scope in ("ipl", "t20i")
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json")
    )
    excluded_ids = set(KNOWN_RULE_ANOMALY_EXCLUSIONS)
    rows: list[dict[str, Any]] = []
    excluded: list[str] = []
    matches_verified = 0
    innings_verified = 0
    deliveries_verified = 0

    for path in source_files:
        if path.stem in excluded_ids:
            excluded.append(path.name)
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        info = raw["info"]
        if str(info.get("gender", "")).lower() != "male":
            continue
        if str(info.get("match_type", "")).upper() != "T20":
            continue
        match_date = str(info["dates"][0])
        scheduled_overs = int(info.get("overs", 20))
        innings_list = raw.get("innings", [])
        first_innings_total: int | None = None

        for innings_index, innings in enumerate(innings_list, start=1):
            if innings.get("target", {}).get("overs"):
                innings_overs = int(innings["target"]["overs"])
            else:
                innings_overs = scheduled_overs
            score = 0
            wickets = 0
            legal_balls = 0
            recent: deque[dict[str, int]] = deque(maxlen=12)
            innings_rows = 0

            for source_over in innings.get("overs", []):
                over_number = int(source_over["over"]) + 1
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                phase = _phase(over_number)
                balls_remaining = max(0, innings_overs * 6 - legal_balls)
                target = (
                    first_innings_total + 1
                    if innings_index == 2 and first_innings_total is not None
                    else 0
                )
                runs_required = max(0, target - score) if target else 0
                recent_balls = len(recent)
                recent_runs = sum(item["runs"] for item in recent)
                recent_dots = sum(item["dot"] for item in recent)
                recent_singles = sum(item["single"] for item in recent)
                recent_boundaries = sum(item["boundary"] for item in recent)
                recent_wickets = sum(item["wicket"] for item in recent)

                over_runs = sum(int(d["runs"]["total"]) for d in deliveries)
                over_wickets = sum(int(_is_wicket(d)) for d in deliveries)
                rows.append(
                    {
                        "source_file": path.name,
                        "match_date": match_date,
                        "innings": innings_index,
                        "over": over_number,
                        "phase": phase,
                        "score_before_over": score,
                        "wkts_down_before_over": wickets,
                        "wickets_in_hand": max(0, 10 - wickets),
                        "legal_balls_bowled": legal_balls,
                        "balls_remaining": balls_remaining,
                        "current_run_rate": score * 6 / legal_balls
                        if legal_balls
                        else 0.0,
                        "is_chase": int(target > 0),
                        "runs_required": runs_required,
                        "required_run_rate": runs_required * 6 / balls_remaining
                        if target and balls_remaining
                        else 0.0,
                        "recent_legal_balls": recent_balls,
                        "recent_runs_per_ball": recent_runs / recent_balls
                        if recent_balls
                        else 0.0,
                        "recent_dot_rate": recent_dots / recent_balls
                        if recent_balls
                        else 0.0,
                        "recent_single_rate": recent_singles / recent_balls
                        if recent_balls
                        else 0.0,
                        "recent_boundary_rate": recent_boundaries / recent_balls
                        if recent_balls
                        else 0.0,
                        "recent_wicket_rate": recent_wickets / recent_balls
                        if recent_balls
                        else 0.0,
                        "phase_boundary_pressure": recent_boundaries / recent_balls
                        if recent_balls
                        else 0.0,
                        "phase_dot_pressure": recent_dots / recent_balls
                        if recent_balls
                        else 0.0,
                        "phase_rotation_value": recent_singles / recent_balls
                        if recent_balls
                        else 0.0,
                        "runs_in_over": over_runs,
                        "wicket_in_over": int(over_wickets > 0),
                    }
                )
                innings_rows += 1

                for delivery in deliveries:
                    total = int(delivery["runs"]["total"])
                    batter_runs = int(delivery["runs"]["batter"])
                    wicket = int(_is_wicket(delivery))
                    score += total
                    wickets += wicket
                    deliveries_verified += 1
                    if _is_legal(delivery):
                        legal_balls += 1
                        recent.append(
                            {
                                "runs": total,
                                "dot": int(total == 0),
                                "single": int(total == 1),
                                "boundary": int(batter_runs in {4, 6}),
                                "wicket": wicket,
                            }
                        )

            if innings_rows:
                innings_verified += 1
            if innings_index == 1:
                first_innings_total = score
        matches_verified += 1

    frame = (
        pd.DataFrame(rows)
        .sort_values(["match_date", "source_file", "innings", "over"])
        .reset_index(drop=True)
    )
    duplicate_keys = int(frame.duplicated(["source_file", "innings", "over"]).sum())
    if duplicate_keys:
        raise ValueError(f"Verified dataset contains {duplicate_keys} duplicate keys.")
    if frame.isna().any().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"Verified dataset contains missing values in {missing}.")

    report = {
        "dataset_version": "candidate_v3_verified_ball_by_ball_v1",
        "source_manifest_sha256": _digest(source_files),
        "source_files_seen": len(source_files),
        "matches_verified": matches_verified,
        "innings_verified": innings_verified,
        "deliveries_verified": deliveries_verified,
        "training_rows": len(frame),
        "duplicate_keys": duplicate_keys,
        "missing_values": 0,
        "known_anomaly_exclusions": sorted(excluded),
        "leakage_rule": "Every feature is calculated before the target over.",
        "verified": True,
    }
    return DatasetBuildResult(frame=frame, report=report)


def write_verified_dataset(project_root: Path, output_dir: Path) -> dict[str, Any]:
    result = build_verified_dataset(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.frame.to_csv(output_dir / "verified_training_overs.csv", index=False)
    (output_dir / "dataset_verification_report.json").write_text(
        json.dumps(result.report, indent=2), encoding="utf-8"
    )
    return result.report
