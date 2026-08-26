"""Per-player phase-scoring PROFILE (2026-08-26, following external
research on batter/bowler role clustering -- e.g. IPL batting-archetype
studies using K-means on phase/style stat profiles). Genuinely different
from the existing `batter_phase_*`/`bowl_phase_avg_runs` features, which
only expose the CURRENT over's phase rate: this exposes all three phases
(powerplay/middle/death) SIMULTANEOUSLY per row, so a player's shape
across the innings (e.g. "anchor": flat across phases, vs "finisher":
low powerplay + high death) can be clustered into a role archetype.

Career-to-date, chronological, leakage-safe (updated only between
matches). Batter identity via `canonical_player_id`; bowler identity the
same. Rates are shrunk toward the GLOBAL phase average with a pseudo-count
(not the player's own overall rate -- deliberately, since the whole point
is to expose the SHAPE across phases, and shrinking toward a global
per-phase baseline preserves that shape for sparse-history players rather
than flattening it toward their own single overall number).

Clustering itself (fitting K-means, assigning archetype labels) happens
in the training script, not here -- this module only builds the raw
3-phase profile vectors, the same separation of concerns as every other
dataset builder in this codebase (temperature scaling, banding, etc. all
live in the training script too).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.candidate_v3_dataset import _is_legal, _phase
from app.ml.ipl_identities import canonical_player_id
from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS

SHRINKAGE_BALLS = 90.0
PHASES = ("powerplay", "middle", "death")


def _empty() -> dict[str, int]:
    return {"balls": 0, "runs": 0}


def build_player_archetype_dataset(
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

    global_phase: dict[str, dict[str, int]] = {p: _empty() for p in PHASES}
    batter_phase: dict[tuple[str, str], dict[str, int]] = {}
    bowler_phase: dict[tuple[str, str], dict[str, int]] = {}

    rows: list[dict[str, Any]] = []

    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        registry = raw.get("info", {}).get("registry", {}).get("people", {})
        match_batter_events: list[tuple[str, str, int, int]] = []  # (batter, phase, balls, runs)
        match_bowler_events: list[tuple[str, str, int, int]] = []  # (bowler, phase, balls, runs)

        for innings_number, innings in enumerate(raw.get("innings", []), start=1):
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                phase = _phase(over_number)
                striker = canonical_player_id(str(deliveries[0].get("batter", "")), registry)
                bowler = canonical_player_id(str(deliveries[0].get("bowler", "")), registry)

                row: dict[str, Any] = {
                    "source_file": path.name, "match_date": match_date,
                    "innings": innings_number, "over": over_number,
                }
                for p in PHASES:
                    gp = global_phase[p]
                    global_rate = (gp["runs"] / gp["balls"]) if gp["balls"] else 0.0
                    bp = batter_phase.get((striker, p), _empty())
                    weight_b = bp["balls"] / (bp["balls"] + SHRINKAGE_BALLS)
                    bp_rate = (bp["runs"] / bp["balls"]) if bp["balls"] else global_rate
                    row[f"striker_{p}_rate_shrunk"] = weight_b * bp_rate + (1 - weight_b) * global_rate

                    wp = bowler_phase.get((bowler, p), _empty())
                    weight_w = wp["balls"] / (wp["balls"] + SHRINKAGE_BALLS)
                    wp_rate = (wp["runs"] / wp["balls"]) if wp["balls"] else global_rate
                    row[f"bowler_{p}_economy_shrunk"] = 6.0 * (weight_w * wp_rate + (1 - weight_w) * global_rate)
                row["striker_archetype_balls"] = batter_phase.get((striker, phase), _empty())["balls"]
                row["bowler_archetype_balls"] = bowler_phase.get((bowler, phase), _empty())["balls"]
                rows.append(row)

                striker_legal_balls = sum(1 for d in deliveries if _is_legal(d))
                striker_legal_runs = sum(
                    int(d.get("runs", {}).get("batter", 0)) for d in deliveries if _is_legal(d)
                )
                total_runs = sum(int(d.get("runs", {}).get("total", 0)) for d in deliveries)
                legal_balls = sum(1 for d in deliveries if _is_legal(d))
                if striker_legal_balls > 0:
                    match_batter_events.append((striker, phase, striker_legal_balls, striker_legal_runs))
                if legal_balls > 0:
                    match_bowler_events.append((bowler, phase, legal_balls, total_runs))

        for player, phase, balls, runs in match_batter_events:
            profile = batter_phase.setdefault((player, phase), _empty())
            profile["balls"] += balls
            profile["runs"] += runs
            gp = global_phase[phase]
            gp["balls"] += balls
            gp["runs"] += runs
        for player, phase, balls, runs in match_bowler_events:
            profile = bowler_phase.setdefault((player, phase), _empty())
            profile["balls"] += balls
            profile["runs"] += runs

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Player archetype dataset contains duplicate over keys.")
    return result
