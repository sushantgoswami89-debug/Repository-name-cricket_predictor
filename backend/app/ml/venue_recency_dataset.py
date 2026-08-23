"""Leakage-safe, recency-weighted venue scoring average.

Strategic follow-up (2026-08-23) to the player-level recency-form work
(`recency_weighted_prior_dataset.py`): `app/ml/ipl_venue_regime_dataset.py`'s
`venue_par_score` is a flat, all-time average of every first-innings total
ever played at a venue (`venue_avg = sum(prior) / len(prior)`, no decay).
Pitches genuinely evolve -- relaid tracks, groundskeeping changes, bat
technology and boundary-size trends shifting T20 scoring norms over years
-- so a venue's score from 2013 counts exactly as much as one from last
month in the existing feature. This is the same shape of gap already
fixed for player form and partnership state, applied to venue conditions
instead: a different category of signal, not another player-level
feature, and one that touches `VENUE_FEATURES` in *both* live models at
once (run-range and wicket both already consume `venue_par_score`/
`venue_scoring_regime`), unlike most single-model features tested this
session.

Innings-indexed EWMA (each innings played at a venue is "one tick," not
calendar time -- venues host relatively few innings per season, so a
per-match decay analogous to player recency would barely move within a
season). `VENUE_DECAY = 0.95` gives roughly a 13-14 innings half-life
(a little under one full IPL season's worth of matches at that venue),
chosen for direct comparability with `MATCH_DECAY` in
`recency_weighted_prior_dataset.py` -- not separately swept, per this
project's standing "don't sweep the same lever repeatedly" practice.

Added alongside the existing flat `venue_par_score` (not replacing it) so
the model can decide via feature importance whether recency-weighting
adds anything beyond the flat average and the existing shrinkage-toward-
global-average logic.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.ipl_venues import normalize_ipl_venue

VENUE_DECAY = 0.95
DEFAULT_PAR_SCORE = 170.0
VENUE_SHRINKAGE_INNINGS = 8.0


def _innings_total(innings: dict[str, Any]) -> int:
    return sum(
        int(delivery.get("runs", {}).get("total", 0))
        for over in innings.get("overs", [])
        for delivery in over.get("deliveries", [])
    )


def compute_final_venue_recency_state(
    project_root: Path, scopes: tuple[str, ...] = ("ipl", "t20i")
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Returns (venue_ewma, global_ewma) as of the most recent available
    match -- for live snapshot builders that only need the final state,
    not the full training frame. `venue_ewma[venue_name] = {"weight":
    ..., "value": ...}`; the caller derives par_score/prior_innings the
    same shrinkage way `build_venue_recency_dataset` does per row.
    """
    paths: list[tuple[str, Path]] = []
    for scope in scopes:
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json"):
            raw = json.loads(path.read_text(encoding="utf-8"))
            paths.append((str(raw["info"]["dates"][0]), path))
    paths.sort(key=lambda item: (item[0], item[1].name))

    venue_ewma: dict[str, dict[str, float]] = defaultdict(lambda: {"weight": 0.0, "value": 0.0})
    global_ewma = {"weight": 0.0, "value": 0.0}
    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        info = raw["info"]
        source_venue = str(info.get("venue") or info.get("city") or "unknown").strip()
        venue = normalize_ipl_venue(source_venue)
        state = venue_ewma[venue]
        regular_innings = [
            innings for innings in raw.get("innings", []) if not bool(innings.get("super_over", False))
        ]
        for innings in regular_innings:
            total = _innings_total(innings)
            state["weight"] = VENUE_DECAY * state["weight"] + 1.0
            state["value"] = VENUE_DECAY * state["value"] + total
            global_ewma["weight"] = VENUE_DECAY * global_ewma["weight"] + 1.0
            global_ewma["value"] = VENUE_DECAY * global_ewma["value"] + total
    return dict(venue_ewma), global_ewma


def build_venue_recency_dataset(
    project_root: Path, scopes: tuple[str, ...] = ("ipl", "t20i")
) -> pd.DataFrame:
    paths: list[tuple[str, Path]] = []
    for scope in scopes:
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json"):
            raw = json.loads(path.read_text(encoding="utf-8"))
            paths.append((str(raw["info"]["dates"][0]), path))
    paths.sort(key=lambda item: (item[0], item[1].name))

    venue_ewma: dict[str, dict[str, float]] = defaultdict(lambda: {"weight": 0.0, "value": 0.0})
    global_ewma = {"weight": 0.0, "value": 0.0}
    rows: list[dict[str, Any]] = []

    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        info = raw["info"]
        source_venue = str(info.get("venue") or info.get("city") or "unknown").strip()
        venue = normalize_ipl_venue(source_venue)
        state = venue_ewma[venue]
        prior_innings = state["weight"]
        global_avg = (
            global_ewma["value"] / global_ewma["weight"] if global_ewma["weight"] > 0 else DEFAULT_PAR_SCORE
        )
        if prior_innings > 0:
            venue_avg = state["value"] / state["weight"]
            par_score = (
                prior_innings * venue_avg + VENUE_SHRINKAGE_INNINGS * global_avg
            ) / (prior_innings + VENUE_SHRINKAGE_INNINGS)
        else:
            par_score = global_avg

        regular_innings = [
            innings for innings in raw.get("innings", []) if not bool(innings.get("super_over", False))
        ]
        for innings_number, innings in enumerate(regular_innings, start=1):
            for over in innings.get("overs", []):
                if not over.get("deliveries"):
                    continue
                rows.append(
                    {
                        "source_file": path.name,
                        "innings": innings_number,
                        "over": int(over["over"]) + 1,
                        "venue_recency_par_score": par_score,
                        "venue_recency_prior_innings": prior_innings,
                    }
                )
            total = _innings_total(innings)
            state["weight"] = VENUE_DECAY * state["weight"] + 1.0
            state["value"] = VENUE_DECAY * state["value"] + total
            global_ewma["weight"] = VENUE_DECAY * global_ewma["weight"] + 1.0
            global_ewma["value"] = VENUE_DECAY * global_ewma["value"] + total

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Venue recency dataset contains duplicate over keys.")
    return result
