"""Leakage-safe batter/bowler venue x phase profiles.

Neither the batter nor the bowler side of this codebase currently has a
per-player venue split (see `ipl_phase_moe_dataset.py` for career x phase,
`ipl_venue_regime_dataset.py` for venue-level par score pooled across all
players). This fills that specific gap: "how does this player perform at
this venue in this phase" -- but a raw per-(player, venue, phase) rate
would mostly be noise. A quick sparsity check on the existing bowler-phase
(2-way, no venue) cells found a median of 42 balls per cell across 8,618
cells; adding a third dimension (315 venues) divides that further, so most
3-way cells will be single-digit balls or empty.

Each cell is therefore shrunk toward the player's own venue-agnostic
(player, phase) rate (already validated elsewhere this session), weighted
by how many balls the venue-specific cell has seen:
`weight = cell_balls / (cell_balls + SHRINKAGE_BALLS)`. A cell with zero
venue-specific history collapses to the plain phase rate; a cell with a
lot of venue-specific history dominates it. Same style of shrinkage
`ipl_venue_regime_dataset.py` already uses for venue par scores
(`VENUE_SHRINKAGE_INNINGS`), applied one level down to the player.

Chronological, leakage-safe: profiles are updated only after every row in
a match has been emitted (mirrors `build_ipl_phase_moe_features`'s
match_batter_events/match_bowler_events deferred-update pattern), so a
match never sees its own outcomes in its own features.

The bowler side is keyed off the over's first-delivery bowler the same
way `build_ipl_phase_moe_features` reports `known_bowler` -- future
information at publish time unless the caller has independently resolved
the upcoming bowler (the same requirement `wicket_contract22_features.py`
already has). Callers without a resolved bowler should zero these columns
out, same convention as `contract22_wicket_v2_batter_state.py`'s
`bowler_known=False` path.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.candidate_v3_dataset import _is_legal, _is_wicket, _phase
from app.ml.ipl_identities import canonical_player_id
from app.ml.ipl_venues import normalize_ipl_venue

UNKNOWN_CATEGORY = "__UNKNOWN__"
KEYS = ["source_file", "match_date", "innings", "over"]
SHRINKAGE_BALLS = 24  # 4 overs -- see module docstring for why
# 1st vs 2nd innings roughly halves phase-cell sample size (not the ~315x
# venue does), so a smaller shrinkage window is enough -- see
# train_run_range_v5_innings_phase.py's docstring for the sparsity check
# that motivated this value.
SHRINKAGE_BALLS_INNINGS = 12


def _empty_batter() -> dict[str, int]:
    return {"balls": 0, "runs": 0, "dots": 0, "boundaries": 0, "dismissals": 0}


def _empty_bowler() -> dict[str, int]:
    return {"balls": 0, "runs": 0, "dots": 0, "boundaries": 0, "wickets": 0}


def _rate(profile: dict[str, int], event: str) -> float:
    return float(profile[event]) / max(1, int(profile["balls"]))


def _shrunk_rate(
    cell: dict[str, int], fallback: dict[str, int], event: str, shrinkage: int = SHRINKAGE_BALLS
) -> float:
    cell_balls = cell["balls"]
    weight = cell_balls / (cell_balls + shrinkage)
    return weight * _rate(cell, event) + (1 - weight) * _rate(fallback, event)


def build_player_venue_phase_features(
    project_root: Path,
    *,
    scopes: tuple[str, ...] = ("ipl",),
) -> pd.DataFrame:
    """Return one pre-over venue x phase profile row per regular T20 over."""

    paths: list[tuple[str, Path]] = []
    for scope in scopes:
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json"):
            raw = json.loads(path.read_text(encoding="utf-8"))
            paths.append((str(raw["info"]["dates"][0]), path))
    paths.sort(key=lambda item: (item[0], item[1].name))

    batter_phase_history: dict[str, dict[str, int]] = defaultdict(_empty_batter)
    bowler_phase_history: dict[str, dict[str, int]] = defaultdict(_empty_bowler)
    batter_venue_phase_history: dict[str, dict[str, int]] = defaultdict(_empty_batter)
    bowler_venue_phase_history: dict[str, dict[str, int]] = defaultdict(_empty_bowler)
    # 1st innings (setting) vs 2nd innings (chasing) -- same phase, same
    # over range, but a batter/bowler's real numbers can differ a lot
    # between the two (e.g. powerplay approach differs when building a
    # total vs. already carrying chase pressure). Shrunk toward the
    # innings-agnostic phase rate above, not toward venue.
    batter_phase_innings_history: dict[str, dict[str, int]] = defaultdict(_empty_batter)
    bowler_phase_innings_history: dict[str, dict[str, int]] = defaultdict(_empty_bowler)

    rows: list[dict[str, Any]] = []

    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        registry = raw["info"].get("registry", {}).get("people", {})

        def player_key(name: str) -> str:
            return canonical_player_id(name, registry)

        venue = normalize_ipl_venue(
            str(raw["info"].get("venue") or raw["info"].get("city") or "unknown")
        )
        regular_innings = [
            item for item in raw.get("innings", []) if not bool(item.get("super_over", False))
        ]

        match_batter_events: list[tuple[str, str, str, dict[str, int]]] = []
        match_bowler_events: list[tuple[str, str, str, dict[str, int]]] = []

        for innings_number, innings in enumerate(regular_innings, start=1):
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                phase = _phase(over_number)
                striker = player_key(str(deliveries[0].get("batter") or UNKNOWN_CATEGORY))
                bowler_of_over = player_key(str(deliveries[0].get("bowler") or UNKNOWN_CATEGORY))

                striker_phase_prior = batter_phase_history[f"{striker}|{phase}"]
                striker_venue_phase_prior = batter_venue_phase_history[f"{striker}|{venue}|{phase}"]
                bowler_phase_prior = bowler_phase_history[f"{bowler_of_over}|{phase}"]
                bowler_venue_phase_prior = bowler_venue_phase_history[
                    f"{bowler_of_over}|{venue}|{phase}"
                ]
                striker_phase_innings_prior = batter_phase_innings_history[
                    f"{striker}|{phase}|{innings_number}"
                ]
                bowler_phase_innings_prior = bowler_phase_innings_history[
                    f"{bowler_of_over}|{phase}|{innings_number}"
                ]

                rows.append(
                    {
                        "source_file": path.name,
                        "match_date": match_date,
                        "innings": innings_number,
                        "over": over_number,
                        "venue_name": venue,
                        "phase": phase,
                        "striker": striker,
                        "known_bowler": bowler_of_over,
                        # Plain (venue-agnostic) phase aggregate -- the batter
                        # side of this never existed anywhere in the codebase
                        # before now (only bowlers had a phase split, via
                        # wicket_contract22_bowler_phase_stats.json). Exposed
                        # here as its own column, not just the internal
                        # shrinkage fallback, so batter and bowler get
                        # symmetric phase-level profiling.
                        "batter_phase_balls": striker_phase_prior["balls"],
                        "batter_phase_runs_per_ball": _rate(striker_phase_prior, "runs"),
                        "batter_phase_boundary_rate": _rate(striker_phase_prior, "boundaries"),
                        "batter_phase_dismissal_rate": _rate(striker_phase_prior, "dismissals"),
                        "bowler_phase_balls": bowler_phase_prior["balls"],
                        "bowler_phase_economy": _rate(bowler_phase_prior, "runs"),
                        "bowler_phase_wicket_rate": _rate(bowler_phase_prior, "wickets"),
                        "batter_venue_phase_balls": striker_venue_phase_prior["balls"],
                        "batter_venue_phase_runs_per_ball_shrunk": _shrunk_rate(
                            striker_venue_phase_prior, striker_phase_prior, "runs"
                        ),
                        "batter_venue_phase_boundary_rate_shrunk": _shrunk_rate(
                            striker_venue_phase_prior, striker_phase_prior, "boundaries"
                        ),
                        "batter_venue_phase_dismissal_rate_shrunk": _shrunk_rate(
                            striker_venue_phase_prior, striker_phase_prior, "dismissals"
                        ),
                        "bowler_venue_phase_balls": bowler_venue_phase_prior["balls"],
                        "bowler_venue_phase_economy_shrunk": _shrunk_rate(
                            bowler_venue_phase_prior, bowler_phase_prior, "runs"
                        ),
                        "bowler_venue_phase_wicket_rate_shrunk": _shrunk_rate(
                            bowler_venue_phase_prior, bowler_phase_prior, "wickets"
                        ),
                        # 1st-innings (setting) vs 2nd-innings (chasing)
                        # split of the same phase -- shrunk toward the
                        # innings-agnostic batter_phase/bowler_phase rates
                        # above, not toward venue.
                        "batter_phase_innings_balls": striker_phase_innings_prior["balls"],
                        "batter_phase_innings_runs_per_ball_shrunk": _shrunk_rate(
                            striker_phase_innings_prior, striker_phase_prior, "runs",
                            shrinkage=SHRINKAGE_BALLS_INNINGS,
                        ),
                        "batter_phase_innings_boundary_rate_shrunk": _shrunk_rate(
                            striker_phase_innings_prior, striker_phase_prior, "boundaries",
                            shrinkage=SHRINKAGE_BALLS_INNINGS,
                        ),
                        "batter_phase_innings_dismissal_rate_shrunk": _shrunk_rate(
                            striker_phase_innings_prior, striker_phase_prior, "dismissals",
                            shrinkage=SHRINKAGE_BALLS_INNINGS,
                        ),
                        "bowler_phase_innings_balls": bowler_phase_innings_prior["balls"],
                        "bowler_phase_innings_economy_shrunk": _shrunk_rate(
                            bowler_phase_innings_prior, bowler_phase_prior, "runs",
                            shrinkage=SHRINKAGE_BALLS_INNINGS,
                        ),
                        "bowler_phase_innings_wicket_rate_shrunk": _shrunk_rate(
                            bowler_phase_innings_prior, bowler_phase_prior, "wickets",
                            shrinkage=SHRINKAGE_BALLS_INNINGS,
                        ),
                    }
                )

                for delivery in deliveries:
                    batter = player_key(str(delivery.get("batter") or UNKNOWN_CATEGORY))
                    bowler = player_key(str(delivery.get("bowler") or UNKNOWN_CATEGORY))
                    total = int(delivery.get("runs", {}).get("total", 0))
                    batter_runs = int(delivery.get("runs", {}).get("batter", 0))
                    wicket = int(_is_wicket(delivery))
                    legal = int(_is_legal(delivery))
                    boundary = int(batter_runs in {4, 6})
                    dot = int(total == 0)
                    if legal:
                        match_batter_events.append(
                            (
                                f"{batter}|{phase}",
                                f"{batter}|{venue}|{phase}",
                                f"{batter}|{phase}|{innings_number}",
                                {
                                    "balls": 1, "runs": batter_runs, "dots": dot,
                                    "boundaries": boundary, "dismissals": 0,
                                },
                            )
                        )
                        match_bowler_events.append(
                            (
                                f"{bowler}|{phase}",
                                f"{bowler}|{venue}|{phase}",
                                f"{bowler}|{phase}|{innings_number}",
                                {
                                    "balls": 1, "runs": total, "dots": dot,
                                    "boundaries": boundary, "wickets": wicket,
                                },
                            )
                        )
                    for dismissal in delivery.get("wickets", []):
                        if dismissal.get("kind") not in {"retired hurt", "obstructing the field"}:
                            out_name = player_key(
                                str(dismissal.get("player_out") or UNKNOWN_CATEGORY)
                            )
                            match_batter_events.append(
                                (
                                    f"{out_name}|{phase}",
                                    f"{out_name}|{venue}|{phase}",
                                    f"{out_name}|{phase}|{innings_number}",
                                    {
                                        "balls": 0, "runs": 0, "dots": 0,
                                        "boundaries": 0, "dismissals": 1,
                                    },
                                )
                            )

        # Update profiles only after every row in the match has been emitted.
        for phase_key, venue_key, phase_innings_key, event in match_batter_events:
            for field, value in event.items():
                batter_phase_history[phase_key][field] += value
                batter_venue_phase_history[venue_key][field] += value
                batter_phase_innings_history[phase_innings_key][field] += value
        for phase_key, venue_key, phase_innings_key, event in match_bowler_events:
            for field, value in event.items():
                bowler_phase_history[phase_key][field] += value
                bowler_venue_phase_history[venue_key][field] += value
                bowler_phase_innings_history[phase_innings_key][field] += value

    frame = pd.DataFrame(rows)
    if not frame.empty and frame.duplicated(KEYS).any():
        raise ValueError("Player venue-phase features contain duplicate over keys.")
    return frame.sort_values(KEYS).reset_index(drop=True)
