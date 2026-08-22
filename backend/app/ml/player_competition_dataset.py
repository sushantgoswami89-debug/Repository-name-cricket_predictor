"""Leakage-safe batter/bowler competition-specific profiles.

User-flagged (2026-08-22): a 20-player spot check of real IPL-vs-T20I
career stats (see chat) found essentially zero population-level bias in
batting strike rate (mean diff -0.33) but enormous player-to-player
heterogeneity (stdev 13.8 SR points -- MS Wade is 31 points higher in
T20I than IPL, MV Boucher 28 points higher in IPL than T20I, no
consistent direction). Every player prior-stat feature this codebase has
ever used (`striker_prior_runs_per_ball` etc., from
`build_ipl_phase_moe_features`) pools a player's IPL and T20I history
into one number -- given swings this large, that pooled average actively
misrepresents either competition specifically for players whose game
genuinely differs between them.

Distinct from yesterday's rejected hypotheses: that work tested
TRAINING-ROW population mixing (reweighting T20I, full model separation)
and found the single pooled model already wins regardless of population.
This is a different question -- not which rows train the model, but
whether each player's own FEATURE VALUE should be split by competition.

This module outputs the RAW competition-specific rate + ball count only.
Shrinkage toward the player's own pooled (cross-competition) rate --
already computed and validated as `striker_prior_runs_per_ball` etc. --
happens at merge/training time (`weight = cell_balls / (cell_balls +
SHRINKAGE_BALLS)`), since the pooled fallback lives in a separately-built
dataframe (`build_ipl_phase_moe_features`). A player with no
competition-specific history collapses to the pooled rate (today's
status quo); a player with a lot of it (like Wade/Boucher above)
dominates it. Same style of shrinkage already used throughout
`player_venue_phase_dataset.py`.

Chronological, leakage-safe: profiles updated only after every row in a
match has been emitted.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.candidate_v3_dataset import _is_legal, _is_wicket
from app.ml.ipl_identities import canonical_player_id

UNKNOWN_CATEGORY = "__UNKNOWN__"
KEYS = ["source_file", "match_date", "innings", "over"]
SHRINKAGE_BALLS = 150  # matches the >=150-ball eligibility bar used in the spot check


def _empty_batter() -> dict[str, int]:
    return {"balls": 0, "runs": 0, "dots": 0, "boundaries": 0, "dismissals": 0}


def _empty_bowler() -> dict[str, int]:
    return {"balls": 0, "runs": 0, "dots": 0, "boundaries": 0, "wickets": 0}


def _rate(profile: dict[str, int], event: str) -> float:
    return float(profile[event]) / max(1, int(profile["balls"]))


def build_player_competition_features(
    project_root: Path,
    *,
    scopes: tuple[str, ...] = ("ipl", "t20i"),
) -> pd.DataFrame:
    """Return one pre-over competition-specific batter/bowler profile row
    per regular T20 over. `scopes` values also become the `competition`
    label (e.g. "ipl", "t20i")."""

    paths: list[tuple[str, str, Path]] = []  # (match_date, scope, path)
    for scope in scopes:
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json"):
            raw = json.loads(path.read_text(encoding="utf-8"))
            paths.append((str(raw["info"]["dates"][0]), scope, path))
    paths.sort(key=lambda item: (item[0], item[2].name))

    batter_competition_history: dict[str, dict[str, int]] = defaultdict(_empty_batter)
    bowler_competition_history: dict[str, dict[str, int]] = defaultdict(_empty_bowler)

    rows: list[dict[str, Any]] = []

    for match_date, scope, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        registry = raw["info"].get("registry", {}).get("people", {})

        def player_key(name: str) -> str:
            return canonical_player_id(name, registry)

        regular_innings = [
            item for item in raw.get("innings", []) if not bool(item.get("super_over", False))
        ]

        match_batter_events: list[tuple[str, dict[str, int]]] = []
        match_bowler_events: list[tuple[str, dict[str, int]]] = []

        for innings_number, innings in enumerate(regular_innings, start=1):
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                striker = player_key(str(deliveries[0].get("batter") or UNKNOWN_CATEGORY))
                bowler_of_over = player_key(str(deliveries[0].get("bowler") or UNKNOWN_CATEGORY))

                striker_comp_prior = batter_competition_history[f"{striker}|{scope}"]
                bowler_comp_prior = bowler_competition_history[f"{bowler_of_over}|{scope}"]

                rows.append(
                    {
                        "source_file": path.name,
                        "match_date": match_date,
                        "innings": innings_number,
                        "over": over_number,
                        "competition": scope,
                        "striker": striker,
                        "known_bowler": bowler_of_over,
                        "batter_competition_balls": striker_comp_prior["balls"],
                        "batter_competition_runs_per_ball": _rate(striker_comp_prior, "runs"),
                        "batter_competition_boundary_rate": _rate(striker_comp_prior, "boundaries"),
                        "batter_competition_dismissal_rate": _rate(striker_comp_prior, "dismissals"),
                        "bowler_competition_balls": bowler_comp_prior["balls"],
                        "bowler_competition_economy": _rate(bowler_comp_prior, "runs"),
                        "bowler_competition_wicket_rate": _rate(bowler_comp_prior, "wickets"),
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
                                f"{batter}|{scope}",
                                {
                                    "balls": 1, "runs": batter_runs, "dots": dot,
                                    "boundaries": boundary, "dismissals": 0,
                                },
                            )
                        )
                        match_bowler_events.append(
                            (
                                f"{bowler}|{scope}",
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
                                    f"{out_name}|{scope}",
                                    {
                                        "balls": 0, "runs": 0, "dots": 0,
                                        "boundaries": 0, "dismissals": 1,
                                    },
                                )
                            )

        for key, event in match_batter_events:
            for field, value in event.items():
                batter_competition_history[key][field] += value
        for key, event in match_bowler_events:
            for field, value in event.items():
                bowler_competition_history[key][field] += value

    frame = pd.DataFrame(rows)
    if not frame.empty and frame.duplicated(KEYS).any():
        raise ValueError("Player competition features contain duplicate over keys.")
    return frame.sort_values(KEYS).reset_index(drop=True)
