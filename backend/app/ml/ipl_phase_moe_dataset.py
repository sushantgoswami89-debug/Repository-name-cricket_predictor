"""Leakage-safe, live-parity features for the IPL phase mixture experiment."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.candidate_v3_dataset import _is_legal, _is_wicket, _phase
from app.ml.ipl_identities import canonical_player_id, canonical_team_id

UNKNOWN_CATEGORY = "__UNKNOWN__"
KEYS = ["source_file", "match_date", "innings", "over"]


def _empty_batter() -> dict[str, int]:
    return {"balls": 0, "runs": 0, "dots": 0, "boundaries": 0, "dismissals": 0}


def _empty_bowler() -> dict[str, int]:
    return {"balls": 0, "runs": 0, "dots": 0, "boundaries": 0, "wickets": 0}


def _rate(profile: dict[str, int], event: str) -> float:
    return float(profile[event]) / max(1, int(profile["balls"]))


def _pressure_state(
    wickets_down: int, recent: deque[dict[str, int]], innings_rate: float
) -> str:
    recent_wickets = sum(item["wicket"] for item in recent)
    recent_rate = (
        sum(item["runs"] for item in recent) / len(recent) if recent else innings_rate
    )
    if recent_wickets >= 2 or wickets_down >= 7:
        return "wicket_pressure"
    if len(recent) >= 6 and recent_rate >= innings_rate + 0.35:
        return "accelerating"
    return "stable"


def _chase_pressure(required_rate: float, current_rate: float, is_chase: bool) -> str:
    if not is_chase:
        return "not_chasing"
    gap = required_rate - current_rate
    if gap >= 3:
        return "high"
    if gap >= 1:
        return "medium"
    return "low"


def build_ipl_phase_moe_features(
    project_root: Path, *, canonical_identities: bool = False
) -> pd.DataFrame:
    """Build one row per regular IPL over using only publish-time information.

    Player career profiles are frozen at match start. The next-over bowler is
    deliberately represented as unknown: Cricsheet's first-delivery bowler is
    future information when the prediction is published before the over.
    """

    paths: list[tuple[str, Path]] = []
    for path in (project_root / "data/raw/cricsheet/ipl").glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        registry = raw["info"].get("registry", {}).get("people", {})
        player_key = (
            (lambda name: canonical_player_id(name, registry))
            if canonical_identities
            else (lambda name: name)
        )
        paths.append((str(raw["info"]["dates"][0]), path))
    paths.sort(key=lambda item: (item[0], item[1].name))

    batter_history: dict[str, dict[str, int]] = defaultdict(_empty_batter)
    bowler_history: dict[str, dict[str, int]] = defaultdict(_empty_bowler)
    rows: list[dict[str, Any]] = []

    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        registry = raw["info"].get("registry", {}).get("people", {})
        player_key = (
            (lambda name: canonical_player_id(name, registry))
            if canonical_identities
            else (lambda name: name)
        )
        regular_innings = [
            item
            for item in raw.get("innings", [])
            if not bool(item.get("super_over", False))
        ]
        match_batter_events: list[tuple[str, dict[str, int]]] = []
        match_bowler_events: list[tuple[str, dict[str, int]]] = []
        first_total: int | None = None

        for innings_number, innings in enumerate(regular_innings, start=1):
            score = wickets = legal_balls = 0
            batter_match: dict[str, dict[str, int]] = defaultdict(_empty_batter)
            recent: deque[dict[str, int]] = deque(maxlen=12)
            scheduled_balls = int(raw["info"].get("overs", 20)) * 6
            target = first_total + 1 if innings_number == 2 and first_total is not None else 0

            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                striker = player_key(
                    str(deliveries[0].get("batter") or UNKNOWN_CATEGORY)
                )
                non_striker = player_key(str(
                    deliveries[0].get("non_striker") or UNKNOWN_CATEGORY
                ))
                striker_match = batter_match[striker]
                partner_match = batter_match[non_striker]
                striker_prior = batter_history[striker]
                partner_prior = batter_history[non_striker]
                balls_remaining = max(0, scheduled_balls - legal_balls)
                current_rate = score * 6 / legal_balls if legal_balls else 0.0
                runs_required = max(0, target - score) if target else 0
                required_rate = (
                    runs_required * 6 / balls_remaining
                    if target and balls_remaining
                    else 0.0
                )
                pair_age = striker_match["balls"] + partner_match["balls"]
                is_new = int(
                    min(striker_match["balls"], partner_match["balls"]) <= 2
                )
                rows.append(
                    {
                        "source_file": path.name,
                        "match_date": match_date,
                        "innings": innings_number,
                        "over": over_number,
                        "batting_team": (
                            canonical_team_id(str(innings.get("team", "")))
                            if canonical_identities
                            else str(innings.get("team") or UNKNOWN_CATEGORY)
                        ),
                        "striker": striker,
                        "non_striker": non_striker,
                        "known_bowler": UNKNOWN_CATEGORY,
                        "active_batter_state": (
                            "new_batter" if is_new else "established_pair"
                        ),
                        "new_batter": is_new,
                        "partnership_legal_ball_age": pair_age,
                        "wickets_remaining_bucket": str(max(0, 10 - wickets)),
                        "state_regime": _pressure_state(
                            wickets, recent, current_rate / 6
                        ),
                        "chase_pressure": _chase_pressure(
                            required_rate, current_rate, bool(target)
                        ),
                        "striker_match_balls": striker_match["balls"],
                        "partner_match_balls": partner_match["balls"],
                        "striker_prior_balls": striker_prior["balls"],
                        "striker_prior_runs_per_ball": _rate(
                            striker_prior, "runs"
                        ),
                        "striker_prior_dot_rate": _rate(striker_prior, "dots"),
                        "striker_prior_boundary_rate": _rate(
                            striker_prior, "boundaries"
                        ),
                        "striker_prior_dismissal_rate": (
                            striker_prior["dismissals"]
                            / max(1, striker_prior["balls"])
                        ),
                        "partner_prior_balls": partner_prior["balls"],
                        "partner_prior_runs_per_ball": _rate(
                            partner_prior, "runs"
                        ),
                        "partner_prior_dot_rate": _rate(partner_prior, "dots"),
                        "partner_prior_boundary_rate": _rate(
                            partner_prior, "boundaries"
                        ),
                    }
                )

                for delivery in deliveries:
                    batter = player_key(
                        str(delivery.get("batter") or UNKNOWN_CATEGORY)
                    )
                    bowler = player_key(
                        str(delivery.get("bowler") or UNKNOWN_CATEGORY)
                    )
                    total = int(delivery.get("runs", {}).get("total", 0))
                    batter_runs = int(delivery.get("runs", {}).get("batter", 0))
                    wicket = int(_is_wicket(delivery))
                    legal = int(_is_legal(delivery))
                    boundary = int(batter_runs in {4, 6})
                    dot = int(total == 0)
                    score += total
                    wickets += wicket
                    if legal:
                        legal_balls += 1
                        batter_match[batter]["balls"] += 1
                        batter_match[batter]["runs"] += batter_runs
                        batter_match[batter]["dots"] += dot
                        batter_match[batter]["boundaries"] += boundary
                        recent.append(
                            {
                                "runs": total,
                                "dot": dot,
                                "boundary": boundary,
                                "wicket": wicket,
                            }
                        )
                    for dismissal in delivery.get("wickets", []):
                        if dismissal.get("kind") not in {
                            "retired hurt",
                            "obstructing the field",
                        }:
                            batter_match[
                                player_key(str(
                                    dismissal.get("player_out") or UNKNOWN_CATEGORY
                                ))
                            ]["dismissals"] += 1
                    match_batter_events.append(
                        (
                            batter,
                            {
                                "balls": legal,
                                "runs": batter_runs,
                                "dots": dot * legal,
                                "boundaries": boundary * legal,
                                "dismissals": 0,
                            },
                        )
                    )
                    match_bowler_events.append(
                        (
                            bowler,
                            {
                                "balls": legal,
                                "runs": total,
                                "dots": dot * legal,
                                "boundaries": boundary * legal,
                                "wickets": wicket,
                            },
                        )
                    )
                    for dismissal in delivery.get("wickets", []):
                        if dismissal.get("kind") not in {
                            "retired hurt",
                            "obstructing the field",
                        }:
                            match_batter_events.append(
                                (
                                    player_key(str(
                                        dismissal.get("player_out")
                                        or UNKNOWN_CATEGORY
                                    )),
                                    {
                                        "balls": 0,
                                        "runs": 0,
                                        "dots": 0,
                                        "boundaries": 0,
                                        "dismissals": 1,
                                    },
                                )
                            )
            if innings_number == 1:
                first_total = score

        # Update profiles only after every row in the match has been emitted.
        for name, event in match_batter_events:
            for field, value in event.items():
                batter_history[name][field] += value
        for name, event in match_bowler_events:
            for field, value in event.items():
                bowler_history[name][field] += value

    frame = pd.DataFrame(rows)
    if not frame.empty and frame.duplicated(KEYS).any():
        raise ValueError("IPL phase-MoE features contain duplicate over keys.")
    return frame.sort_values(KEYS).reset_index(drop=True)
