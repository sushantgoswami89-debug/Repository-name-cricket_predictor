"""Leakage-safe, recency-weighted batter/bowler form.

CTO-flagged direction (2026-08-23): every prior-stat feature in this
codebase (`striker_prior_runs_per_ball` in `ipl_phase_moe_dataset.py`,
`bowl_hist_avg_runs_conceded` in `train_contract22_rigorous.py`) is a flat
career-to-date cumulative average -- `profile[field] += value`, applied
once per match, no decay. A player's rate from 40 matches ago counts
exactly as much as last week's. Real player form changes over a career
(form slumps, improvement, aging); this has never been tested in this
codebase.

Same leakage-safe shape as the existing flat trackers (state frozen at
match start, updated only after a match completes -- see
`ipl_phase_moe_dataset.py`'s `batter_history[name][field] += value` for
the pattern this mirrors), except each update first decays the existing
state:

    state[field] = MATCH_DECAY * state[field] + this_match_event[field]

This is a match-indexed EWMA (each of a player's own matches is "one
tick," not calendar time), the standard "recent form" framing. Both
numerator and denominator (runs and balls, wickets and balls) decay
identically, so the resulting rate stays a proper weighted average --
just one that weights recent matches more heavily. `MATCH_DECAY = 0.95`
gives roughly a 13-14 match half-life (about one IPL season), the
simplest defensible default -- not swept/tuned, to avoid the exact
multi-variant search this project has already flagged as a dead end
elsewhere (`candidate_ipl_wicket_v7_2_spell_features.md`'s "weight
schemes... genuinely exhausted" note).

Batter recency features are usable everywhere (the striker is always
known at publish time). Bowler recency features follow the same
known/unknown convention already used for `bowl_hist_*` -- usable only
when the over's bowler is actually known, same infrastructure, no new
leakage risk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.candidate_v3_dataset import _phase
from app.ml.ipl_identities import canonical_player_id

MATCH_DECAY = 0.95


def _empty_batter() -> dict[str, float]:
    return {"balls": 0.0, "runs": 0.0, "dots": 0.0, "boundaries": 0.0, "dismissals": 0.0}


def _empty_bowler() -> dict[str, float]:
    return {"balls": 0.0, "runs": 0.0, "wickets": 0.0}


def _rate(profile: dict[str, float], event: str) -> float:
    return float(profile[event]) / max(1.0, float(profile["balls"]))


def _iter_matches(
    project_root: Path, scopes: tuple[str, ...]
) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for scope in scopes:
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json"):
            raw = json.loads(path.read_text(encoding="utf-8"))
            paths.append((str(raw["info"]["dates"][0]), path))
    paths.sort(key=lambda item: (item[0], item[1].name))
    return paths


def compute_final_recency_state(
    project_root: Path, scopes: tuple[str, ...] = ("ipl", "t20i")
) -> tuple[
    dict[str, dict[str, float]], dict[str, dict[str, float]],
    dict[str, dict[str, float]], dict[str, dict[str, float]],
]:
    """Returns (batter_form, bowler_form, bowler_phase_form, batter_phase_form)
    as of the most recent available match.

    Same chronological EWMA update as `build_recency_weighted_prior_dataset`,
    without materializing per-over rows -- for live snapshot builders that
    only need the final state, not the full training frame. See
    `build_recency_weighted_live_snapshots.py`. `bowler_phase_form`/
    `batter_phase_form` are keyed "{player_id}|{phase}" -- see
    `build_bowler_phase_recency_dataset`/`build_batter_phase_recency_dataset`'s
    docstrings for why phase-specific recency is a distinct signal from
    phase-agnostic recency.
    """
    batter_form: dict[str, dict[str, float]] = {}
    bowler_form: dict[str, dict[str, float]] = {}
    bowler_phase_form: dict[str, dict[str, float]] = {}
    batter_phase_form: dict[str, dict[str, float]] = {}
    for match_date, path in _iter_matches(project_root, scopes):
        raw = json.loads(path.read_text(encoding="utf-8"))
        registry = raw["info"].get("registry", {}).get("people", {})
        regular_innings = [
            item for item in raw.get("innings", []) if not bool(item.get("super_over", False))
        ]
        match_batter_events: dict[str, dict[str, float]] = {}
        match_bowler_events: dict[str, dict[str, float]] = {}
        match_bowler_phase_events: dict[str, dict[str, float]] = {}
        match_batter_phase_events: dict[str, dict[str, float]] = {}
        for innings in regular_innings:
            for source_over in innings.get("overs", []):
                over_number = int(source_over["over"]) + 1
                phase = _phase(over_number)
                for delivery in source_over.get("deliveries", []):
                    batter = canonical_player_id(str(delivery.get("batter", "")), registry)
                    this_bowler = canonical_player_id(str(delivery.get("bowler", "")), registry)
                    extras = delivery.get("extras", {})
                    is_wide_or_noball = "wides" in extras or "noballs" in extras
                    runs = int(delivery.get("runs", {}).get("batter", 0))
                    total_runs = int(delivery.get("runs", {}).get("total", 0))
                    is_wicket = bool(delivery.get("wickets"))
                    batter_dismissed = any(
                        canonical_player_id(str(w.get("player_out", "")), registry) == batter
                        for w in delivery.get("wickets", [])
                    )
                    if not is_wide_or_noball:
                        event = match_batter_events.setdefault(batter, dict(_empty_batter()))
                        event["balls"] += 1
                        event["runs"] += runs
                        event["dots"] += int(runs == 0 and total_runs == 0)
                        event["boundaries"] += int(runs in (4, 6))
                        event["dismissals"] += int(batter_dismissed)

                        batter_phase_key = f"{batter}|{phase}"
                        batter_phase_event = match_batter_phase_events.setdefault(batter_phase_key, dict(_empty_batter()))
                        batter_phase_event["balls"] += 1
                        batter_phase_event["runs"] += runs
                        batter_phase_event["dots"] += int(runs == 0 and total_runs == 0)
                        batter_phase_event["boundaries"] += int(runs in (4, 6))
                        batter_phase_event["dismissals"] += int(batter_dismissed)

                        bowler_event = match_bowler_events.setdefault(this_bowler, dict(_empty_bowler()))
                        bowler_event["balls"] += 1
                        bowler_event["runs"] += total_runs
                        bowler_event["wickets"] += int(is_wicket and delivery.get("wickets", [{}])[0].get("kind") != "run out")

                        phase_key = f"{this_bowler}|{phase}"
                        phase_event = match_bowler_phase_events.setdefault(phase_key, dict(_empty_bowler()))
                        phase_event["balls"] += 1
                        phase_event["runs"] += total_runs
                        phase_event["wickets"] += int(is_wicket and delivery.get("wickets", [{}])[0].get("kind") != "run out")
        for name, event in match_batter_events.items():
            state = batter_form.get(name, _empty_batter())
            batter_form[name] = {field: MATCH_DECAY * state[field] + event[field] for field in state}
        for name, event in match_bowler_events.items():
            state = bowler_form.get(name, _empty_bowler())
            bowler_form[name] = {field: MATCH_DECAY * state[field] + event[field] for field in state}
        for name, event in match_bowler_phase_events.items():
            state = bowler_phase_form.get(name, _empty_bowler())
            bowler_phase_form[name] = {field: MATCH_DECAY * state[field] + event[field] for field in state}
        for name, event in match_batter_phase_events.items():
            state = batter_phase_form.get(name, _empty_batter())
            batter_phase_form[name] = {field: MATCH_DECAY * state[field] + event[field] for field in state}
    return batter_form, bowler_form, bowler_phase_form, batter_phase_form


def build_bowler_phase_recency_dataset(
    project_root: Path, scopes: tuple[str, ...] = ("ipl", "t20i")
) -> pd.DataFrame:
    """Phase-specific bowler recency: is a recent economy/wicket-rate
    computed WITHIN the same phase as the current over more predictive
    than the phase-agnostic `bowler_recency_economy`/`bowler_recency_wicket_rate`
    from `build_recency_weighted_prior_dataset`? Motivation: bowlers are
    often phase specialists (strong at death, weak in the powerplay, or
    vice versa) -- a phase-agnostic recency average blurs together very
    different situations the existing flat `bowl_phase_avg_runs` feature
    already knows to keep separate, just without any recency weighting.
    This combines both levers rather than testing either alone again.

    Separate function (not an extension of `build_recency_weighted_prior_dataset`)
    so the already-promoted v9/v10 training scripts' exact reproducibility
    is untouched by this addition.
    """
    paths = _iter_matches(project_root, scopes)
    bowler_phase_form: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []

    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        registry = raw["info"].get("registry", {}).get("people", {})
        regular_innings = [
            item for item in raw.get("innings", []) if not bool(item.get("super_over", False))
        ]
        match_bowler_phase_events: dict[str, dict[str, float]] = {}

        for innings_number, innings in enumerate(regular_innings, start=1):
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                phase = _phase(over_number)
                bowler = canonical_player_id(str(deliveries[0].get("bowler", "")), registry)
                phase_state = bowler_phase_form.get(f"{bowler}|{phase}", _empty_bowler())
                rows.append(
                    {
                        "source_file": path.name,
                        "innings": innings_number,
                        "over": over_number,
                        "bowler_recency_phase_balls": phase_state["balls"],
                        "bowler_recency_phase_economy": 6.0 * _rate(phase_state, "runs"),
                        "bowler_recency_phase_wicket_rate": _rate(phase_state, "wickets"),
                    }
                )
                for delivery in deliveries:
                    this_bowler = canonical_player_id(str(delivery.get("bowler", "")), registry)
                    extras = delivery.get("extras", {})
                    is_wide_or_noball = "wides" in extras or "noballs" in extras
                    total_runs = int(delivery.get("runs", {}).get("total", 0))
                    is_wicket = bool(delivery.get("wickets"))
                    if not is_wide_or_noball:
                        key = f"{this_bowler}|{phase}"
                        event = match_bowler_phase_events.setdefault(key, dict(_empty_bowler()))
                        event["balls"] += 1
                        event["runs"] += total_runs
                        event["wickets"] += int(is_wicket and delivery.get("wickets", [{}])[0].get("kind") != "run out")

        for key, event in match_bowler_phase_events.items():
            state = bowler_phase_form.get(key, _empty_bowler())
            bowler_phase_form[key] = {
                field: MATCH_DECAY * state[field] + event[field] for field in state
            }

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Bowler-phase recency dataset contains duplicate over keys.")
    return result


def build_batter_phase_recency_dataset(
    project_root: Path, scopes: tuple[str, ...] = ("ipl", "t20i")
) -> pd.DataFrame:
    """Phase-specific batter recency: is a recent strike-rate/dismissal-rate
    computed WITHIN the same phase as the current over more predictive
    than the phase-agnostic `striker_recency_runs_per_ball` from
    `build_recency_weighted_prior_dataset`, or the flat, non-recency
    `batter_phase_runs_per_ball` already live in
    `run_range_v11_partnership_rate`? Motivation, from real-world research
    (2026-08-23): death overs are the single most decisive phase in IPL
    match outcomes (71.4% win correlation per external analysis, vs 64.2%
    powerplay); "finishing ability" (death-overs hitting specifically) is
    a well-documented distinct skill from general batting form, and a
    player's current finishing form can be in a slump or a purple patch
    just like any other skill -- the same "no recency weighting" gap
    already fixed for the phase-agnostic version, now applied to the
    phase-specific one. Mirrors `build_bowler_phase_recency_dataset`'s
    approach exactly, batter side.

    Separate function, same reproducibility-safety rationale as the
    bowler-phase version.
    """
    paths = _iter_matches(project_root, scopes)
    batter_phase_form: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []

    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        registry = raw["info"].get("registry", {}).get("people", {})
        regular_innings = [
            item for item in raw.get("innings", []) if not bool(item.get("super_over", False))
        ]
        match_batter_phase_events: dict[str, dict[str, float]] = {}

        for innings_number, innings in enumerate(regular_innings, start=1):
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                phase = _phase(over_number)
                striker = canonical_player_id(str(deliveries[0].get("batter", "")), registry)
                phase_state = batter_phase_form.get(f"{striker}|{phase}", _empty_batter())
                rows.append(
                    {
                        "source_file": path.name,
                        "innings": innings_number,
                        "over": over_number,
                        "striker_recency_phase_balls": phase_state["balls"],
                        "striker_recency_phase_runs_per_ball": _rate(phase_state, "runs"),
                        "striker_recency_phase_boundary_rate": _rate(phase_state, "boundaries"),
                        "striker_recency_phase_dismissal_rate": _rate(phase_state, "dismissals"),
                    }
                )
                for delivery in deliveries:
                    batter = canonical_player_id(str(delivery.get("batter", "")), registry)
                    extras = delivery.get("extras", {})
                    is_wide_or_noball = "wides" in extras or "noballs" in extras
                    runs = int(delivery.get("runs", {}).get("batter", 0))
                    total_runs = int(delivery.get("runs", {}).get("total", 0))
                    batter_dismissed = any(
                        canonical_player_id(str(w.get("player_out", "")), registry) == batter
                        for w in delivery.get("wickets", [])
                    )
                    if not is_wide_or_noball:
                        key = f"{batter}|{phase}"
                        event = match_batter_phase_events.setdefault(key, dict(_empty_batter()))
                        event["balls"] += 1
                        event["runs"] += runs
                        event["dots"] += int(runs == 0 and total_runs == 0)
                        event["boundaries"] += int(runs in (4, 6))
                        event["dismissals"] += int(batter_dismissed)

        for key, event in match_batter_phase_events.items():
            state = batter_phase_form.get(key, _empty_batter())
            batter_phase_form[key] = {
                field: MATCH_DECAY * state[field] + event[field] for field in state
            }

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Batter-phase recency dataset contains duplicate over keys.")
    return result


def build_recency_weighted_prior_dataset(
    project_root: Path, scopes: tuple[str, ...] = ("ipl", "t20i")
) -> pd.DataFrame:
    paths = _iter_matches(project_root, scopes)

    batter_form: dict[str, dict[str, float]] = {}
    bowler_form: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []

    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        registry = raw["info"].get("registry", {}).get("people", {})
        regular_innings = [
            item for item in raw.get("innings", []) if not bool(item.get("super_over", False))
        ]
        match_batter_events: dict[str, dict[str, float]] = {}
        match_bowler_events: dict[str, dict[str, float]] = {}

        for innings_number, innings in enumerate(regular_innings, start=1):
            for source_over in innings.get("overs", []):
                deliveries = source_over.get("deliveries", [])
                if not deliveries:
                    continue
                over_number = int(source_over["over"]) + 1
                striker = canonical_player_id(str(deliveries[0].get("batter", "")), registry)
                non_striker = canonical_player_id(str(deliveries[0].get("non_striker", "")), registry)
                bowler = canonical_player_id(str(deliveries[0].get("bowler", "")), registry)
                striker_form = batter_form.get(striker, _empty_batter())
                partner_form = batter_form.get(non_striker, _empty_batter())
                bowler_form_state = bowler_form.get(bowler, _empty_bowler())
                rows.append(
                    {
                        "source_file": path.name,
                        "innings": innings_number,
                        "over": over_number,
                        "striker_recency_balls": striker_form["balls"],
                        "striker_recency_runs_per_ball": _rate(striker_form, "runs"),
                        "striker_recency_dot_rate": _rate(striker_form, "dots"),
                        "striker_recency_boundary_rate": _rate(striker_form, "boundaries"),
                        "striker_recency_dismissal_rate": _rate(striker_form, "dismissals"),
                        "partner_recency_runs_per_ball": _rate(partner_form, "runs"),
                        "bowler_recency_balls": bowler_form_state["balls"],
                        "bowler_recency_economy": 6.0 * _rate(bowler_form_state, "runs"),
                        "bowler_recency_wicket_rate": _rate(bowler_form_state, "wickets"),
                    }
                )
                for delivery in deliveries:
                    batter = canonical_player_id(str(delivery.get("batter", "")), registry)
                    this_bowler = canonical_player_id(str(delivery.get("bowler", "")), registry)
                    extras = delivery.get("extras", {})
                    is_wide_or_noball = "wides" in extras or "noballs" in extras
                    runs = int(delivery.get("runs", {}).get("batter", 0))
                    total_runs = int(delivery.get("runs", {}).get("total", 0))
                    is_wicket = bool(delivery.get("wickets"))
                    batter_dismissed = any(
                        canonical_player_id(str(w.get("player_out", "")), registry) == batter
                        for w in delivery.get("wickets", [])
                    )
                    if not is_wide_or_noball:
                        event = match_batter_events.setdefault(batter, dict(_empty_batter()))
                        event["balls"] += 1
                        event["runs"] += runs
                        event["dots"] += int(runs == 0 and total_runs == 0)
                        event["boundaries"] += int(runs in (4, 6))
                        event["dismissals"] += int(batter_dismissed)

                        bowler_event = match_bowler_events.setdefault(this_bowler, dict(_empty_bowler()))
                        bowler_event["balls"] += 1
                        bowler_event["runs"] += total_runs
                        bowler_event["wickets"] += int(is_wicket and delivery.get("wickets", [{}])[0].get("kind") != "run out")

        for name, event in match_batter_events.items():
            state = batter_form.get(name, _empty_batter())
            batter_form[name] = {
                field: MATCH_DECAY * state[field] + event[field] for field in state
            }
        for name, event in match_bowler_events.items():
            state = bowler_form.get(name, _empty_bowler())
            bowler_form[name] = {
                field: MATCH_DECAY * state[field] + event[field] for field in state
            }

    result = pd.DataFrame(rows)
    if result.duplicated(["source_file", "innings", "over"]).any():
        raise ValueError("Recency-weighted prior dataset contains duplicate over keys.")
    return result
