"""Leakage-safe IPL venue scoring regimes for pre-over prediction."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.candidate_v3_dataset import _is_legal, _is_wicket, _phase
from app.ml.ipl_venues import normalize_ipl_venue, team_venue_context

MIN_PRIOR_INNINGS = 5
DEFAULT_PAR_SCORE = 170.0
# Venues with 1-4 prior innings used to fall all the way back to the
# global average -- a hard cliff that throws away real (if thin) venue
# signal. Blend the venue average toward the global average instead,
# weighted by how many prior innings actually exist (a pseudo-count of
# VENUE_SHRINKAGE_INNINGS "trust" in the global prior). At 0 innings this
# collapses to the pure global/default value exactly as before.
VENUE_SHRINKAGE_INNINGS = 8.0


def venue_regime(par_score: float, prior_innings: int) -> str:
    """Classify a ground using only its historical first-innings par."""
    if par_score >= 180:
        return "high"
    if par_score >= 150:
        return "medium"
    return "low"


def _innings_total(innings: dict[str, Any]) -> int:
    return sum(
        int(delivery.get("runs", {}).get("total", 0))
        for over in innings.get("overs", [])
        for delivery in over.get("deliveries", [])
    )


def _batting_average(
    profile: dict[str, int], global_profile: dict[str, int]
) -> float:
    global_average = (
        global_profile["runs"] / global_profile["dismissals"]
        if global_profile["dismissals"]
        else 20.0
    )
    return (profile.get("runs", 0) + global_average * 5) / (
        profile.get("dismissals", 0) + 5
    )


def _momentum_score(recent: deque[dict[str, int]]) -> float:
    if not recent:
        return 50.0
    balls = len(recent)
    runs_per_ball = sum(item["runs"] for item in recent) / balls
    boundary_rate = sum(item["boundary"] for item in recent) / balls
    dot_rate = sum(item["dot"] for item in recent) / balls
    wicket_rate = sum(item["wicket"] for item in recent) / balls
    return max(
        0.0,
        min(
            100.0,
            50.0
            + (runs_per_ball - 1.25) * 20
            + boundary_rate * 30
            - dot_rate * 20
            - wicket_rate * 35,
        ),
    )


def build_ipl_venue_regime_dataset(
    project_root: Path, *, scopes: tuple[str, ...] = ("ipl",)
) -> pd.DataFrame:
    """Return one pre-over venue regime row for every regular T20 innings.

    `scopes` selects which `data/raw/cricsheet/<scope>` directories to draw
    matches from. Defaults to IPL-only (unchanged, existing-caller-safe
    behavior); pass `("ipl", "t20i")` to also include T20I matches. The
    par-score-vs-venue-history math is format-agnostic; `team_venue_context`
    (home/away/neutral) is IPL-franchise-specific and will read as neutral
    for national teams, since T20I doesn't have a "home franchise venue"
    concept the way IPL does.
    """
    paths: list[tuple[str, Path]] = []
    for scope in scopes:
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json"):
            raw = json.loads(path.read_text(encoding="utf-8"))
            paths.append((str(raw["info"]["dates"][0]), path))
    paths.sort(key=lambda item: (item[0], item[1].name))

    totals: dict[str, list[int]] = defaultdict(list)
    global_totals: list[int] = []
    batter_profiles: dict[str, dict[str, int]] = defaultdict(
        lambda: {"runs": 0, "dismissals": 0}
    )
    global_batting_profile = {"runs": 0, "dismissals": 0}
    rows: list[dict[str, Any]] = []
    for match_date, path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        info = raw["info"]
        source_venue = str(
            info.get("venue") or info.get("city") or "unknown"
        ).strip()
        venue = normalize_ipl_venue(source_venue)
        prior = totals[venue]
        global_avg = float(sum(global_totals) / len(global_totals)) if global_totals else DEFAULT_PAR_SCORE
        if prior:
            venue_avg = float(sum(prior) / len(prior))
            par_score = (
                len(prior) * venue_avg + VENUE_SHRINKAGE_INNINGS * global_avg
            ) / (len(prior) + VENUE_SHRINKAGE_INNINGS)
            par_source = "venue" if len(prior) >= MIN_PRIOR_INNINGS else "venue_blended"
        elif global_totals:
            par_score = global_avg
            par_source = "global"
        else:
            par_score = DEFAULT_PAR_SCORE
            par_source = "default"
        regime = venue_regime(par_score, len(prior))

        regular_innings = [
            innings
            for innings in raw.get("innings", [])
            if not bool(innings.get("super_over", False))
        ]
        match_batting_events: list[tuple[str, int, int]] = []
        for innings_number, innings in enumerate(regular_innings, start=1):
            innings_overs = innings.get("overs", [])
            first_deliveries = [
                delivery
                for over in innings_overs
                for delivery in over.get("deliveries", [])
            ]
            opening_names = []
            if first_deliveries:
                opening_names = [
                    str(first_deliveries[0].get("batter", "")),
                    str(first_deliveries[0].get("non_striker", "")),
                ]
            opening_profiles = [
                batter_profiles[name] for name in opening_names if name
            ]
            opening_average = (
                sum(
                    _batting_average(profile, global_batting_profile)
                    for profile in opening_profiles
                )
                / len(opening_profiles)
                if opening_profiles
                else _batting_average({}, global_batting_profile)
            )
            opening_dismissals = sum(
                profile["dismissals"] for profile in opening_profiles
            )
            recent: deque[dict[str, int]] = deque(maxlen=12)
            batter_runs: dict[str, int] = defaultdict(int)
            batter_balls: dict[str, int] = defaultdict(int)
            for over in innings_overs:
                deliveries = over.get("deliveries", [])
                phase = _phase(int(over["over"]) + 1)
                current_pair = []
                if deliveries:
                    current_pair = [
                        str(deliveries[0].get("batter", "")),
                        str(deliveries[0].get("non_striker", "")),
                    ]
                pair_runs = sum(batter_runs[name] for name in current_pair)
                pair_balls = sum(batter_balls[name] for name in current_pair)
                rows.append(
                    {
                        "source_file": path.name,
                        "match_date": match_date,
                        "innings": innings_number,
                        "over": int(over["over"]) + 1,
                        "venue_name": venue,
                        "venue_source_name": source_venue,
                        "batting_team_venue_context": team_venue_context(
                            str(innings.get("team", "")), source_venue
                        ),
                        "venue_prior_innings": len(prior),
                        "venue_par_score": par_score,
                        "venue_par_source": par_source,
                        "venue_scoring_regime": regime,
                        "phase_venue_regime": f"{phase}|{regime}",
                        "momentum_score": _momentum_score(recent),
                        "current_pair_strike_rate": (
                            pair_runs * 100 / pair_balls if pair_balls else 0.0
                        ),
                        "opening_batters_prior_average": opening_average,
                        "opening_batters_prior_dismissals": opening_dismissals,
                    }
                )
                for delivery in deliveries:
                    batter = str(delivery.get("batter", ""))
                    batter_run = int(delivery.get("runs", {}).get("batter", 0))
                    batter_runs[batter] += batter_run
                    dismissal_count = 0
                    for wicket in delivery.get("wickets", []):
                        kind = str(wicket.get("kind", ""))
                        if kind not in {"retired hurt", "obstructing the field"}:
                            dismissed = str(wicket.get("player_out", ""))
                            if dismissed:
                                match_batting_events.append((dismissed, 0, 1))
                                dismissal_count += 1
                    match_batting_events.append((batter, batter_run, 0))
                    if _is_legal(delivery):
                        batter_balls[batter] += 1
                        total = int(delivery.get("runs", {}).get("total", 0))
                        recent.append(
                            {
                                "runs": total,
                                "boundary": int(batter_run in {4, 6}),
                                "dot": int(total == 0),
                                "wicket": int(_is_wicket(delivery)),
                            }
                        )

        # Freeze the venue state for the whole match, then learn its first
        # innings only after every pre-over row has been emitted.
        if regular_innings:
            first_innings_total = _innings_total(regular_innings[0])
            totals[venue].append(first_innings_total)
            global_totals.append(first_innings_total)
        for batter, runs, dismissals in match_batting_events:
            batter_profiles[batter]["runs"] += runs
            batter_profiles[batter]["dismissals"] += dismissals
            global_batting_profile["runs"] += runs
            global_batting_profile["dismissals"] += dismissals

    result = pd.DataFrame(rows)
    keys = ["source_file", "innings", "over"]
    if not result.empty and result.duplicated(keys).any():
        raise ValueError("IPL venue-regime dataset contains duplicate over keys.")
    return result.sort_values(
        ["match_date", "source_file", "innings", "over"]
    ).reset_index(drop=True)
