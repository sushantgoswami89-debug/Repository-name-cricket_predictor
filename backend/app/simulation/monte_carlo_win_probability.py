"""WinViz-style Monte Carlo win-probability simulation for chases (2026-08-24).

Follow-up to the LSTM sequence-model investigation (negative, closed --
see `docs/finding_wicket_lstm_sequence_model_not_promoted.md`). Rather
than another single-shot classifier, this simulates the REST of the chase
forward ball-by-ball, thousands of times, and reports the fraction of
simulated futures where the batting team reaches the target -- a
genuinely different mechanism from `match_winner_v1`'s trained LightGBM
classifier, motivated by CricViz's real WinViz system
(cricviz.com/winviz).

**Deliberate scope cut, disclosed up front**: real WinViz simulates at
the individual player level (each batter/bowler's own historical
distribution). This v1 is TEAM-AGNOSTIC -- outcome probabilities are
conditioned only on (phase, wickets_in_hand, pressure bucket), the same
match-state granularity `contract22_wicket_v16`'s `BASE_FEATURES` already
uses, not on player identity. This tests the core "simulate forward vs.
classify directly" mechanism first; player-level identity is a natural v2
extension if this shows real promise, not free to add now (would need
per-player run/dismissal distributions bootstrapped from career data with
proper shrinkage, plus a modeled batting order).

Also disclosed: wide/no-ball deliveries are folded into the outcome
distribution as +1 run, extra ball (not the exact real run penalty or
free-hit implications) -- a standard simplifying assumption for a first
prototype, not expected to matter much since illegal deliveries are a
small minority of balls.

**Performance note**: the simulation is vectorized across all `n_sims`
trajectories with numpy (categorical sampling via cumulative-sum +
uniform draw, table stored as a dense array for O(1) fancy-indexed
lookup) -- a naive per-simulation Python loop measured at ~0.7s/row
(2000 sims), which would have taken ~3.5 hours over the real ~18k-row
chase holdout. The vectorized version fixes the actual bottleneck rather
than silently subsampling.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from app.ml.candidate_v3_dataset import _is_legal, _is_wicket, _phase
from app.replay.evaluation_policy import KNOWN_RULE_ANOMALY_EXCLUSIONS

CATEGORIES = ["wide", "noball", "wicket", "0", "1", "2", "3", "4", "5", "6"]
CATEGORY_INDEX = {c: i for i, c in enumerate(CATEGORIES)}
RUN_VALUE = {"wide": 1, "noball": 1, "wicket": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
PRESSURE_CAP = 3
PHASE_ORDER = ["powerplay", "middle", "death"]
PHASE_INDEX = {p: i for i, p in enumerate(PHASE_ORDER)}
MAX_WICKETS_IN_HAND = 10


def _delivery_category(delivery: dict[str, Any]) -> str:
    if not _is_legal(delivery):
        extras = delivery.get("extras", {})
        return "noball" if "noballs" in extras else "wide"
    if _is_wicket(delivery):
        return "wicket"
    return str(min(int(delivery["runs"]["total"]), 6))


def build_outcome_table(
    project_root: Path, eligible_files: set[str], before_date: str = "2024-01-01"
) -> dict[tuple[str, int, int], np.ndarray]:
    """Team-agnostic per-ball outcome distribution, conditioned on
    (phase, wickets_in_hand, pressure_bucket), estimated from real
    training-period deliveries only (leakage-safe: `before_date` matches
    the project's standard train<=2023 cutoff).
    """

    excluded = set(KNOWN_RULE_ANOMALY_EXCLUSIONS)
    counts: dict[tuple[str, int, int], np.ndarray] = defaultdict(
        lambda: np.zeros(len(CATEGORIES), dtype=np.float64)
    )

    for scope in ("t20i", "ipl"):
        for path in (project_root / "data/raw/cricsheet" / scope).glob("*.json"):
            if path.stem in excluded or path.name not in eligible_files:
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            if str(raw["info"]["dates"][0]) >= before_date:
                continue
            for innings in raw.get("innings", []):
                wickets_down = 0
                dot_streak = 0
                for source_over in innings.get("overs", []):
                    deliveries = source_over.get("deliveries", [])
                    over_number = int(source_over["over"]) + 1
                    phase = _phase(over_number)
                    for delivery in deliveries:
                        wickets_in_hand = max(0, 10 - wickets_down)
                        pressure = min(dot_streak, PRESSURE_CAP)
                        category = _delivery_category(delivery)
                        counts[(phase, wickets_in_hand, pressure)][CATEGORY_INDEX[category]] += 1

                        if category == "wicket":
                            wickets_down += 1
                            dot_streak = 0
                        elif category in ("wide", "noball"):
                            pass
                        elif category == "0":
                            dot_streak += 1
                        else:
                            dot_streak = 0

    table: dict[tuple[str, int, int], np.ndarray] = {}
    for key, raw_counts in counts.items():
        smoothed = raw_counts + 1.0
        table[key] = smoothed / smoothed.sum()
    return table


def build_outcome_array(table: dict[tuple[str, int, int], np.ndarray]) -> np.ndarray:
    """Dense (phase=3, wickets_in_hand=0..10, pressure=0..3, category)
    array for fast vectorized fancy-indexed lookup during simulation.
    Missing (sparse/unobserved) buckets fall back to the global average
    distribution across all real observed buckets."""

    fallback = np.sum(list(table.values()), axis=0)
    fallback = fallback / fallback.sum()

    arr = np.tile(fallback, (len(PHASE_ORDER), MAX_WICKETS_IN_HAND + 1, PRESSURE_CAP + 1, 1))
    for (phase, wickets_in_hand, pressure), probs in table.items():
        arr[PHASE_INDEX[phase], wickets_in_hand, pressure] = probs
    return arr


def simulate_chase_win_probability(
    table_arr: np.ndarray,
    score: int,
    wickets_in_hand: int,
    legal_balls_bowled: int,
    total_legal_balls: int,
    target: int,
    n_sims: int = 2000,
    rng: np.random.Generator | None = None,
) -> float:
    """Fraction of `n_sims` simulated futures (vectorized with numpy)
    where the batting team reaches `target`, simulating forward from the
    given real mid-chase state. Pressure (dot-streak) resets to 0 at the
    start of each simulation -- the real recent-momentum state isn't
    carried in, a disclosed simplification (only phase/wickets_in_hand
    come from the real snapshot)."""

    if rng is None:
        rng = np.random.default_rng(42)
    if score >= target:
        return 1.0
    if wickets_in_hand <= 0 or legal_balls_bowled >= total_legal_balls:
        return 0.0

    category_values = np.array([RUN_VALUE[c] for c in CATEGORIES], dtype=np.int64)
    wicket_idx = CATEGORY_INDEX["wicket"]
    dot_idx = CATEGORY_INDEX["0"]
    wide_idx = CATEGORY_INDEX["wide"]
    noball_idx = CATEGORY_INDEX["noball"]

    scores = np.full(n_sims, score, dtype=np.int64)
    wickets = np.full(n_sims, wickets_in_hand, dtype=np.int64)
    balls = np.full(n_sims, legal_balls_bowled, dtype=np.int64)
    dot_streak = np.zeros(n_sims, dtype=np.int64)
    active = np.ones(n_sims, dtype=bool)

    max_attempts = max(1, total_legal_balls - legal_balls_bowled) * 4 + 20
    for _ in range(max_attempts):
        if not active.any():
            break

        over_number = balls // 6 + 1
        phase_idx = np.where(over_number <= 6, 0, np.where(over_number <= 15, 1, 2))
        wkt_idx_safe = np.clip(wickets, 1, MAX_WICKETS_IN_HAND)
        pressure_idx = np.minimum(dot_streak, PRESSURE_CAP)

        probs = table_arr[phase_idx, wkt_idx_safe, pressure_idx]
        cum = np.cumsum(probs, axis=1)
        u = rng.random(n_sims)
        outcome = np.sum(cum < u[:, None], axis=1)
        outcome = np.clip(outcome, 0, len(CATEGORIES) - 1)

        is_wkt = outcome == wicket_idx
        is_illegal = (outcome == wide_idx) | (outcome == noball_idx)
        is_dot = outcome == dot_idx
        is_legal = ~is_illegal

        scores = np.where(active, scores + category_values[outcome], scores)
        wickets = np.where(active & is_wkt, wickets - 1, wickets)

        reset_mask = active & (is_wkt | (is_legal & ~is_dot))
        incr_mask = active & is_dot
        dot_streak = np.where(incr_mask, dot_streak + 1, np.where(reset_mask, 0, dot_streak))

        balls = np.where(active & is_legal, balls + 1, balls)
        active = active & (scores < target) & (wickets > 0) & (balls < total_legal_balls)

    return float(np.sum(scores >= target)) / n_sims
