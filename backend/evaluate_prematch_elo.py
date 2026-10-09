"""Backtest a pre-match Elo win predictor on men's T20I history.

Honest-baseline protocol (same as the rest of this project): ratings are
updated chronologically and every prediction is made BEFORE the match's own
result is applied. K is picked on 2023-2024 only; 2025+ is the untouched
holdout. Compared against a coin flip (Brier 0.25) and reported split by
full-member vs associate involvement, because a blended number hides that
associate fixtures are often lopsided.
"""

from __future__ import annotations

import math

from app.live.prematch_analysis import _all_rows

FULL_MEMBERS = {
    "India", "Australia", "England", "South Africa", "New Zealand", "Pakistan",
    "Sri Lanka", "Bangladesh", "West Indies", "Afghanistan", "Zimbabwe", "Ireland",
}
HOLDOUT_FROM = "2025-01-01"
TUNE_FROM = "2023-01-01"


def run(k: float, initial: float = 1500.0):
    ratings: dict[str, float] = {}
    preds = []
    for row in _all_rows():
        if not row["winner"] or len(row["teams"]) != 2:
            continue
        a, b = sorted(row["teams"])
        ra, rb = ratings.get(a, initial), ratings.get(b, initial)
        p_a = 1.0 / (1.0 + 10 ** (-(ra - rb) / 400.0))
        outcome = 1.0 if row["winner"] == a else 0.0
        preds.append((row["date"], a, b, p_a, outcome))
        ratings[a] = ra + k * (outcome - p_a)
        ratings[b] = rb + k * ((1 - outcome) - (1 - p_a))
    return ratings, preds


def metrics(rows):
    if not rows:
        return None
    brier = sum((p - o) ** 2 for _, _, _, p, o in rows) / len(rows)
    ll = -sum(math.log(max(1e-6, p if o else 1 - p)) for _, _, _, p, o in rows) / len(rows)
    acc = sum((p >= 0.5) == bool(o) for _, _, _, p, o in rows) / len(rows)
    return len(rows), acc, brier, ll


def fmt(label, m):
    if m is None:
        print(f"  {label}: no matches")
        return
    n, acc, brier, ll = m
    print(f"  {label}: n={n} accuracy={acc:.3f} brier={brier:.4f} (coin 0.2500) logloss={ll:.4f} (coin 0.6931)")


def main() -> None:
    print("K selection on 2023-2024 only:")
    best_k, best_brier = None, 9.0
    for k in (16, 24, 32, 48, 64, 96, 128):
        _, preds = run(k)
        tune = [r for r in preds if TUNE_FROM <= r[0] < HOLDOUT_FROM]
        m = metrics(tune)
        print(f"  K={k}: brier={m[2]:.4f} acc={m[1]:.3f} (n={m[0]})")
        if m[2] < best_brier:
            best_k, best_brier = k, m[2]
    print(f"chosen K={best_k}\n")

    ratings, preds = run(best_k)
    holdout = [r for r in preds if r[0] >= HOLDOUT_FROM]
    print(f"Holdout {HOLDOUT_FROM}+ (never used to pick K):")
    fmt("all men's T20Is", metrics(holdout))
    both_full = [r for r in holdout if r[1] in FULL_MEMBERS and r[2] in FULL_MEMBERS]
    assoc = [r for r in holdout if not (r[1] in FULL_MEMBERS and r[2] in FULL_MEMBERS)]
    fmt("both full members", metrics(both_full))
    fmt("associate involved", metrics(assoc))
    print("\nCurrent ratings, full members:")
    for team in sorted(FULL_MEMBERS, key=lambda t: -ratings.get(t, 1500)):
        print(f"  {team:<14} {ratings.get(team, 1500):.0f}")


if __name__ == "__main__":
    main()
