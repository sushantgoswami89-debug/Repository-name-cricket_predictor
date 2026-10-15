"""Backtest the pre-match win model against plain Elo on men's T20Is.

Protocol (same honesty rules as the rest of the project):
  * every feature is point-in-time (see app/live/prematch_model.py)
  * variants and the L2 strength are chosen on 2023-2024 using a model trained on
    matches before 2023; 2025+ is never used for any choice
  * the final model is refit on everything before 2025 and scored once on 2025+
  * results are split full-member vs associate, with a paired bootstrap on the
    Brier-score difference against Elo-only so "better" is not just noise
Run with --save to write models/prematch_t20i_logit.json (only do so when the
holdout says the model beats Elo on the full-member split).
"""

from __future__ import annotations

import argparse
import json
import math
import random

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.live.prematch_model import FEATURES, MODEL_PATH, build_dataset

FULL_MEMBERS = {
    "India", "Australia", "England", "South Africa", "New Zealand", "Pakistan",
    "Sri Lanka", "Bangladesh", "West Indies", "Afghanistan", "Zimbabwe", "Ireland",
}
TUNE_FROM, HOLDOUT_FROM = "2023-01-01", "2025-01-01"
VARIANTS = {
    "elo": ("elo",),
    "elo+form": ("elo", "form"),
    "elo+h2h": ("elo", "h2h", "h2h5"),
    "elo+venue(T20I)": ("elo", "venue_t20i"),
    "elo+venue(T20I+IPL)": ("elo", "venue_t20i", "venue_ipl"),
    "all": FEATURES,
}


def _matrix(rows, names, key):
    idx = [FEATURES.index(n) for n in names]
    x = np.array([[r[key][i] for i in idx] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows], dtype=float)
    return x, y


def _fit(rows, names, key, c):
    x, y = _matrix(rows, names, key)
    return LogisticRegression(C=c, fit_intercept=False, max_iter=1000).fit(x, y)


def _probs(model, rows, names, key):
    x, _ = _matrix(rows, names, key)
    return model.predict_proba(x)[:, 1]


def _scores(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": len(y),
        "acc": float(np.mean((p >= 0.5) == (y == 1))),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
    }


def _fmt(label, s):
    return (
        f"    {label:<24} n={s['n']:<4} acc={s['acc']:.3f} brier={s['brier']:.4f} "
        f"logloss={s['logloss']:.4f}"
    )


def _boot_ci(p_new, p_base, y, reps=2000, seed=7):
    """95% CI of (brier_base - brier_new): positive means the new model is better."""
    rng = random.Random(seed)
    d = (p_base - y) ** 2 - (p_new - y) ** 2
    n = len(d)
    means = sorted(float(np.mean([d[rng.randrange(n)] for _ in range(n)])) for _ in range(reps))
    return float(np.mean(d)), means[int(0.025 * reps)], means[int(0.975 * reps)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    data = build_dataset()
    train = [r for r in data if r["date"] < TUNE_FROM]
    tune = [r for r in data if TUNE_FROM <= r["date"] < HOLDOUT_FROM]
    full_train = [r for r in data if r["date"] < HOLDOUT_FROM]
    holdout = [r for r in data if r["date"] >= HOLDOUT_FROM]
    print(f"matches: train<2023 {len(train)}, tune 2023-24 {len(tune)}, holdout 2025+ {len(holdout)}")

    # Choose C per variant on the tune window (train<2023 -> score on 2023-24).
    chosen = {}
    print("\nTune window (2023-24), post-toss features:")
    for name, feats in VARIANTS.items():
        best = None
        for c in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
            model = _fit(train, feats, "x", c)
            s = _scores(_probs(model, tune, feats, "x"), np.array([r["y"] for r in tune], float))
            if best is None or s["brier"] < best[1]["brier"]:
                best = (c, s)
        chosen[name] = best[0]
        print(_fmt(f"{name} (C={best[0]})", best[1]))

    def is_full(r):
        return r["a"] in FULL_MEMBERS and r["b"] in FULL_MEMBERS

    results = {}
    for key, label in (("x", "POST-TOSS (bat-first known)"), ("x_pretoss", "PRE-TOSS")):
        print(f"\nHOLDOUT 2025+  {label}  (refit on <2025, scored once)")
        probs = {}
        for name, feats in VARIANTS.items():
            if key == "x_pretoss" and not (set(feats) - {"venue_t20i", "venue_ipl"}):
                pass
            model = _fit(full_train, feats, key, chosen[name])
            probs[name] = (model, _probs(model, holdout, feats, key))
        y_all = np.array([r["y"] for r in holdout], float)
        full_mask = np.array([is_full(r) for r in holdout])
        for split, mask in (("both full members", full_mask), ("associate involved", ~full_mask), ("all", np.ones(len(holdout), bool))):
            print(f"  {split}:")
            for name in VARIANTS:
                print(_fmt(name, _scores(probs[name][1][mask], y_all[mask])))
            print(_fmt("raw Elo (no fit)", _scores(np.array([r["elo_p"] for r, m in zip(holdout, mask) if m]), y_all[mask])))
        # Paired bootstrap vs elo-only, full-member split
        for name in VARIANTS:
            if name == "elo":
                continue
            diff, lo, hi = _boot_ci(probs[name][1][full_mask], probs["elo"][1][full_mask], y_all[full_mask])
            print(f"  full-member brier gain vs fitted Elo, {name}: {diff:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        results[key] = probs

    if args.save:
        feats = VARIANTS["all"]
        post = _fit(full_train, feats, "x", chosen["all"])
        pre = _fit(full_train, feats, "x_pretoss", chosen["all"])
        probs_post = results["x"]["all"][1]
        y_all = np.array([r["y"] for r in holdout], float)
        full_mask = np.array([is_full(r) for r in holdout])
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_text(
            json.dumps(
                {
                    "features": list(feats),
                    "weights_post_toss": [float(w) for w in post.coef_[0]],
                    "weights_pre_toss": [float(w) for w in pre.coef_[0]],
                    "trained_through": full_train[-1]["date"],
                    "holdout": {
                        "from": HOLDOUT_FROM,
                        "full_member": _scores(probs_post[full_mask], y_all[full_mask]),
                        "associate": _scores(probs_post[~full_mask], y_all[~full_mask]),
                    },
                },
                indent=2,
            )
        )
        print(f"\nsaved {MODEL_PATH}")


if __name__ == "__main__":
    main()
