"""Backtest player-level (actual XI) spin/pace matchups against Elo+H2H(+dew).

Same protocol as the other pre-match evaluations: C chosen on 2023-24 (fit <2023),
final fit <2025, 2025+ scored once, paired-bootstrap Brier gain vs the
Elo+H2H+dew model, split full-member vs all. `xi_quality` (plain XI strength) is
tested separately so any matchup gain is beyond just "better players".
"""

from __future__ import annotations

import random

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.live.prematch_xi import build_dataset_xi

TUNE_FROM, HOLDOUT_FROM = "2023-01-01", "2025-01-01"
FULL = {"India", "Australia", "England", "South Africa", "New Zealand", "Pakistan", "Sri Lanka",
        "Bangladesh", "West Indies", "Afghanistan", "Zimbabwe", "Ireland"}
BASE = ("elo", "h2h", "h2h5")
DEW = BASE + ("dew_chase",)
VARIANTS = {
    "elo+h2h+dew (current)": DEW,
    "+xi_quality": DEW + ("xi_quality",),
    "+xi_matchup": DEW + ("xi_matchup",),
    "+xi_matchup_pitch": DEW + ("xi_matchup_pitch",),
    "+quality+matchup_pitch": DEW + ("xi_quality", "xi_matchup_pitch"),
    "matchup only (no dew)": BASE + ("xi_matchup_pitch",),
}


def mat(rows, names):
    return np.array([[r["x"][n] for n in names] for r in rows], float)


def fit(rows, names, c):
    x = mat(rows, names)
    scale = np.maximum(x.std(0), 1e-6)
    y = np.array([r["y"] for r in rows], float)
    return LogisticRegression(C=c, fit_intercept=False, max_iter=2000).fit(x / scale, y), scale


def proba(fitted, rows, names):
    model, scale = fitted
    return model.predict_proba(mat(rows, names) / scale)[:, 1]


def scores(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.mean((p >= .5) == (y == 1))), float(np.mean((p - y) ** 2)), float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def boot(new, base, y, reps=2000, seed=5):
    rng = random.Random(seed)
    d = (base - y) ** 2 - (new - y) ** 2
    n = len(d)
    m = sorted(float(np.mean([d[rng.randrange(n)] for _ in range(n)])) for _ in range(reps))
    return float(d.mean()), m[int(.025 * reps)], m[int(.975 * reps)]


def main() -> None:
    data = build_dataset_xi()
    train = [r for r in data if r["date"] < TUNE_FROM]
    tune = [r for r in data if TUNE_FROM <= r["date"] < HOLDOUT_FROM]
    full_train = [r for r in data if r["date"] < HOLDOUT_FROM]
    hold = [r for r in data if r["date"] >= HOLDOUT_FROM]
    y_t = np.array([r["y"] for r in tune], float)
    y_h = np.array([r["y"] for r in hold], float)
    full = np.array([r["a"] in FULL and r["b"] in FULL for r in hold])
    chosen = {}
    for name, feats in VARIANTS.items():
        best = min(((c, scores(proba(fit(train, feats, c), tune, feats), y_t)) for c in (0.03, 0.1, 0.3, 1, 3)),
                   key=lambda t: t[1][1])
        chosen[name] = best[0]
        print(f"tune  {name:<26} C={best[0]:<5} brier={best[1][1]:.4f} acc={best[1][0]:.3f}")
    probs = {n: proba(fit(full_train, f, chosen[n]), hold, f) for n, f in VARIANTS.items()}
    base = probs["elo+h2h+dew (current)"]
    for label, mask in (("both full members", full), ("all", np.ones(len(hold), bool)), ("associate involved", ~full)):
        print(f"\nHOLDOUT 2025+ {label} (n={int(mask.sum())})")
        for n in VARIANTS:
            acc, br, ll = scores(probs[n][mask], y_h[mask])
            tail = ""
            if n != "elo+h2h+dew (current)":
                g, lo, hi = boot(probs[n][mask], base[mask], y_h[mask])
                tail = f"  gain vs current {g:+.4f} [{lo:+.4f},{hi:+.4f}]"
            print(f"  {n:<26} acc={acc:.3f} brier={br:.4f} logloss={ll:.4f}{tail}")
    model, scale = fit(full_train, VARIANTS["+quality+matchup_pitch"], chosen["+quality+matchup_pitch"])
    print("\nweights (scaled units) +quality+matchup_pitch:",
          dict(zip(VARIANTS["+quality+matchup_pitch"], np.round(model.coef_[0], 3))))


def save_model() -> None:
    """Fit the XI quality + pitch-tilted matchup model on <2025 and save it.
    C=0.3 is the strength used in the rolling 2021-25 check (+0.0046 Brier over
    Elo+H2H+dew on 2,146 games, CI [+0.0019,+0.0073]; full members +0.0029,
    CI [-0.0019,+0.0081], n=443)."""
    import json

    from app.live.prematch_xi import XI_MODEL_PATH

    names = DEW + ("xi_quality", "xi_matchup_pitch")
    data = build_dataset_xi()
    train = [r for r in data if r["date"] < HOLDOUT_FROM]
    hold = [r for r in data if r["date"] >= HOLDOUT_FROM]
    model, scale = fit(train, names, 0.3)
    p = proba((model, scale), hold, names)
    y = np.array([r["y"] for r in hold], float)
    full = np.array([r["a"] in FULL and r["b"] in FULL for r in hold])
    fa, fb, _ = scores(p[full], y[full])
    aa, ab, _ = scores(p, y)
    XI_MODEL_PATH.write_text(json.dumps({
        "features": list(names), "scale": [float(v) for v in scale], "weights": [float(w) for w in model.coef_[0]],
        "C": 0.3, "trained_through": train[-1]["date"],
        "holdout": {"from": HOLDOUT_FROM,
                    "full_member": {"n": int(full.sum()), "acc": fa, "brier": fb},
                    "all": {"n": len(y), "acc": aa, "brier": ab}},
        "rolling_2021_25": {"all_n": 2146, "gain_vs_dew_model": 0.0046, "ci": [0.0019, 0.0073],
                            "full_member_n": 443, "full_member_gain": 0.0029, "full_member_ci": [-0.0019, 0.0081]},
    }, indent=2))
    print("saved", XI_MODEL_PATH, dict(zip(names, np.round(model.coef_[0], 3))))
    print("holdout full-member acc/brier", round(fa, 3), round(fb, 4), "| all", round(aa, 3), round(ab, 4))


if __name__ == "__main__":
    import sys

    save_model() if "--save" in sys.argv else main()
