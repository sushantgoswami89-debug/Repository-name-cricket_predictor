"""Backtest the conditions model (dew, cloud, pitch-type x team matchups) against
the current saved pre-match model and plain Elo on men's T20Is.

Same protocol as evaluate_prematch_model.py: variants and L2 strength are chosen
on 2023-24 (fit on <2023); the final fit uses <2025; 2025+ is scored once. A
feature block is only worth keeping if its paired-bootstrap Brier gain over the
Elo+H2H baseline on the 2025+ data excludes zero. Features are scaled (no
centering) so the zero-intercept antisymmetry is preserved.
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.live.prematch_conditions import DEW_MODEL_PATH, FEATURES_V2, build_dataset_v2

TUNE_FROM, HOLDOUT_FROM = "2023-01-01", "2025-01-01"
FULL = {"India", "Australia", "England", "South Africa", "New Zealand", "Pakistan", "Sri Lanka",
        "Bangladesh", "West Indies", "Afghanistan", "Zimbabwe", "Ireland"}
BASE = ("elo", "h2h", "h2h5")
VARIANTS = {
    "elo+h2h (baseline)": BASE,
    "+form": BASE + ("form",),
    "+dew": BASE + ("dew_chase",),
    "+cloud": BASE + ("cloud_chase",),
    "+dew+cloud": BASE + ("dew_chase", "cloud_chase"),
    "+spin matchups (pitch x team)": BASE + ("finger_x", "wrist_x"),
    "+pace matchup (pitch x team)": BASE + ("pace_x",),
    "+all matchups": BASE + ("finger_x", "wrist_x", "pace_x"),
    "everything": FEATURES_V2,
}


def mat(rows, names, key, scale):
    idx = [FEATURES_V2.index(n) for n in names]
    x = np.array([[r[key][i] for i in idx] for r in rows], float) / scale[idx] if scale is not None else None
    return x


def fit(train, names, key, c):
    idx = [FEATURES_V2.index(n) for n in names]
    raw = np.array([[r[key][i] for i in idx] for r in train], float)
    scale = np.maximum(raw.std(0), 1e-6)
    y = np.array([r["y"] for r in train], float)
    model = LogisticRegression(C=c, fit_intercept=False, max_iter=2000).fit(raw / scale, y)
    return model, scale, idx


def proba(fitted, rows, key):
    model, scale, idx = fitted
    x = np.array([[r[key][i] for i in idx] for r in rows], float) / scale
    return model.predict_proba(x)[:, 1]


def scores(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return len(y), float(np.mean((p >= 0.5) == (y == 1))), float(np.mean((p - y) ** 2)), float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def boot(p_new, p_base, y, reps=2000, seed=11):
    rng = random.Random(seed)
    d = (p_base - y) ** 2 - (p_new - y) ** 2
    n = len(d)
    means = sorted(float(np.mean([d[rng.randrange(n)] for _ in range(n)])) for _ in range(reps))
    return float(np.mean(d)), means[int(.025 * reps)], means[int(.975 * reps)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dew", action="store_true")
    args = parser.parse_args()
    data = build_dataset_v2()
    weather_share = np.mean([r["has_weather"] for r in data])
    print(f"matches {len(data)}; with weather {weather_share:.0%}")
    train = [r for r in data if r["date"] < TUNE_FROM]
    tune = [r for r in data if TUNE_FROM <= r["date"] < HOLDOUT_FROM]
    full_train = [r for r in data if r["date"] < HOLDOUT_FROM]
    hold = [r for r in data if r["date"] >= HOLDOUT_FROM]
    y_t = np.array([r["y"] for r in tune], float)
    y_h = np.array([r["y"] for r in hold], float)
    full_mask = np.array([r["a"] in FULL and r["b"] in FULL for r in hold])
    wx_mask = np.array([r["has_weather"] for r in hold])

    chosen = {}
    for name, feats in VARIANTS.items():
        best = None
        for c in (0.03, 0.1, 0.3, 1.0, 3.0):
            s = scores(proba(fit(train, feats, "x", c), tune, "x"), y_t)
            if best is None or s[2] < best[1][2]:
                best = (c, s)
        chosen[name] = best[0]
        print(f"tune  {name:<32} C={best[0]:<5} brier={best[1][2]:.4f} acc={best[1][1]:.3f}")

    probs = {n: proba(fit(full_train, f, "x", chosen[n]), hold, "x") for n, f in VARIANTS.items()}
    base = probs["elo+h2h (baseline)"]
    for label, mask in (("both full members", full_mask), ("all", np.ones(len(hold), bool)),
                        ("rows with weather", wx_mask), ("full members with weather", full_mask & wx_mask)):
        print(f"\nHOLDOUT 2025+  {label}  (n={int(mask.sum())})")
        for n in VARIANTS:
            if mask.sum() == 0:
                continue
            nn, acc, br, ll = scores(probs[n][mask], y_h[mask])
            tail = ""
            if n != "elo+h2h (baseline)":
                g, lo, hi = boot(probs[n][mask], base[mask], y_h[mask])
                tail = f"  gain vs baseline {g:+.4f} [{lo:+.4f},{hi:+.4f}]"
            print(f"  {n:<32} acc={acc:.3f} brier={br:.4f} logloss={ll:.4f}{tail}")




def save_dew_model() -> None:
    """Fit elo+h2h+h2h5+dew on everything before 2025 and save it for live use."""
    data = build_dataset_v2()
    full_train = [r for r in data if r["date"] < HOLDOUT_FROM]
    hold = [r for r in data if r["date"] >= HOLDOUT_FROM]
    names = BASE + ("dew_chase",)
    model, scale, idx = fit(full_train, names, "x", 3.0)
    p = model.predict_proba(np.array([[r["x"][i] for i in idx] for r in hold], float) / scale)[:, 1]
    y = np.array([r["y"] for r in hold], float)
    full = np.array([r["a"] in FULL and r["b"] in FULL for r in hold])
    n1, a1, b1, _ = scores(p[full], y[full])
    n2, a2, b2, _ = scores(p, y)
    DEW_MODEL_PATH.write_text(json.dumps({
        "features": list(names), "scale": [float(v) for v in scale], "weights": [float(w) for w in model.coef_[0]],
        "C": 3.0, "trained_through": full_train[-1]["date"],
        "holdout": {"from": HOLDOUT_FROM,
                    "full_member": {"n": n1, "acc": a1, "brier": b1},
                    "all": {"n": n2, "acc": a2, "brier": b2}},
    }, indent=2))
    print("saved", DEW_MODEL_PATH, "weights", dict(zip(names, np.round(model.coef_[0], 3))))


if __name__ == "__main__":
    if "--save-dew" in __import__("sys").argv:
        save_dew_model()
    else:
        main()
