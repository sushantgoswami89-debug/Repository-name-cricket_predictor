"""Backtest and fit the ODI pre-match base model (Elo + head-to-head); saves models/prematch_odi_base.json.

Rolling origin: for each test year, fit on strictly earlier years and score that year. Compared with a coin
(Brier 0.2500) and with plain Elo. The saved model is fitted on ALL local ODIs; the backtest is the honest
accuracy figure. Usage: python train_prematch_odi.py [--grid]
"""

from __future__ import annotations

import json
import sys

import numpy as np

from app.live import prematch_model as pm
from app.live.prematch_odi import MODEL_PATH, build_dataset
from train_prematch_stages import C, fit_offset

FEATS = ("elo", "h2h", "h2h5")
FULL_MEMBERS = {
    "India", "Australia", "England", "South Africa", "New Zealand", "Pakistan", "Sri Lanka",
    "Bangladesh", "West Indies", "Afghanistan", "Zimbabwe", "Ireland",
}
TEST_YEARS = range(2019, 2026)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit(rows):
    lam = 1.0 / (2 * C * len(rows))
    y = np.array([r["y"] for r in rows], float)
    x = np.array([[r["x"][k] for k in FEATS] for r in rows], float)
    scale = np.maximum(x.std(0), 1e-6)
    w = fit_offset(x / scale, y, np.zeros(len(rows)), lam)
    return {"features": list(FEATS), "scale": [float(v) for v in scale], "weights": [float(v) for v in w]}


def predict(block, rows):
    x = np.array([[r["x"][k] for k in FEATS] for r in rows], float)
    return _sig((x / np.array(block["scale"])) @ np.array(block["weights"]))


def backtest(rows, label=""):
    ps, ys, elo, full = [], [], [], []
    for year in TEST_YEARS:
        train = [r for r in rows if int(r["date"][:4]) < year]
        test = [r for r in rows if int(r["date"][:4]) == year]
        if len(train) < 300 or not test:
            continue
        p = predict(fit(train), test)
        ps.extend(p)
        ys.extend(r["y"] for r in test)
        elo.extend(r["elo_p"] for r in test)
        full.extend(r["a"] in FULL_MEMBERS and r["b"] in FULL_MEMBERS for r in test)
    ps, ys, elo, full = map(np.array, (ps, ys, elo, full))
    out = {}
    for name, mask in (("all", np.ones(len(ys), bool)), ("full_members", full)):
        b = float(np.mean((ps[mask] - ys[mask]) ** 2))
        be = float(np.mean((elo[mask] - ys[mask]) ** 2))
        acc = float(np.mean((ps[mask] >= 0.5) == (ys[mask] == 1)))
        out[name] = {"n": int(mask.sum()), "brier": round(b, 4), "brier_plain_elo": round(be, 4), "accuracy": round(acc, 3)}
    # calibration: predicted-favourite probability bins
    fav = np.maximum(ps, 1 - ps)
    won = ((ps >= 0.5) == (ys == 1))
    bins = []
    for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)):
        m = (fav >= lo) & (fav < hi)
        if m.sum():
            bins.append({"said": f"{lo:.0%}-{min(hi, 1):.0%}", "n": int(m.sum()), "won": round(float(won[m].mean()), 3)})
    out["calibration"] = bins
    print(label, json.dumps(out))
    return out


def main() -> None:
    if "--grid" in sys.argv:
        for kmin, kmax, games in ((16, 96, 30), (16, 64, 30), (24, 64, 20), (12, 48, 40), (32, 32, 1), (20, 80, 25)):
            pm.ELO_K_MIN, pm.ELO_K_MAX, pm.ELO_K_GAMES = float(kmin), float(kmax), float(games)
            rows = build_dataset()
            backtest(rows, f"K {kmin}-{kmax}/{games}:")
        return
    rows = build_dataset()
    result = backtest(rows, "default:")
    model = fit(rows)
    MODEL_PATH.write_text(json.dumps({
        "base": model, "C": C, "trained_on": len(rows), "trained_through": rows[-1]["date"],
        "backtest": {"protocol": f"rolling origin {TEST_YEARS[0]}-{TEST_YEARS[-1]}, fit on earlier years only", **result},
    }, indent=2))
    print("saved", MODEL_PATH, dict(zip(FEATS, np.round(model["weights"], 3))))


if __name__ == "__main__":
    main()
