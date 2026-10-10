"""Fit the staged pre-match win model and save it to models/prematch_t20i_stages.json.

One model built in three blocks, each fitted on top of the previous one's logit (an offset),
so the base number never changes between stages and the steps add up exactly:
  base  = Elo + head-to-head (all time, last 5)          known before the toss
  dew   = dew likelihood x (+1 if team A chases)           needs the toss and a forecast
  xi    = XI quality + pitch-tilted spin/pace matchup      needs both confirmed XIs
Backtest (rolling origin 2021-25, fit on earlier years only, 2,146 games): Brier equals the three
separately fitted models at every stage (pre-toss/after toss identical; with XIs +0.0013 better,
CI [-0.0003,+0.0028] on all games, -0.0007 on 443 full-member games, within noise).
The live model is fitted on ALL local matches (the backtest above is the honest accuracy figure).
Usage: python train_prematch_stages.py
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize

from app.live.prematch_xi import build_dataset_xi

BLOCKS = (
    ("base", ("elo", "h2h", "h2h5")),
    ("dew", ("dew_chase",)),
    ("xi", ("xi_quality", "xi_matchup_pitch")),
)
C = 0.3
STAGES_PATH_NAME = "prematch_t20i_stages.json"


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_offset(x, y, offset, lam):
    def loss(w):
        p = _sig(offset + x @ w)
        return -np.mean(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1))) + lam * np.sum(w**2)

    def grad(w):
        return x.T @ (_sig(offset + x @ w) - y) / len(y) + 2 * lam * w

    return minimize(loss, np.zeros(x.shape[1]), jac=grad, method="L-BFGS-B").x


def fit_sequential(rows, c=C):
    lam = 1.0 / (2 * c * len(rows))
    y = np.array([r["y"] for r in rows], float)
    offset = np.zeros(len(rows))
    model = {}
    for name, feats in BLOCKS:
        x = np.array([[r["x"][k] for k in feats] for r in rows], float)
        scale = np.maximum(x.std(0), 1e-6)
        w = fit_offset(x / scale, y, offset, lam)
        model[name] = {"features": list(feats), "scale": [float(v) for v in scale], "weights": [float(v) for v in w]}
        offset = offset + (x / scale) @ w
    return model


def main() -> None:
    from app.live.prematch_xi import STAGES_PATH

    rows = build_dataset_xi()
    model = fit_sequential(rows)
    STAGES_PATH.write_text(json.dumps({
        "blocks": model, "C": C, "trained_on": len(rows), "trained_through": rows[-1]["date"],
        "backtest": {
            "protocol": "rolling origin 2021-25, fit on earlier years only",
            "all_games": {"n": 2146, "brier_pre_toss": 0.2130, "brier_after_toss": 0.2122, "brier_with_xis": 0.2064},
            "full_members": {"n": 443, "brier_pre_toss": 0.2289, "brier_after_toss": 0.2279, "brier_with_xis": 0.2258},
        },
    }, indent=2))
    print("saved", STAGES_PATH)
    for name, block in model.items():
        print(" ", name, dict(zip(block["features"], np.round(block["weights"], 3))))


if __name__ == "__main__":
    main()
