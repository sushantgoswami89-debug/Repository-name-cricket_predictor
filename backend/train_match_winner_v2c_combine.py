"""Part 3/3: combine the GBM-only (v2a) and NN-only (v2b) match_winner
outputs into a Platt-calibrated 2-input logistic-regression stacker --
exact same combination method as the proven wicket ensemble
(train_wicket_v19c_combine.py). Win condition: does the ensemble beat
the GBM alone (which reproduces live match_winner_v1's own
bowler_known_ceiling numbers) on the real 2025+ holdout."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

VERSION = "match_winner_v2_nn_gbm_ensemble"
root = Path(__file__).resolve().parents[1]
output = root / "models/candidates" / VERSION
g = np.load(output / "gbm_only.npz", allow_pickle=True)
m = np.load(output / "nn_only.npz", allow_pickle=True)
cal_y, holdout_y = g["cal_actual"], g["holdout_actual"]
holdout_is_ipl = g["holdout_is_ipl"].astype(bool)


def platt(cal_p, cal_y, target_p):
    lc = np.clip(cal_p, 1e-6, 1 - 1e-6)
    logit = np.log(lc / (1 - lc)).reshape(-1, 1)
    fit = LogisticRegression(C=1.0, random_state=42).fit(logit, cal_y)
    lt = np.clip(target_p, 1e-6, 1 - 1e-6)
    return fit.predict_proba(np.log(lt / (1 - lt)).reshape(-1, 1))[:, 1]


gbm_cal_platt = platt(g["gbm_cal"], cal_y, g["gbm_cal"])
gbm_holdout_platt = platt(g["gbm_cal"], cal_y, g["gbm_holdout"])
nn_cal_platt = platt(m["nn_cal"], cal_y, m["nn_cal"])
nn_holdout_platt = platt(m["nn_cal"], cal_y, m["nn_holdout"])


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


stacker = LogisticRegression(C=1.0, random_state=42).fit(
    np.column_stack([logit(gbm_cal_platt), logit(nn_cal_platt)]), cal_y)
ensemble_holdout = stacker.predict_proba(np.column_stack([logit(gbm_holdout_platt), logit(nn_holdout_platt)]))[:, 1]


def metrics(y, p, mask=None):
    if mask is not None:
        y, p = y[mask], p[mask]
    return {"rows": int(len(y)), "auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p))}


def split(p):
    return {"blended": metrics(holdout_y, p), "ipl": metrics(holdout_y, p, holdout_is_ipl), "t20i": metrics(holdout_y, p, ~holdout_is_ipl)}


gbm_alone = split(gbm_holdout_platt)
nn_alone = split(nn_holdout_platt)
ensemble = split(ensemble_holdout)

reference_v1 = {
    "blended": {"auc": 0.8587, "brier": 0.1538},
    "ipl": {"auc": 0.7889, "brier": 0.1905},
    "t20i": {"auc": 0.8703, "brier": 0.1472},
}
beats_v1 = bool(
    ensemble["blended"]["auc"] > reference_v1["blended"]["auc"]
    and ensemble["blended"]["brier"] <= reference_v1["blended"]["brier"]
)
beats_v1_ipl = bool(ensemble["ipl"]["auc"] > reference_v1["ipl"]["auc"])
beats_v1_t20i = bool(ensemble["t20i"]["auc"] > reference_v1["t20i"]["auc"])

report = {
    "candidate_version": VERSION,
    "candidate_only": True,
    "production_changed": False,
    "note": (
        "Architecture test, not a new feature: does a GBM+NN ensemble "
        "(the same idea that gave real wins on wicket and run-range) "
        "beat the GBM alone for win-probability, using the exact same "
        "feature set as the live match_winner_v1 bowler-known cut?"
    ),
    "reference_match_winner_v1_bowler_known_ceiling": reference_v1,
    "gbm_alone_holdout": gbm_alone,
    "nn_alone_holdout": nn_alone,
    "ensemble_holdout": ensemble,
    "beats_v1_blended": beats_v1,
    "beats_v1_ipl": beats_v1_ipl,
    "beats_v1_t20i": beats_v1_t20i,
    "decision": "promote_candidate" if beats_v1 else "reject_keep_research",
}
(output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
