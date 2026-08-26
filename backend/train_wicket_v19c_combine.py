"""Part 3/3: combine the GBM-only (train_wicket_v19a) and NN-only
(train_wicket_v19b) outputs into a Platt-calibrated 2-input logistic-
regression stacker (ensemble). Win condition: does the ensemble beat the
GBM alone on the real 2025+ holdout -- not whether the NN alone wins.
See train_wicket_v19a_gbm_only.py's docstring for why this is 3 separate
scripts (torch/lightgbm process-coexistence issue on this machine)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

root = Path(__file__).resolve().parents[1]
output = root / "models/candidates/wicket_v19_nn_gbm_ensemble"
g = np.load(output / "gbm_only.npz")
m = np.load(output / "nn_only.npz")
cal_y, holdout_y, holdout_is_ipl = g["cal_y"], g["holdout_y"], g["holdout_is_ipl"].astype(bool)

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
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)), "rows": int(len(y))}

def split(p):
    return {"blended": metrics(holdout_y, p), "ipl": metrics(holdout_y, p, holdout_is_ipl), "t20i": metrics(holdout_y, p, ~holdout_is_ipl)}

report = {
    "candidate_version": "wicket_v19_nn_gbm_ensemble",
    "reference_v16": {"auc_known_blended": 0.6126, "auc_known_ipl": 0.6151, "auc_known_t20i": 0.6113},
    "gbm_alone": split(gbm_holdout_platt),
    "nn_alone": split(nn_holdout_platt),
    "ensemble": split(ensemble_holdout),
}
(output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
