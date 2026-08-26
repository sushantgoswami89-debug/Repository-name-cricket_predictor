"""Part 4: does letting the GBM/NN blend weight vary by SITUATION (phase,
wickets in hand, IPL vs T20I) beat one fixed global weight? User-requested
2026-08-25: "it should not be hard coded... situation specific... take
weightage depending on those values." No lightgbm/torch needed here --
just reloads the cached predictions from v19a/v19b and refits the
stacker with context features added. Same holdout, same IPL/T20I split."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from app.ml.cascade_features import build_wicket_base_dataset

root = Path(__file__).resolve().parents[1]
output = root / "models/candidates/wicket_v19_nn_gbm_ensemble"
g = np.load(output / "gbm_only.npz")
m = np.load(output / "nn_only.npz")
cal_y, holdout_y, holdout_is_ipl = g["cal_y"], g["holdout_y"], g["holdout_is_ipl"].astype(bool)

print("Rebuilding dataset for context columns (phase, wickets_in_hand)...", flush=True)
data = build_wicket_base_dataset(root)
cal = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)
assert len(cal) == len(cal_y) and len(holdout) == len(holdout_y), "row-order mismatch vs cached predictions"

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def platt(cal_p, target_p):
    fit = LogisticRegression(C=1.0, random_state=42).fit(logit(cal_p).reshape(-1, 1), cal_y)
    return fit.predict_proba(logit(target_p).reshape(-1, 1))[:, 1]

gbm_cal_p, gbm_holdout_p = platt(g["gbm_cal"], g["gbm_cal"]), platt(g["gbm_cal"], g["gbm_holdout"])
nn_cal_p, nn_holdout_p = platt(m["nn_cal"], m["nn_cal"]), platt(m["nn_cal"], m["nn_holdout"])

def context(d):
    phase_dummies = np.stack([(d["phase"].astype(str) == p).to_numpy(dtype=float) for p in ("powerplay", "middle", "death")], axis=1)
    wkts = (d["wickets_in_hand"].to_numpy(dtype=float) - 5) / 3
    ipl = d["is_ipl"].to_numpy(dtype=float)
    return phase_dummies, wkts, ipl

cal_phase, cal_wkts, cal_ipl = context(cal)
holdout_phase, holdout_wkts, holdout_ipl = context(holdout)

# Flat (reference, same as v19c): [gbm_logit, nn_logit] only
flat = LogisticRegression(C=1.0, random_state=42).fit(np.column_stack([logit(gbm_cal_p), logit(nn_cal_p)]), cal_y)
flat_holdout = flat.predict_proba(np.column_stack([logit(gbm_holdout_p), logit(nn_holdout_p)]))[:, 1]

# Situational: add phase, wickets_in_hand, is_ipl, AND their interactions
# with each model's logit (so the LEARNED weight on each model can shift
# by situation, not just the intercept) -- this is what actually makes it
# "situation-specific," not just adding context as flat extra inputs.
def situational_input(gbm_logit, nn_logit, phase, wkts, ipl):
    return np.column_stack([
        gbm_logit, nn_logit, phase, wkts, ipl,
        gbm_logit[:, None].squeeze() * phase[:, 0], gbm_logit[:, None].squeeze() * wkts, gbm_logit[:, None].squeeze() * ipl,
        nn_logit[:, None].squeeze() * phase[:, 0], nn_logit[:, None].squeeze() * wkts, nn_logit[:, None].squeeze() * ipl,
    ])

cal_input = situational_input(logit(gbm_cal_p), logit(nn_cal_p), cal_phase, cal_wkts, cal_ipl)
holdout_input = situational_input(logit(gbm_holdout_p), logit(nn_holdout_p), holdout_phase, holdout_wkts, holdout_ipl)
situational = LogisticRegression(C=1.0, random_state=42, max_iter=2000).fit(cal_input, cal_y)
situational_holdout = situational.predict_proba(holdout_input)[:, 1]

def metrics(y, p, mask=None):
    if mask is not None:
        y, p = y[mask], p[mask]
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)), "rows": int(len(y))}

def split(p):
    return {"blended": metrics(holdout_y, p), "ipl": metrics(holdout_y, p, holdout_is_ipl), "t20i": metrics(holdout_y, p, ~holdout_is_ipl)}

report = {
    "candidate_version": "wicket_v19d_situational_stacker",
    "flat_stacker_gbm_nn_only": split(flat_holdout),
    "situational_stacker_context_aware": split(situational_holdout),
}
(output / "situational_stacker_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
