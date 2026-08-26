"""Part 6: does a LIVE-available momentum/volatility signal (not the
retrospective match-outcome labels from v19e) predict when to trust GBM
vs NN? User-requested 2026-08-25 follow-up to the close/blowout/swing
finding -- that used the final result, which isn't known live. This uses
only real rolling history within the same match, available at prediction
time: how much the (current_run_rate - required_run_rate) gap has been
swinging over the last 5 overs (volatility) and its direction (momentum).
No retraining of GBM/NN -- reuses cached v19a/v19b predictions."""
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

data = build_wicket_base_dataset(root)
data = data.sort_values(["source_file", "innings", "over"]).reset_index(drop=True)
gap = data["current_run_rate"] - data["required_run_rate"]
grp = data.groupby(["source_file", "innings"])["over"]  # just to align groupby object
gap_by_group = gap.groupby([data["source_file"], data["innings"]])
# shift(1) excludes the CURRENT over -- only past overs of this same match/innings feed the rolling window (leakage-safe)
past_gap = gap_by_group.shift(1)
data["gap_volatility_5"] = past_gap.groupby([data["source_file"], data["innings"]]).rolling(5, min_periods=2).std().reset_index(level=[0, 1], drop=True).fillna(0.0)
gap_5ago = gap_by_group.shift(6)
data["gap_momentum_5"] = (past_gap.groupby([data["source_file"], data["innings"]]).shift(0) - gap_5ago).fillna(0.0)

cal = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)
assert len(cal) == len(cal_y) and len(holdout) == len(holdout_y), "row-order mismatch"

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def platt(cal_p, target_p):
    fit = LogisticRegression(C=1.0, random_state=42).fit(logit(cal_p).reshape(-1, 1), cal_y)
    return fit.predict_proba(logit(target_p).reshape(-1, 1))[:, 1]

gbm_cal_p, gbm_holdout_p = platt(g["gbm_cal"], g["gbm_cal"]), platt(g["gbm_cal"], g["gbm_holdout"])
nn_cal_p, nn_holdout_p = platt(m["nn_cal"], m["nn_cal"]), platt(m["nn_cal"], m["nn_holdout"])

flat = LogisticRegression(C=1.0, random_state=42).fit(np.column_stack([logit(gbm_cal_p), logit(nn_cal_p)]), cal_y)
flat_holdout = flat.predict_proba(np.column_stack([logit(gbm_holdout_p), logit(nn_holdout_p)]))[:, 1]

def vol_input(gl, nl, vol, mom):
    return np.column_stack([gl, nl, vol, mom, gl * vol, nl * vol])

cal_vol, cal_mom = cal["gap_volatility_5"].to_numpy(), cal["gap_momentum_5"].to_numpy()
holdout_vol, holdout_mom = holdout["gap_volatility_5"].to_numpy(), holdout["gap_momentum_5"].to_numpy()
cal_input = vol_input(logit(gbm_cal_p), logit(nn_cal_p), cal_vol, cal_mom)
holdout_input = vol_input(logit(gbm_holdout_p), logit(nn_holdout_p), holdout_vol, holdout_mom)
volstack = LogisticRegression(C=1.0, random_state=42, max_iter=2000).fit(cal_input, cal_y)
vol_holdout = volstack.predict_proba(holdout_input)[:, 1]

def metrics(y, p, mask=None):
    if mask is not None:
        y, p = y[mask], p[mask]
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)), "rows": int(len(y))}

def split(p):
    return {"blended": metrics(holdout_y, p), "ipl": metrics(holdout_y, p, holdout_is_ipl), "t20i": metrics(holdout_y, p, ~holdout_is_ipl)}

report = {
    "candidate_version": "wicket_v19f_live_volatility_stacker",
    "flat_stacker": split(flat_holdout),
    "live_volatility_momentum_stacker": split(vol_holdout),
    "stacker_coefficients": {"gbm": float(volstack.coef_[0][0]), "nn": float(volstack.coef_[0][1]),
                              "volatility": float(volstack.coef_[0][2]), "momentum": float(volstack.coef_[0][3]),
                              "gbm_x_volatility": float(volstack.coef_[0][4]), "nn_x_volatility": float(volstack.coef_[0][5])},
}
(output / "live_volatility_stacker_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
