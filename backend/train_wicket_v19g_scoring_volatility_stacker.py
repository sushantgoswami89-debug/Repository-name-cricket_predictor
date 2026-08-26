"""Part 7: a more direct volatility concept than v19f's chase-equation
gap (which only meant much during chases) -- how erratic recent SCORING
has been (std of runs-per-over, last 5 overs) and how clustered recent
WICKETS have been (count in last 5 overs), both live-available, both
apply to every innings not just chases. User-requested follow-up: "
volatility is exact word... look forward" to it more directly. No
retraining of GBM/NN -- reuses cached v19a/v19b predictions."""
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
grp_key = [data["source_file"], data["innings"]]

# runs_in_over/wicket_in_over describe THIS over -- shift(1) so only PAST
# overs of this same match/innings feed the rolling window (leakage-safe).
past_runs = data["runs_in_over"].groupby(grp_key).shift(1)
past_wkt = data["wicket_in_over"].groupby(grp_key).shift(1)
data["runs_volatility_5"] = past_runs.groupby(grp_key).rolling(5, min_periods=2).std().reset_index(level=[0, 1], drop=True).fillna(0.0)
data["wickets_last_5overs"] = past_wkt.groupby(grp_key).rolling(5, min_periods=1).sum().reset_index(level=[0, 1], drop=True).fillna(0.0)

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

def build_input(gl, nl, vol, wkts):
    return np.column_stack([gl, nl, vol, wkts, gl * vol, nl * vol, gl * wkts, nl * wkts])

cal_vol, cal_wkts = cal["runs_volatility_5"].to_numpy(), cal["wickets_last_5overs"].to_numpy()
holdout_vol, holdout_wkts = holdout["runs_volatility_5"].to_numpy(), holdout["wickets_last_5overs"].to_numpy()
cal_input = build_input(logit(gbm_cal_p), logit(nn_cal_p), cal_vol, cal_wkts)
holdout_input = build_input(logit(gbm_holdout_p), logit(nn_holdout_p), holdout_vol, holdout_wkts)
volstack = LogisticRegression(C=1.0, random_state=42, max_iter=2000).fit(cal_input, cal_y)
vol_holdout = volstack.predict_proba(holdout_input)[:, 1]

def metrics(y, p, mask=None):
    if mask is not None:
        y, p = y[mask], p[mask]
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)), "rows": int(len(y))}

def split(p):
    return {"blended": metrics(holdout_y, p), "ipl": metrics(holdout_y, p, holdout_is_ipl), "t20i": metrics(holdout_y, p, ~holdout_is_ipl)}

report = {
    "candidate_version": "wicket_v19g_scoring_volatility_stacker",
    "flat_stacker": split(flat_holdout),
    "scoring_volatility_wicket_clustering_stacker": split(vol_holdout),
    "stacker_coefficients": {
        "gbm": float(volstack.coef_[0][0]), "nn": float(volstack.coef_[0][1]),
        "runs_volatility": float(volstack.coef_[0][2]), "wickets_last_5": float(volstack.coef_[0][3]),
        "gbm_x_volatility": float(volstack.coef_[0][4]), "nn_x_volatility": float(volstack.coef_[0][5]),
        "gbm_x_wickets5": float(volstack.coef_[0][6]), "nn_x_wickets5": float(volstack.coef_[0][7]),
    },
}
(output / "scoring_volatility_stacker_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
