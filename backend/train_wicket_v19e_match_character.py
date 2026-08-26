"""Part 5: does GBM vs NN's relative strength (for wicket_in_over) differ
by match CHARACTER -- blowout, close/nail-biting finish, or a "swing"
match (close at over 10, decisively not close by the end)? User-requested
2026-08-25. Reuses the cached predictions from v19a/v19b (no retraining)
-- just re-evaluates them within each match-character bucket.

Categories, derived from real Cricsheet outcomes + our own over-level
state (no modeling, just real data):
  - close: final margin <15 runs (setting-team win) or <=2 wickets in
    hand (chasing-team win) -- genuinely tight finish.
  - blowout: final margin >=40 runs or >=7 wickets in hand -- comfortable.
  - swing: a CHASE that was close at over 10 (|current_run_rate -
    required_run_rate| <= 1.5) but ended up a blowout by the end -- i.e.
    was even, then swung decisively.
  - moderate: everything else (not neatly one of the above).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from app.ml.cascade_features import build_wicket_base_dataset
from train_phase_calibrated_sharp_range_v33 import male_source_files

root = Path(__file__).resolve().parents[1]
output = root / "models/candidates/wicket_v19_nn_gbm_ensemble"
g = np.load(output / "gbm_only.npz")
m = np.load(output / "nn_only.npz")
cal_y, holdout_y = g["cal_y"], g["holdout_y"]

data = build_wicket_base_dataset(root)
cal = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)
assert len(cal) == len(cal_y) and len(holdout) == len(holdout_y)

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def platt(cal_p, target_p):
    fit = LogisticRegression(C=1.0, random_state=42).fit(logit(cal_p).reshape(-1, 1), cal_y)
    return fit.predict_proba(logit(target_p).reshape(-1, 1))[:, 1]

gbm_holdout_p = platt(g["gbm_cal"], g["gbm_holdout"])
nn_holdout_p = platt(m["nn_cal"], m["nn_holdout"])
flat = LogisticRegression(C=1.0, random_state=42).fit(
    np.column_stack([logit(platt(g["gbm_cal"], g["gbm_cal"])), logit(platt(m["nn_cal"], m["nn_cal"]))]), cal_y)
ensemble_holdout = flat.predict_proba(np.column_stack([logit(gbm_holdout_p), logit(nn_holdout_p)]))[:, 1]

# ---- Real match-outcome margins from raw Cricsheet, holdout matches only ----
print("Deriving real match-character categories from Cricsheet outcomes...", flush=True)
holdout_files = set(holdout["source_file"].unique())
eligible = male_source_files(root)
margin_kind: dict[str, str] = {}   # match_id -> "close" / "blowout" / "moderate"
for scope in ("ipl", "t20i"):
    for path in (root / "data/raw/cricsheet" / scope).glob("*.json"):
        if path.name not in holdout_files or path.name not in eligible:
            continue
        info = json.loads(path.read_text(encoding="utf-8"))["info"]
        outcome = info.get("outcome", {})
        by = outcome.get("by", {})
        if "runs" in by:
            margin_kind[path.name] = "close" if by["runs"] < 15 else ("blowout" if by["runs"] >= 40 else "moderate")
        elif "wickets" in by:
            margin_kind[path.name] = "close" if by["wickets"] <= 2 else ("blowout" if by["wickets"] >= 7 else "moderate")
        else:
            margin_kind[path.name] = "moderate"  # tie/no-result/super-over edge cases

# ---- "Swing" chases: close at over 10, blowout by the end ----
over10 = holdout[(holdout["innings"] == 2) & (holdout["over"] == 11)]
close_at_10 = set(over10.loc[(over10["current_run_rate"] - over10["required_run_rate"]).abs() <= 1.5, "source_file"])
swing_matches = {f for f in close_at_10 if margin_kind.get(f) == "blowout"}

category = pd.Series("moderate", index=holdout.index)
category[holdout["source_file"].map(margin_kind) == "close"] = "close"
category[holdout["source_file"].map(margin_kind) == "blowout"] = "blowout"
category[holdout["source_file"].isin(swing_matches)] = "swing"
category = category.to_numpy()

def metrics(y, p, mask):
    if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
        return {"rows": int(mask.sum()), "note": "too few rows / one-class"}
    return {"rows": int(mask.sum()), "auc": float(roc_auc_score(y[mask], p[mask])), "brier": float(brier_score_loss(y[mask], p[mask]))}

report = {}
for cat in ("close", "blowout", "swing", "moderate"):
    mask = category == cat
    report[cat] = {
        "matches": int(len(set(holdout.loc[mask, "source_file"]))),
        "gbm": metrics(holdout_y, gbm_holdout_p, mask),
        "nn": metrics(holdout_y, nn_holdout_p, mask),
        "ensemble": metrics(holdout_y, ensemble_holdout, mask),
    }
(output / "match_character_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
