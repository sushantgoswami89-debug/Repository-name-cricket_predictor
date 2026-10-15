"""Part 1/3 of a GBM+NN ensemble test for match_winner_v1 -- the proven
architecture lever from wicket (finding_wicket_nn_gbm_ensemble_real_win.md)
and run-range (finding_run_range_nn_gbm_ensemble_real_win.md), never
tried on win-probability. Same exact feature set as the live
match_winner_v1 (bowler-known cut) -- this tests architecture, not new
inputs. Split into separate processes because torch and lightgbm cannot
coexist in one process on this machine. This script: lightgbm only, no
torch. Also caches the built feature dataframe to pickle so parts 2 and 3
don't rebuild the ~15 dataset joins from scratch."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from train_match_winner_v1 import CATEGORICAL, DEFAULT_LGBM_PARAMS, FEATURES, build_dataset

VERSION = "match_winner_v2_nn_gbm_ensemble"
root = Path(__file__).resolve().parents[1]
output = root / "models/candidates" / VERSION
output.mkdir(parents=True, exist_ok=True)

print("Building match_winner_v1 dataset (reused, not rebuilt)...", flush=True)
data = build_dataset(root)

train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)


def frame(d):
    r = d[FEATURES].copy()
    for c in CATEGORICAL:
        r[c] = r[c].fillna("__UNKNOWN__").astype("category")
    return r


cache_columns = list(dict.fromkeys(FEATURES + CATEGORICAL + ["batting_team_won", "phase", "is_ipl"]))
train[cache_columns].to_pickle(output / "train_cache.pkl")
calibration[cache_columns].to_pickle(output / "calibration_cache.pkl")
holdout[cache_columns].to_pickle(output / "holdout_cache.pkl")
print("Cached train/calibration/holdout feature frames.")

print("Training GBM (bowler-known cut, matches live match_winner_v1's ceiling report)...", flush=True)
model = lgb.LGBMClassifier(**DEFAULT_LGBM_PARAMS)
model.fit(frame(train), train["batting_team_won"], categorical_feature=CATEGORICAL)

gbm_cal = model.predict_proba(frame(calibration))[:, 1]
gbm_holdout = model.predict_proba(frame(holdout))[:, 1]

np.savez(
    output / "gbm_only.npz",
    gbm_cal=gbm_cal, gbm_holdout=gbm_holdout,
    cal_actual=calibration["batting_team_won"].to_numpy(),
    holdout_actual=holdout["batting_team_won"].to_numpy(),
    holdout_is_ipl=holdout["is_ipl"].to_numpy(),
    holdout_phase=holdout["phase"].astype(str).to_numpy(),
)
print("Saved gbm_only.npz")
print(json.dumps({"train_rows": len(train), "cal_rows": len(calibration), "holdout_rows": len(holdout)}))
