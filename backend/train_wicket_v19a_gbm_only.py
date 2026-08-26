"""Part 1/3 of the NN+GBM ensemble test (see train_wicket_v19c_combine.py
for the full rationale). Split into separate processes because torch and
lightgbm cannot coexist in one process on this machine (confirmed: hangs/
silently dies regardless of import order or KMP_DUPLICATE_LIB_OK). This
script: lightgbm only, no torch."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import lightgbm as lgb
from app.ml.cascade_features import WICKET_CATEGORICAL, WICKET_FEATURES, build_wicket_base_dataset

CAT = WICKET_CATEGORICAL
root = Path(__file__).resolve().parents[1]
output = root / "models/candidates/wicket_v19_nn_gbm_ensemble"
output.mkdir(parents=True, exist_ok=True)

print("Building dataset...", flush=True)
data = build_wicket_base_dataset(root)
train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
cal = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)

def frame(d):
    r = d[WICKET_FEATURES].copy()
    for c in CAT:
        r[c] = r[c].fillna("__UNKNOWN__").astype("category")
    return r

print("Training GBM...", flush=True)
gbm = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.03, max_depth=5,
                          num_leaves=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
gbm.fit(frame(train), train["wicket_in_over"], categorical_feature=CAT)
gbm_cal = gbm.predict_proba(frame(cal))[:, 1]
gbm_holdout = gbm.predict_proba(frame(holdout))[:, 1]

np.savez(output / "gbm_only.npz",
          gbm_cal=gbm_cal, gbm_holdout=gbm_holdout,
          cal_y=cal["wicket_in_over"].to_numpy(), holdout_y=holdout["wicket_in_over"].to_numpy(),
          holdout_is_ipl=holdout["is_ipl"].to_numpy())
print("Saved gbm_only.npz")
