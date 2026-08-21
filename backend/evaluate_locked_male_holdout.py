"""Evaluate the locked runs model on the male-only 2025+ holdout."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_phase_calibrated_sharp_range_v33 import male_source_files


root = Path(__file__).resolve().parents[1]
data = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
data["match_date"] = pd.to_datetime(data["match_date"])
data = data[data["source_file"].isin(male_source_files(root))]
holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)

locked_dir = root / "models/locked/cricketbaba_candidate_v3"
feature_cols = joblib.load(locked_dir / "feature_cols.pkl")
cat_cols = joblib.load(locked_dir / "cat_cols.pkl")
x_holdout = holdout[feature_cols].copy()
for column in cat_cols:
    x_holdout[column] = x_holdout[column].astype("category")

point_prediction = joblib.load(locked_dir / "runs_model.pkl").predict(x_holdout)
actual = holdout["runs_in_over"].to_numpy()
low = np.maximum(0, np.rint(point_prediction - 1).astype(int))
high = low + 2
hit_rate = np.mean((actual >= low) & (actual <= high))

print(f"male_only_holdout_overs={len(holdout)}")
print(f"locked_model_hit_rate={hit_rate:.6f}")
