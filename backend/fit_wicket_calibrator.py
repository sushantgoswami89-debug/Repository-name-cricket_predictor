"""Fit per-phase isotonic-regression calibrators for the ACTUAL production
wicket model's (models/wkt_model.pkl) output probabilities.

Recovers that model's own original held-out test split (src/train.py uses
train_test_split(match_ids, test_size=0.2, random_state=42) against
data/real_overs.csv -- deterministic and reproducible as long as the CSV is
unchanged), then further splits that 20% into a calibration half and a
final honest-eval half, so calibrators are fit and validated on the real
deployed model's own genuinely-out-of-sample predictions, not a retrained
clone's.

Saves models/wkt_calibrator.pkl as {phase: IsotonicRegression}.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "backend"))
from features import build_features, get_feature_columns  # noqa: E402
from app.ml.model_repository import ModelRepository  # noqa: E402

df = pd.read_csv(PROJECT_ROOT / "data" / "real_overs.csv")
df = build_features(df)
numeric_cols, cat_cols = get_feature_columns()
for c in cat_cols:
    df[c] = df[c].astype("category")
feature_cols = numeric_cols + cat_cols

# Recover the production model's own original held-out split.
match_ids = df["match_id"].unique()
_train_ids, held_out_ids = train_test_split(match_ids, test_size=0.2, random_state=42)
calib_ids, test_ids = train_test_split(held_out_ids, test_size=0.5, random_state=42)

calib_mask = df["match_id"].isin(calib_ids)
test_mask = df["match_id"].isin(test_ids)

repository = ModelRepository()
wkt_model = repository.get_wicket_model()

X = df[feature_cols]
calib_df = df[calib_mask].copy()
calib_df["raw_pred"] = wkt_model.predict_proba(X[calib_mask])[:, 1]
test_df = df[test_mask].copy()
test_df["raw_pred"] = wkt_model.predict_proba(X[test_mask])[:, 1]

print(f"Calibrate: {calib_mask.sum()} rows | Final honest test: {test_mask.sum()} rows\n")

calibrators = {}
print(f"{'Phase':10} | {'CalibN':>7} | {'TestN':>6} | {'AUC(test)':>9} | {'Brier before':>12} | {'Brier after':>11} | {'Naive':>7}")
for phase in ("powerplay", "middle", "death"):
    calib_sub = calib_df[calib_df["phase"] == phase]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(calib_sub["raw_pred"], calib_sub["wicket_in_over"])
    calibrators[phase] = calibrator

    test_sub = test_df[test_df["phase"] == phase]
    calibrated = calibrator.predict(test_sub["raw_pred"])
    auc = roc_auc_score(test_sub["wicket_in_over"], test_sub["raw_pred"])
    brier_before = brier_score_loss(test_sub["wicket_in_over"], test_sub["raw_pred"])
    brier_after = brier_score_loss(test_sub["wicket_in_over"], calibrated)
    base_rate = test_sub["wicket_in_over"].mean()
    naive = base_rate * (1 - base_rate)
    print(
        f"{phase:10} | {len(calib_sub):7d} | {len(test_sub):6d} | {auc:9.4f} | "
        f"{brier_before:12.4f} | {brier_after:11.4f} | {naive:7.4f}"
    )

out_path = PROJECT_ROOT / "models" / "wkt_calibrator.pkl"
joblib.dump(calibrators, out_path)
print(f"\nSaved phase-keyed calibrators (fit on the real production model) to {out_path}")
