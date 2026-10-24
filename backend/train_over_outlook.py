"""Train the over outlook (user 2026-10-24): the MEAN runs of the next over, the chance of a big over (10+ runs) and of a quiet one (4 or fewer),
and the chance of at least one wicket. Same v3 features and data as the live GBM; L2 (mean) objective instead of L1 (median).

Writes models/over_outlook_v1/ (4 models + OUTLOOK_MANIFEST.json with the holdout numbers). The holdout numbers come from a fit on <= 2023
tested on 2025+; the shipped models are fitted on every row. Usage: python train_over_outlook.py
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from train_validate_candidate_v3 import V3_FEATURES, CATEGORICAL_FEATURES, _features

root = Path(__file__).resolve().parents[1]
OUT = root / "models" / "over_outlook_v1"
BIG, QUIET = 10, 4
KW = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, max_depth=7, min_child_samples=80, subsample=0.85, colsample_bytree=0.9,
          random_state=42, verbose=-1)


def fit(frame: pd.DataFrame) -> dict:
    X = _features(frame, V3_FEATURES)
    return {
        "mean": lgb.LGBMRegressor(objective="regression", **KW).fit(X, frame["runs_in_over"], categorical_feature=CATEGORICAL_FEATURES),
        "big": lgb.LGBMClassifier(**KW).fit(X, (frame["runs_in_over"] >= BIG).astype(int), categorical_feature=CATEGORICAL_FEATURES),
        "quiet": lgb.LGBMClassifier(**KW).fit(X, (frame["runs_in_over"] <= QUIET).astype(int), categorical_feature=CATEGORICAL_FEATURES),
        "wicket": lgb.LGBMClassifier(**KW).fit(X, frame["wicket_in_over"], categorical_feature=CATEGORICAL_FEATURES),
    }


def main() -> None:
    frame = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    year = pd.to_datetime(frame["match_date"]).dt.year
    train, hold = frame[year <= 2023], frame[year >= 2025]
    models = fit(train)
    X = _features(hold, V3_FEATURES)
    actual = hold["runs_in_over"].to_numpy()
    mean = np.clip(models["mean"].predict(X), 0, None)
    report = {"holdout_rows": int(len(hold)), "mean_bias": round(float((mean - actual).mean()), 3), "mean_mae": round(float(np.abs(mean - actual).mean()), 3)}
    for name, target in (("big", actual >= BIG), ("quiet", actual <= QUIET), ("wicket", hold["wicket_in_over"].to_numpy() == 1)):
        p = models[name].predict_proba(X)[:, 1]
        report[name] = {"base_rate": round(float(target.mean()), 3), "mean_predicted": round(float(p.mean()), 3),
                        "auc": round(float(roc_auc_score(target, p)), 3), "brier": round(float(brier_score_loss(target, p)), 4),
                        "base_rate_brier": round(float(target.mean() * (1 - target.mean())), 4)}
    print(json.dumps(report, indent=2))
    final = fit(frame)
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, model in final.items():
        path = OUT / f"{name}.pkl"
        joblib.dump(model, path)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "OUTLOOK_MANIFEST.json").write_text(json.dumps({
        "version": "over_outlook_v1", "trained_at": datetime.now(timezone.utc).isoformat(), "rows": int(len(frame)),
        "features": V3_FEATURES, "categorical": CATEGORICAL_FEATURES, "big_over_runs": BIG, "quiet_over_runs": QUIET,
        "holdout_2025_plus_fit_on_up_to_2023": report, "sha256": files}, indent=2), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
