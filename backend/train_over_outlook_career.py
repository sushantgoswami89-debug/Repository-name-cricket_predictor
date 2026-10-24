"""Train the career-strike-rate version of the over outlook (shadow candidate; user 2026-10-24: "start the career strike rate shadow test").
Same recipe as train_over_outlook.py (L2 mean + big / quiet / wicket classifiers, train <= 2023, holdout 2025+) with seven extra inputs from
app/ml/career_strike_rates.py: the two batters' career strike rates (overall and in the coming over's phase) and balls faced. The v1 models are
refitted on the same rows for a like-for-like holdout table. Writes models/over_outlook_v2_career/ (models + OUTLOOK_MANIFEST.json) and
models/career_sr_v1 (the table used live). Usage: python train_over_outlook_career.py
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from app.ml import career_strike_rates as career
from train_over_outlook import BIG, QUIET, fit
from train_validate_candidate_v3 import CATEGORICAL_FEATURES, V3_FEATURES, _features

root = Path(__file__).resolve().parents[1]
OUT = root / "models" / "over_outlook_v2_career"
FEATURES_V2 = V3_FEATURES + career.FEATURES


def evaluate(models: dict, hold: pd.DataFrame, features: list[str]) -> dict:
    X = _features(hold, features)
    actual = hold["runs_in_over"].to_numpy()
    mean = np.clip(models["mean"].predict(X), 0, None)
    report = {"mean_bias": round(float((mean - actual).mean()), 3), "mean_rmse": round(float(np.sqrt(((mean - actual) ** 2).mean())), 3)}
    for name, target in (("big", actual >= BIG), ("quiet", actual <= QUIET), ("wicket", hold["wicket_in_over"].to_numpy() == 1)):
        p = models[name].predict_proba(X)[:, 1]
        report[name] = {"auc": round(float(roc_auc_score(target, p)), 4), "brier": round(float(brier_score_loss(target, p)), 4)}
    return report


def main() -> None:
    print("career table ...", flush=True)
    manifest = career.save()
    print(f"  {manifest['players']} players from {manifest['matches']} matches up to {manifest['as_of']}", flush=True)
    frame = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    extra = career.build_training_features(root)
    frame = frame.merge(extra, on=["source_file", "innings", "over"], how="inner", validate="one_to_one")
    year = pd.to_datetime(frame["match_date"]).dt.year
    train, hold = frame[year <= 2023], frame[year >= 2025]
    print(f"rows {len(frame)} | train {len(train)} | holdout {len(hold)}", flush=True)

    def fit_with(data: pd.DataFrame, features: list[str]) -> dict:
        import lightgbm as lgb
        from train_over_outlook import KW
        X = _features(data, features)
        return {
            "mean": lgb.LGBMRegressor(objective="regression", **KW).fit(X, data["runs_in_over"], categorical_feature=CATEGORICAL_FEATURES),
            "big": lgb.LGBMClassifier(**KW).fit(X, (data["runs_in_over"] >= BIG).astype(int), categorical_feature=CATEGORICAL_FEATURES),
            "quiet": lgb.LGBMClassifier(**KW).fit(X, (data["runs_in_over"] <= QUIET).astype(int), categorical_feature=CATEGORICAL_FEATURES),
            "wicket": lgb.LGBMClassifier(**KW).fit(X, data["wicket_in_over"], categorical_feature=CATEGORICAL_FEATURES),
        }

    v1 = evaluate(fit_with(train, V3_FEATURES), hold, V3_FEATURES)
    v2 = evaluate(fit_with(train, FEATURES_V2), hold, FEATURES_V2)
    print("holdout 2025+, same rows:\n  v1 (live outlook):", json.dumps(v1), "\n  v2 (career rates): ", json.dumps(v2))
    final = fit_with(frame, FEATURES_V2)
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, model in final.items():
        path = OUT / f"{name}.pkl"
        joblib.dump(model, path)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "OUTLOOK_MANIFEST.json").write_text(json.dumps({
        "version": "over_outlook_v2_career", "trained_at": datetime.now(timezone.utc).isoformat(), "rows": int(len(frame)),
        "features": FEATURES_V2, "categorical": CATEGORICAL_FEATURES, "big_over_runs": BIG, "quiet_over_runs": QUIET, "career_table_as_of": manifest["as_of"],
        "holdout_2025_plus_fit_on_up_to_2023": {"v1": v1, "v2": v2}, "sha256": files}, indent=2), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
