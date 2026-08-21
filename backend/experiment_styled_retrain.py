"""Retrain runs_model and wkt_model on data/real_overs_styled.csv (real
batsman_style/bowler_type instead of constant "unknown") and compare
against the current production model on the exact same held-out split.

Does not touch production models/*.pkl.
"""

import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
from features import build_features, get_feature_columns  # noqa: E402

STYLED_PATH = PROJECT_ROOT / "data" / "real_overs_styled.csv"
ORIGINAL_PATH = PROJECT_ROOT / "data" / "real_overs.csv"

numeric_cols, cat_cols = get_feature_columns()
feature_cols = numeric_cols + cat_cols

HPARAMS = dict(
    n_estimators=300, learning_rate=0.03, max_depth=5, num_leaves=20,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
)


def load_and_split(path):
    df = pd.read_csv(path)
    df = build_features(df)
    for c in cat_cols:
        df[c] = df[c].astype("category")
    match_ids = df["match_id"].unique()
    train_ids, test_ids = train_test_split(match_ids, test_size=0.2, random_state=42)
    train_mask = df["match_id"].isin(train_ids)
    test_mask = df["match_id"].isin(test_ids)
    return df, train_mask, test_mask


results = {}
for label, path in [("original (unknown styles)", ORIGINAL_PATH), ("styled (real styles)", STYLED_PATH)]:
    df, train_mask, test_mask = load_and_split(path)
    X = df[feature_cols]
    y_runs = df["runs_in_over"]
    y_wkt = df["wicket_in_over"]

    X_train, X_test = X[train_mask], X[test_mask]
    y_runs_train, y_runs_test = y_runs[train_mask], y_runs[test_mask]
    y_wkt_train, y_wkt_test = y_wkt[train_mask], y_wkt[test_mask]

    runs_model = lgb.LGBMRegressor(**HPARAMS)
    runs_model.fit(X_train, y_runs_train, categorical_feature=cat_cols)
    runs_mae = mean_absolute_error(y_runs_test, runs_model.predict(X_test))

    wkt_model = lgb.LGBMClassifier(**HPARAMS)
    wkt_model.fit(X_train, y_wkt_train, categorical_feature=cat_cols)
    wkt_proba = wkt_model.predict_proba(X_test)[:, 1]
    wkt_auc = roc_auc_score(y_wkt_test, wkt_proba)
    wkt_brier = brier_score_loss(y_wkt_test, wkt_proba)

    print(f"{label}: RunMAE={runs_mae:.4f}  WktAUC={wkt_auc:.4f}  WktBrier={wkt_brier:.4f}")
    results[label] = (runs_model, wkt_model, runs_mae, wkt_auc, wkt_brier)

    if label.startswith("styled"):
        imp = pd.Series(wkt_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("\nTop 10 wicket-model feature importances (styled):")
        print(imp.head(10))

out_dir = PROJECT_ROOT / "models" / "candidates" / "styled_retrain_2026"
out_dir.mkdir(parents=True, exist_ok=True)
runs_model, wkt_model, *_ = results["styled (real styles)"]
joblib.dump(runs_model, out_dir / "runs_model.pkl")
joblib.dump(wkt_model, out_dir / "wkt_model.pkl")
joblib.dump(feature_cols, out_dir / "feature_cols.pkl")
joblib.dump(cat_cols, out_dir / "cat_cols.pkl")
print(f"\nSaved styled candidate models to {out_dir}")
