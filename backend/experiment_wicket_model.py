"""Experiment: can the wicket classifier's discrimination (AUC) be improved
beyond the current production model (measured AUC ~0.55 in our replay
diagnostics), via class-imbalance handling or hyperparameter changes?

Trains candidate models to models/candidates/wkt_experiment_*/ -- does NOT
touch the production models/wkt_model.pkl. Uses the exact same group
(by match_id) train/test split methodology as src/train.py for an honest,
large-scale comparison (not just our small replay-diagnostic samples).
"""

import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
from features import build_features, get_feature_columns  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "real_overs.csv"

print(f"Loading {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
df = build_features(df)

numeric_cols, cat_cols = get_feature_columns()
for c in cat_cols:
    df[c] = df[c].astype("category")

feature_cols = numeric_cols + cat_cols
X = df[feature_cols]
y_wkt = df["wicket_in_over"]

match_ids = df["match_id"].unique()
train_ids, test_ids = train_test_split(match_ids, test_size=0.2, random_state=42)
train_mask = df["match_id"].isin(train_ids)
test_mask = df["match_id"].isin(test_ids)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y_wkt[train_mask], y_wkt[test_mask]

print(f"Train overs: {len(X_train)}, Test overs: {len(X_test)}")
print(f"Train wicket rate: {y_train.mean():.3f}, Test wicket rate: {y_test.mean():.3f}\n")

naive_pred = [y_train.mean()] * len(y_test)
print(
    f"NAIVE (constant train rate): "
    f"AUC=0.500  Brier={brier_score_loss(y_test, naive_pred):.4f}  "
    f"LogLoss={log_loss(y_test, naive_pred):.4f}\n"
)

configs = {
    "baseline_current_hparams": dict(
        n_estimators=300, learning_rate=0.03, max_depth=5, num_leaves=20,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    ),
    "is_unbalance": dict(
        n_estimators=300, learning_rate=0.03, max_depth=5, num_leaves=20,
        subsample=0.8, colsample_bytree=0.8, random_state=42, is_unbalance=True,
    ),
    "deeper_more_trees": dict(
        n_estimators=600, learning_rate=0.02, max_depth=7, num_leaves=40,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        min_child_samples=30, reg_lambda=1.0,
    ),
    "deeper_unbalanced": dict(
        n_estimators=600, learning_rate=0.02, max_depth=7, num_leaves=40,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        min_child_samples=30, reg_lambda=1.0, is_unbalance=True,
    ),
}

results = {}
for name, params in configs.items():
    model = lgb.LGBMClassifier(verbose=-1, **params)
    model.fit(X_train, y_train, categorical_feature=cat_cols)
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    ll = log_loss(y_test, proba)
    results[name] = (auc, brier, ll, model)
    print(f"{name:24s} AUC={auc:.4f}  Brier={brier:.4f}  LogLoss={ll:.4f}")

best_name = max(results, key=lambda k: results[k][0])
print(f"\nBest by AUC: {best_name}")

out_dir = PROJECT_ROOT / "models" / "candidates" / "wkt_experiment_2026"
out_dir.mkdir(parents=True, exist_ok=True)
for name, (auc, brier, ll, model) in results.items():
    joblib.dump(model, out_dir / f"{name}.pkl")
print(f"Saved all candidate models to {out_dir}")
