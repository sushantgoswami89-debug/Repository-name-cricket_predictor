import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score, log_loss
import joblib
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)
from features import build_features, get_feature_columns

MODEL_DIR = os.path.join(PROJECT_DIR, "models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="synthetic",
        help="Which dataset to train on: 'synthetic' (default) or 'real' "
        "(uses data/real_overs.csv from parse_cricsheet.py), or a full file path",
    )
    args = parser.parse_args()

    if args.data == "synthetic":
        data_path = os.path.join(PROJECT_DIR, "data", "synthetic_overs.csv")
    elif args.data == "real":
        data_path = os.path.join(PROJECT_DIR, "data", "real_overs.csv")
    else:
        data_path = args.data

    print(f"Training on: {data_path}")
    df = pd.read_csv(data_path)
    df = build_features(df)

    numeric_cols, cat_cols = get_feature_columns()

    # LightGBM handles categoricals natively if we cast them to 'category' dtype
    for c in cat_cols:
        df[c] = df[c].astype("category")

    feature_cols = numeric_cols + cat_cols
    X = df[feature_cols]
    y_runs = df["runs_in_over"]
    y_wkt = df["wicket_in_over"]

    # split by match_id so we don't leak future overs of a match into train
    # (group split, not random row split)
    match_ids = df["match_id"].unique()
    train_ids, test_ids = train_test_split(match_ids, test_size=0.2, random_state=42)

    train_mask = df["match_id"].isin(train_ids)
    test_mask = df["match_id"].isin(test_ids)

    X_train, X_test = X[train_mask], X[test_mask]
    y_runs_train, y_runs_test = y_runs[train_mask], y_runs[test_mask]
    y_wkt_train, y_wkt_test = y_wkt[train_mask], y_wkt[test_mask]

    print(f"Train overs: {len(X_train)}, Test overs: {len(X_test)}")

    # --- Model 1: Runs in next over (regression) ---
    runs_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    runs_model.fit(
        X_train,
        y_runs_train,
        categorical_feature=cat_cols,
    )
    runs_pred = runs_model.predict(X_test)
    runs_mae = mean_absolute_error(y_runs_test, runs_pred)
    print(
        f"\n[Runs Model] MAE: {runs_mae:.2f} runs (baseline mean-predictor MAE: "
        f"{mean_absolute_error(y_runs_test, [y_runs_train.mean()]*len(y_runs_test)):.2f})"
    )

    # --- Model 2: Wicket in next over (binary classification) ---
    wkt_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    wkt_model.fit(
        X_train,
        y_wkt_train,
        categorical_feature=cat_cols,
    )
    wkt_pred_proba = wkt_model.predict_proba(X_test)[:, 1]
    wkt_auc = roc_auc_score(y_wkt_test, wkt_pred_proba)
    wkt_ll = log_loss(y_wkt_test, wkt_pred_proba)
    print(
        f"[Wicket Model] AUC: {wkt_auc:.3f} (0.5 = random guessing), LogLoss: {wkt_ll:.3f}"
    )

    # --- Feature importance (this is the "why" you'd want for commentary) ---
    print("\nTop features driving RUNS predictions:")
    imp = pd.Series(runs_model.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    print(imp.head(8))

    print("\nTop features driving WICKET predictions:")
    imp_w = pd.Series(wkt_model.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    print(imp_w.head(8))

    # Save everything needed for inference
    joblib.dump(runs_model, f"{MODEL_DIR}/runs_model.pkl")
    joblib.dump(wkt_model, f"{MODEL_DIR}/wkt_model.pkl")
    joblib.dump(feature_cols, f"{MODEL_DIR}/feature_cols.pkl")
    joblib.dump(cat_cols, f"{MODEL_DIR}/cat_cols.pkl")
    print(f"\nModels saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
