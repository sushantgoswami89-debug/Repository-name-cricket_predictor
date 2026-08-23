"""Web-research-driven direction (2026-08-23). See
`train_run_range_v13_batter_phase_recency.py` for the full rationale
(death overs are the most decisive IPL phase; "finishing ability" is a
distinct, form-sensitive skill). That candidate was rejected for
run-range (worse on all three cuts). Tested here on the wicket side: is
a batter's *recent* dismissal rate specifically in this phase (e.g.
currently going for big shots and getting out more in the death overs)
more predictive of wicket-in-over risk than the phase-agnostic recency
already live in `contract22_wicket_v10_partnership_rate`? Note: unlike
the bowler-phase-recency test, the wicket line has no existing flat
batter-phase feature at all (`BATTER_FEATURES` here has no phase split,
only `run_range_v4_batter_phase`'s line does) -- this tests whether
phase-specific batter dismissal tendency matters for wicket risk at all,
recency-weighted from the start. Same architecture, split,
bowler-known/unknown evaluation, and real IPL/T20I split methodology as
the recent promotions. Candidate-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from app.ml.ipl_phase_moe_dataset import build_ipl_phase_moe_features
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from app.ml.partnership_dataset import build_partnership_dataset
from app.ml.recency_weighted_prior_dataset import (
    build_batter_phase_recency_dataset, build_recency_weighted_prior_dataset,
)
from train_contract22_rigorous import (
    BASE_FEATURES, BATTER_FEATURES, BOWLER_FEATURES, MATCHUP_FEATURES,
    VENUE_FEATURES, build_enriched,
)
from train_contract22_wicket_v2_batter_state import STATE_FEATURES
from train_phase_calibrated_sharp_range_v33 import male_source_files

VERSION = "contract22_wicket_v15_batter_phase_recency"
RECENCY_FEATURES = [
    "striker_recency_balls", "striker_recency_runs_per_ball", "striker_recency_dot_rate",
    "striker_recency_boundary_rate", "striker_recency_dismissal_rate",
    "partner_recency_runs_per_ball",
    "bowler_recency_balls", "bowler_recency_economy", "bowler_recency_wicket_rate",
]
PARTNERSHIP_FEATURES = ["partnership_runs", "partnership_balls", "partnership_run_rate"]
BATTER_PHASE_RECENCY_FEATURES = [
    "striker_recency_phase_balls", "striker_recency_phase_runs_per_ball",
    "striker_recency_phase_boundary_rate", "striker_recency_phase_dismissal_rate",
]
FEATURES = (
    BASE_FEATURES + BATTER_FEATURES + BOWLER_FEATURES + MATCHUP_FEATURES
    + VENUE_FEATURES + STATE_FEATURES + RECENCY_FEATURES + PARTNERSHIP_FEATURES
    + BATTER_PHASE_RECENCY_FEATURES
)
CATEGORICAL = ["phase", "striker_batting_style", "bowler_type", "venue_scoring_regime", "venue_par_source", "active_batter_state"]


def frame(d: pd.DataFrame, bowler_known: bool) -> pd.DataFrame:
    result = d[FEATURES].copy()
    if not bowler_known:
        for col in BOWLER_FEATURES + MATCHUP_FEATURES + ["bowler_recency_balls", "bowler_recency_economy", "bowler_recency_wicket_rate"]:
            if col in CATEGORICAL:
                result[col] = "__UNKNOWN__"
            else:
                result[col] = 0.0
    for column in CATEGORICAL:
        result[column] = result[column].fillna("__UNKNOWN__").astype("category")
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "models/candidates" / VERSION
    output.mkdir(parents=True, exist_ok=True)

    eligible = male_source_files(root)
    base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    base = base[base["source_file"].isin(eligible)].copy()
    base["match_date"] = pd.to_datetime(base["match_date"])

    enriched = build_enriched(root, eligible)
    venue = build_ipl_venue_regime_dataset(root, scopes=("ipl", "t20i"))
    moe = build_ipl_phase_moe_features(root, canonical_identities=True, scopes=("ipl", "t20i"))
    recency = build_recency_weighted_prior_dataset(root, scopes=("ipl", "t20i"))
    partnership = build_partnership_dataset(root, scopes=("ipl", "t20i"))
    batter_phase_recency = build_batter_phase_recency_dataset(root, scopes=("ipl", "t20i"))
    keys = ["source_file", "innings", "over"]
    data = base.merge(enriched, on=keys, how="inner", validate="one_to_one")
    data = data.merge(
        venue[keys + ["venue_par_score", "venue_prior_innings", "venue_scoring_regime", "venue_par_source"]],
        on=keys, how="inner", validate="one_to_one",
    )
    data = data.merge(moe[keys + STATE_FEATURES], on=keys, how="inner", validate="one_to_one")
    data = data.merge(recency[keys + RECENCY_FEATURES], on=keys, how="inner", validate="one_to_one")
    data = data.merge(partnership[keys + PARTNERSHIP_FEATURES], on=keys, how="inner", validate="one_to_one")
    data = data.merge(batter_phase_recency[keys + BATTER_PHASE_RECENCY_FEATURES], on=keys, how="inner", validate="one_to_one")
    print(f"Merged rows: {len(data)} (base was {len(base)})")

    ipl_files = {p.name for p in (root / "data/raw/cricsheet/ipl").glob("*.json")}
    data["is_ipl"] = data["source_file"].isin(ipl_files)

    train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
    calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
    holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)
    holdout_is_ipl = holdout["is_ipl"].to_numpy()

    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=300, learning_rate=0.03, max_depth=5,
        num_leaves=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    )
    model.fit(frame(train, bowler_known=True), train["wicket_in_over"], categorical_feature=CATEGORICAL)

    def _platt_fit(raw: np.ndarray, actual: np.ndarray) -> LogisticRegression:
        clipped = np.clip(raw, 1e-6, 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        calibrator = LogisticRegression(C=1.0, random_state=42)
        calibrator.fit(logit, actual)
        return calibrator

    def _platt_apply(calibrator: LogisticRegression, raw: np.ndarray) -> np.ndarray:
        clipped = np.clip(raw, 1e-6, 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        return calibrator.predict_proba(logit)[:, 1]

    def _metrics(actual: np.ndarray, proba: np.ndarray) -> dict:
        if len(np.unique(actual)) < 2:
            return {"rows": int(len(actual)), "event_rate": float(actual.mean())}
        return {
            "rows": int(len(actual)),
            "event_rate": float(actual.mean()),
            "auc": float(roc_auc_score(actual, proba)),
            "brier": float(brier_score_loss(actual, proba)),
        }

    def evaluate(bowler_known: bool) -> dict:
        calibration_raw = model.predict_proba(frame(calibration, bowler_known))[:, 1]
        holdout_raw = model.predict_proba(frame(holdout, bowler_known))[:, 1]
        actual_cal = calibration["wicket_in_over"].to_numpy()
        actual = holdout["wicket_in_over"].to_numpy()

        platt = _platt_fit(calibration_raw, actual_cal)
        platt_proba = _platt_apply(platt, holdout_raw)

        return {
            "auc": float(roc_auc_score(actual, holdout_raw)),
            "brier_platt": float(brier_score_loss(actual, platt_proba)),
            "brier_uncalibrated": float(brier_score_loss(actual, holdout_raw)),
            "event_rate": float(actual.mean()),
            "rows": len(actual),
            "ipl_platt": _metrics(actual[holdout_is_ipl], platt_proba[holdout_is_ipl]),
            "t20i_platt": _metrics(actual[~holdout_is_ipl], platt_proba[~holdout_is_ipl]),
        }

    ceiling = evaluate(bowler_known=True)
    realistic = evaluate(bowler_known=False)

    importances = sorted(
        zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]
    )
    partnership_rank = {name: rank + 1 for rank, (name, _) in enumerate(importances)}

    report = {
        "candidate_version": VERSION,
        "candidate_only": True,
        "production_changed": False,
        "note": (
            "contract22_wicket_v10_partnership_rate (currently live) + "
            "phase-specific batter recency (recent strike-rate/"
            "dismissal-rate within the same phase as the current over). "
            "Reference: v10 scored AUC 0.6118/0.6102 blended "
            "(known/unknown), IPL 0.6104/0.6059, T20I 0.6113/0.6110 on "
            "this same population."
        ),
        "split": {"train_rows": len(train), "holdout_rows": len(holdout)},
        "reference_contract22_wicket_v10_partnership_rate": {
            "auc_known_blended": 0.6118, "auc_unknown_blended": 0.6102,
            "auc_known_ipl": 0.6104, "auc_unknown_ipl": 0.6059,
            "auc_known_t20i": 0.6113, "auc_unknown_t20i": 0.6110,
        },
        "bowler_known_ceiling": ceiling,
        "bowler_unknown_realistic": realistic,
        "batter_phase_recency_feature_importance_rank": {
            f: partnership_rank[f] for f in BATTER_PHASE_RECENCY_FEATURES
        },
        "total_features": len(FEATURES),
    }
    import joblib

    calibration_raw_prod = model.predict_proba(frame(calibration, bowler_known=True))[:, 1]
    production_calibrator = _platt_fit(calibration_raw_prod, calibration["wicket_in_over"].to_numpy())
    joblib.dump(model, output / "wicket_model.pkl")
    joblib.dump(production_calibrator, output / "wicket_calibrator.pkl")
    joblib.dump(FEATURES, output / "feature_cols.pkl")
    joblib.dump(CATEGORICAL, output / "categorical_cols.pkl")
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
