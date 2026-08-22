"""Neither the batter nor bowler side of any model in this codebase has a
per-player venue split -- `ipl_venue_regime_dataset.py` pools ALL players
at a venue into one par score, and the richest per-player split so far
(`wicket_contract22_bowler_phase_stats.json`) is player x phase, no venue.
`player_venue_phase_dataset.py` fills that gap with shrinkage toward the
player's own venue-agnostic phase rate (a raw 3-way cell would mostly be
noise -- see that module's docstring for the sparsity numbers that
motivated shrinkage instead of a flat groupby).

This tests whether that signal adds real discrimination on top of
`contract22_wicket_v2_batter_state` (the currently-live wicket model:
AUC 0.6081/0.6072 known/unknown, Brier ~0.2049), the same way
`contract22_wicket_v3_pressure_state` tested wicket-regime/chase-pressure
granularity and found it added nothing. Same architecture, split, and
bowler-known/unknown evaluation as v2 otherwise. Candidate-only -- does
not touch models/ or the live artifacts.
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
from app.ml.player_venue_phase_dataset import build_player_venue_phase_features
from train_contract22_rigorous import (
    BASE_FEATURES, BATTER_FEATURES, BOWLER_FEATURES, MATCHUP_FEATURES,
    VENUE_FEATURES, build_enriched,
)
from train_contract22_wicket_v2_batter_state import STATE_FEATURES
from train_phase_calibrated_sharp_range_v33 import male_source_files

VERSION = "contract22_wicket_v4_venue_phase"
BATTER_VENUE_PHASE_FEATURES = [
    "batter_venue_phase_balls", "batter_venue_phase_runs_per_ball_shrunk",
    "batter_venue_phase_boundary_rate_shrunk", "batter_venue_phase_dismissal_rate_shrunk",
]
BOWLER_VENUE_PHASE_FEATURES = [
    "bowler_venue_phase_balls", "bowler_venue_phase_economy_shrunk",
    "bowler_venue_phase_wicket_rate_shrunk",
]
FEATURES = (
    BASE_FEATURES + BATTER_FEATURES + BOWLER_FEATURES + MATCHUP_FEATURES
    + VENUE_FEATURES + STATE_FEATURES + BATTER_VENUE_PHASE_FEATURES + BOWLER_VENUE_PHASE_FEATURES
)
CATEGORICAL = [
    "phase", "striker_batting_style", "bowler_type", "venue_scoring_regime",
    "venue_par_source", "active_batter_state",
]


def frame(d: pd.DataFrame, bowler_known: bool) -> pd.DataFrame:
    result = d[FEATURES].copy()
    if not bowler_known:
        for col in BOWLER_FEATURES + MATCHUP_FEATURES + BOWLER_VENUE_PHASE_FEATURES:
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
    venue_phase = build_player_venue_phase_features(root, scopes=("ipl", "t20i"))
    keys = ["source_file", "innings", "over"]
    data = base.merge(enriched, on=keys, how="inner", validate="one_to_one")
    data = data.merge(
        venue[keys + ["venue_par_score", "venue_prior_innings", "venue_scoring_regime", "venue_par_source"]],
        on=keys, how="inner", validate="one_to_one",
    )
    data = data.merge(moe[keys + STATE_FEATURES], on=keys, how="inner", validate="one_to_one")
    data = data.merge(
        venue_phase[keys + BATTER_VENUE_PHASE_FEATURES + BOWLER_VENUE_PHASE_FEATURES],
        on=keys, how="inner", validate="one_to_one",
    )
    print(f"Merged rows: {len(data)} (base was {len(base)})")

    train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
    calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
    holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)

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
        }

    ceiling = evaluate(bowler_known=True)
    realistic = evaluate(bowler_known=False)

    importances = sorted(
        zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]
    )
    venue_phase_rank = {
        name: rank + 1 for rank, (name, _) in enumerate(importances)
    }

    report = {
        "candidate_version": VERSION,
        "candidate_only": True,
        "production_changed": False,
        "note": (
            "contract22_wicket_v2_batter_state + shrunk batter/bowler "
            "venue x phase profiles (player_venue_phase_dataset.py). "
            "Reference: v2 scored AUC 0.6081/0.6072 (known/unknown), "
            "Brier ~0.2049 (Platt) on this same population."
        ),
        "split": {"train_rows": len(train), "holdout_rows": len(holdout)},
        "reference_contract22_wicket_v2_batter_state": {
            "auc_known": 0.6081, "auc_unknown": 0.6072, "brier_platt": 0.2049,
        },
        "bowler_known_ceiling": ceiling,
        "bowler_unknown_realistic": realistic,
        "venue_phase_feature_importance_rank": {
            f: venue_phase_rank[f]
            for f in BATTER_VENUE_PHASE_FEATURES + BOWLER_VENUE_PHASE_FEATURES
        },
        "total_features": len(FEATURES),
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
