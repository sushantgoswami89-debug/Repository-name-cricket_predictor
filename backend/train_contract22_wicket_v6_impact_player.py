"""CTO-agreed next priority (2026-08-23): IPL Impact Player state as a real
feature, tested here on the wicket side. `ipl_impact_dataset.py` already
existed (leakage-safe, parses Cricsheet's real `replacements.match[].reason
== "impact_player"` events) but was never wired into any live model.

Tested first on run-range (`train_run_range_v8_impact_player.py`, against
the real live `run_range_v7_competition_prior` baseline): no signal, all
impact features ranked in the bottom 15 of 69 by importance, hit rate
slightly below baseline. This candidate tests the more plausible hypothesis
for this signal -- a batter who has *just been subbed in* as an Impact
Player is arguably closer to "new batter at the crease" risk (a known real
wicket-risk factor, see v2's own docstring) than to a run-scoring effect.
Same architecture, split, and bowler-known/unknown evaluation as
`contract22_wicket_v2_batter_state` (currently live) otherwise.
Candidate-only -- does not touch models/ or the live artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from app.ml.ipl_impact_dataset import build_ipl_impact_dataset
from app.ml.ipl_phase_moe_dataset import build_ipl_phase_moe_features
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from train_contract22_rigorous import (
    BASE_FEATURES, BATTER_FEATURES, BOWLER_FEATURES, MATCHUP_FEATURES,
    VENUE_FEATURES, build_enriched,
)
from train_contract22_wicket_v2_batter_state import STATE_FEATURES
from train_phase_calibrated_sharp_range_v33 import male_source_files

VERSION = "contract22_wicket_v6_impact_player"
IMPACT_FEATURES = [
    "ipl_impact_rule_era",
    "ipl_impact_batting_used", "ipl_impact_bowling_used",
    "ipl_impact_batting_available", "ipl_impact_bowling_available",
    "ipl_impact_batting_role", "ipl_impact_bowling_role",
    "ipl_impact_batting_overs_since", "ipl_impact_bowling_overs_since",
]
FEATURES = (
    BASE_FEATURES + BATTER_FEATURES + BOWLER_FEATURES + MATCHUP_FEATURES
    + VENUE_FEATURES + STATE_FEATURES + IMPACT_FEATURES
)
CATEGORICAL = [
    "phase", "striker_batting_style", "bowler_type", "venue_scoring_regime",
    "venue_par_source", "active_batter_state",
    "ipl_impact_batting_role", "ipl_impact_bowling_role",
]

IMPACT_DEFAULTS = {
    "ipl_impact_rule_era": 0, "ipl_impact_batting_used": 0, "ipl_impact_bowling_used": 0,
    "ipl_impact_batting_available": 0, "ipl_impact_bowling_available": 0,
    "ipl_impact_batting_role": "none", "ipl_impact_bowling_role": "none",
    "ipl_impact_batting_overs_since": -1, "ipl_impact_bowling_overs_since": -1,
}


def frame(d: pd.DataFrame, bowler_known: bool) -> pd.DataFrame:
    result = d[FEATURES].copy()
    if not bowler_known:
        for col in BOWLER_FEATURES + MATCHUP_FEATURES:
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
    impact = build_ipl_impact_dataset(root)
    keys = ["source_file", "innings", "over"]
    data = base.merge(enriched, on=keys, how="inner", validate="one_to_one")
    data = data.merge(
        venue[keys + ["venue_par_score", "venue_prior_innings", "venue_scoring_regime", "venue_par_source"]],
        on=keys, how="inner", validate="one_to_one",
    )
    data = data.merge(moe[keys + STATE_FEATURES], on=keys, how="inner", validate="one_to_one")
    data = data.merge(impact[keys + IMPACT_FEATURES], on=keys, how="left", validate="one_to_one")
    print(f"Merged rows: {len(data)} (base was {len(base)})")

    for column, default in IMPACT_DEFAULTS.items():
        data[column] = data[column].fillna(default)
    for column in (
        "ipl_impact_rule_era", "ipl_impact_batting_used", "ipl_impact_bowling_used",
        "ipl_impact_batting_available", "ipl_impact_bowling_available",
        "ipl_impact_batting_overs_since", "ipl_impact_bowling_overs_since",
    ):
        data[column] = data[column].astype(int)

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
    impact_rank = {name: rank + 1 for rank, (name, _) in enumerate(importances)}

    report = {
        "candidate_version": VERSION,
        "candidate_only": True,
        "production_changed": False,
        "note": (
            "contract22_wicket_v2_batter_state + IPL impact-player state "
            "(used/available/role/overs-since, leakage-safe as-of-start-"
            "of-over). Reference: v2 scored AUC 0.6081/0.6072 "
            "(known/unknown), Brier ~0.2049 (Platt) on this same "
            "population."
        ),
        "split": {"train_rows": len(train), "holdout_rows": len(holdout)},
        "reference_contract22_wicket_v2_batter_state": {
            "auc_known": 0.6081, "auc_unknown": 0.6072, "brier_platt": 0.2049,
        },
        "bowler_known_ceiling": ceiling,
        "bowler_unknown_realistic": realistic,
        "impact_feature_importance_rank": {
            f: impact_rank[f] for f in IMPACT_FEATURES
        },
        "total_features": len(FEATURES),
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
