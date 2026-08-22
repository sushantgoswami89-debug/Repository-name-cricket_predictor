"""contract22_wicket_rigorous (now live in PredictionEngine, AUC 0.5944/
Brier 0.2098 on real replay) has no "new batter at the crease" or
partnership-age signal at all -- one of the best-known wicket-risk factors
in cricket (a batter who has just arrived is measurably more likely to
get out than one who's settled), and this session already computed and
validated this exact signal for the run-range model
(active_batter_state/new_batter/partnership_legal_ball_age/
striker_match_balls/partner_match_balls, via build_ipl_phase_moe_features).
This candidate adds it to the wicket side too, on the same chronological
holdout, testing whether it adds real discrimination the way it did for
runs -- not assumed, checked.

Also compares Platt vs. isotonic calibration on the same holdout (Platt
was what contract22_wicket_rigorous shipped with; isotonic is more
flexible and is what the v7.x wicket line used, which had a larger
calibration sample available per fold).

Same architecture, split, and bowler-known/unknown evaluation as
contract22_wicket_rigorous otherwise. Candidate-only -- does not touch
models/ or the live contract22_wicket_rigorous artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from app.ml.ipl_phase_moe_dataset import build_ipl_phase_moe_features
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from train_contract22_rigorous import (
    BASE_FEATURES, BATTER_FEATURES, BOWLER_FEATURES, MATCHUP_FEATURES,
    VENUE_FEATURES, build_enriched,
)
from train_phase_calibrated_sharp_range_v33 import male_source_files

VERSION = "contract22_wicket_v2_batter_state"
STATE_FEATURES = [
    "active_batter_state", "new_batter", "partnership_legal_ball_age",
    "striker_match_balls", "partner_match_balls",
]
FEATURES = BASE_FEATURES + BATTER_FEATURES + BOWLER_FEATURES + MATCHUP_FEATURES + VENUE_FEATURES + STATE_FEATURES
CATEGORICAL = ["phase", "striker_batting_style", "bowler_type", "venue_scoring_regime", "venue_par_source", "active_batter_state"]


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
    keys = ["source_file", "innings", "over"]
    data = base.merge(enriched, on=keys, how="inner", validate="one_to_one")
    data = data.merge(
        venue[keys + ["venue_par_score", "venue_prior_innings", "venue_scoring_regime", "venue_par_source"]],
        on=keys, how="inner", validate="one_to_one",
    )
    data = data.merge(moe[keys + STATE_FEATURES], on=keys, how="inner", validate="one_to_one")
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

        isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        isotonic.fit(calibration_raw, actual_cal)
        isotonic_proba = isotonic.predict(holdout_raw)

        return {
            "auc": float(roc_auc_score(actual, holdout_raw)),  # calibration-invariant
            "brier_platt": float(brier_score_loss(actual, platt_proba)),
            "brier_isotonic": float(brier_score_loss(actual, isotonic_proba)),
            "brier_uncalibrated": float(brier_score_loss(actual, holdout_raw)),
            "event_rate": float(actual.mean()),
            "rows": len(actual),
        }

    ceiling = evaluate(bowler_known=True)
    realistic = evaluate(bowler_known=False)

    # Production artifacts: Platt calibrator (isotonic landed within noise
    # of Platt on this holdout, no reason to switch), fit on bowler_known
    # calibration data (same rationale as contract22_wicket_rigorous --
    # validated bowler_known vs unknown barely differ).
    import joblib

    calibration_raw_prod = model.predict_proba(frame(calibration, bowler_known=True))[:, 1]
    production_calibrator = _platt_fit(calibration_raw_prod, calibration["wicket_in_over"].to_numpy())
    joblib.dump(model, output / "wicket_model.pkl")
    joblib.dump(production_calibrator, output / "wicket_calibrator.pkl")
    joblib.dump(FEATURES, output / "feature_cols.pkl")
    joblib.dump(CATEGORICAL, output / "categorical_cols.pkl")

    report = {
        "candidate_version": VERSION,
        "candidate_only": True,
        "production_changed": False,
        "note": (
            "contract22_wicket_rigorous + partnership-age/new-batter state "
            "features (active_batter_state, new_batter, "
            "partnership_legal_ball_age, striker_match_balls, "
            "partner_match_balls) from build_ipl_phase_moe_features. Also "
            "compares Platt vs isotonic calibration on the same holdout. "
            "Reference: contract22_wicket_rigorous itself scored AUC "
            "0.6069/0.6063 (bowler known/unknown), Brier ~0.2051 (Platt) "
            "on this same population."
        ),
        "split": {"train_rows": len(train), "holdout_rows": len(holdout)},
        "reference_contract22_wicket_rigorous": {
            "auc_known": 0.6069, "auc_unknown": 0.6063, "brier_platt": 0.2051,
        },
        "bowler_known_ceiling": ceiling,
        "bowler_unknown_realistic": realistic,
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
