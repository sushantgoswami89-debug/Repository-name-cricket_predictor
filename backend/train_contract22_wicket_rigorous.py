"""Same investigation as train_contract22_rigorous.py (which found the live
runs_model.pkl's random-split/no-dates training was untrustworthy, and
that v3 beats it under real conditions), applied to the wicket side:
models/wkt_model.pkl is trained by the exact same src/train.py, same
train_test_split(match_ids, test_size=0.2, random_state=42) on the same
undated data/real_overs.csv -- the identical methodology gap.

Reuses build_enriched() from train_contract22_rigorous.py unchanged (the
batter/bowler/venue/h2h feature computation doesn't depend on the
prediction target) and trains a binary wicket-in-over classifier instead
of the multiclass runs model, with the same real chronological split
(train<=2023/calib=2024/holdout>=2025) and the same bowler-known-ceiling /
bowler-unknown-realistic evaluation split.

Unlike the runs comparison, this session's earlier v7.x wicket-model line
(11 candidates, docs/candidate_ipl_wicket_v7_2_spell_features.md) already
did extensive rigorous work on this exact target with proper temporal
cutoffs -- best result ipl_wicket_v7_7_t20i_augmented: Brier 0.19384,
ROC-AUC 0.5991 on its own locked holdout (never cleared the stricter
precision/recall promotion gates, but that's a different, harder bar than
plain AUC/Brier). This candidate is evaluated the same way (Brier, AUC) so
all three numbers -- naive/leaky legacy, this rigorous contract-22
version, and the v7.x line's best -- are on the same basis.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from train_contract22_rigorous import (
    BASE_FEATURES, BATTER_FEATURES, BOWLER_FEATURES, MATCHUP_FEATURES,
    VENUE_FEATURES, CATEGORICAL, build_enriched,
)
from train_phase_calibrated_sharp_range_v33 import male_source_files

VERSION = "contract22_wicket_rigorous"
FEATURES = BASE_FEATURES + BATTER_FEATURES + BOWLER_FEATURES + MATCHUP_FEATURES + VENUE_FEATURES


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
    keys = ["source_file", "innings", "over"]
    data = base.merge(enriched, on=keys, how="inner", validate="one_to_one")
    data = data.merge(
        venue[keys + ["venue_par_score", "venue_prior_innings", "venue_scoring_regime", "venue_par_source"]],
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

    def _platt(raw: np.ndarray, actual: np.ndarray) -> LogisticRegression:
        clipped = np.clip(raw, 1e-6, 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        calibrator = LogisticRegression(C=1.0, random_state=42)
        calibrator.fit(logit, actual)
        return calibrator

    def _apply(calibrator: LogisticRegression, raw: np.ndarray) -> np.ndarray:
        clipped = np.clip(raw, 1e-6, 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        return calibrator.predict_proba(logit)[:, 1]

    def evaluate(bowler_known: bool) -> dict:
        # Same Platt-calibration methodology as this session's v7.x wicket
        # line: fit on the 2024 calibration year (same model, same bowler
        # availability as the holdout being scored), apply to holdout.
        calibration_raw = model.predict_proba(frame(calibration, bowler_known))[:, 1]
        calibrator = _platt(calibration_raw, calibration["wicket_in_over"].to_numpy())
        holdout_raw = model.predict_proba(frame(holdout, bowler_known))[:, 1]
        proba = _apply(calibrator, holdout_raw)
        actual = holdout["wicket_in_over"].to_numpy()
        return {
            "auc": float(roc_auc_score(actual, proba)),
            "brier": float(brier_score_loss(actual, proba)),
            "brier_uncalibrated": float(brier_score_loss(actual, holdout_raw)),
            "event_rate": float(actual.mean()),
            "rows": len(actual),
        }

    ceiling = evaluate(bowler_known=True)
    realistic = evaluate(bowler_known=False)

    # Single production calibrator: bowler_known=True calibration fit
    # (uses the full training-distribution richness); validated above that
    # bowler_known vs bowler_unknown calibration barely differs (Brier
    # 0.20507 vs 0.20506), so one calibrator serves both live conditions.
    import joblib

    calibration_raw_prod = model.predict_proba(frame(calibration, bowler_known=True))[:, 1]
    production_calibrator = _platt(calibration_raw_prod, calibration["wicket_in_over"].to_numpy())
    joblib.dump(model, output / "wicket_model.pkl")
    joblib.dump(production_calibrator, output / "wicket_calibrator.pkl")
    joblib.dump(FEATURES, output / "feature_cols.pkl")
    joblib.dump(CATEGORICAL, output / "categorical_cols.pkl")

    report = {
        "candidate_version": VERSION,
        "candidate_only": True,
        "production_changed": False,
        "note": (
            "Wicket-side counterpart to contract22_rigorous.py: same "
            "chronological split, same feature set, binary wicket_in_over "
            "target instead of the multiclass runs model. Compares against "
            "(a) the naive replay-based estimate for the currently-live "
            "models/wkt_model.pkl (AUC 0.5880, Brier 0.1981, bowler-known-"
            "via-replay, same leakage caveats as the runs comparison) and "
            "(b) this session's rigorously-validated v7.x wicket line's "
            "best result (ipl_wicket_v7_7_t20i_augmented: Brier 0.19384, "
            "AUC 0.5991)."
        ),
        "split": {"train_rows": len(train), "holdout_rows": len(holdout)},
        "legacy_naive_replay_estimate": {"auc": 0.5880, "brier": 0.1981},
        "v7_7_t20i_augmented_rigorous": {"brier": 0.19384, "auc": 0.5991},
        "bowler_known_ceiling": ceiling,
        "bowler_unknown_realistic": realistic,
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
