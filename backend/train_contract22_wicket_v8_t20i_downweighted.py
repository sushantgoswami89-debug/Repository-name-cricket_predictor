"""CTO-agreed roadmap priority #3 (2026-08-23): reconcile the v7.x wicket
research line's one proven lever into the model that's actually live.

`ipl_wicket_v7_7_t20i_augmented` (low-weighted T20I rows added to
training, sample_weight=0.2, IPL-only calibration/threshold/holdout) was
the only thing that measurably helped wicket prediction in that research
line -- but it lived in a separate CatBoost/~13-feature/IPL-only-holdout
lineage that was never merged into `contract22_wicket_v2_batter_state`
(LightGBM, full feature set, blended IPL+T20I holdout), which is what's
actually live today. Those two lineages aren't directly comparable, so
this re-tests the same lever -- down-weighting T20I training rows -- on
v2's actual live architecture and features.

Also: `contract22_wicket_v2_batter_state` has only ever reported a
blended IPL+T20I holdout AUC (0.6081/0.6072), never split by competition
-- exactly the standing practice this project adopted after
`finding_blended_holdout_masks_ipl_accuracy.md` (which flagged this
applies to "both wicket and run-range lines", never followed up on for
wicket). This script reports both: the exact-v2 baseline (T20I
weight=1.0, i.e. today's status quo) AND the T20I-downweighted candidate
(weight=0.2, matching the proven v7.x value), each split IPL-only /
T20I-only / blended.
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
from train_contract22_rigorous import (
    BASE_FEATURES, BATTER_FEATURES, BOWLER_FEATURES, MATCHUP_FEATURES,
    VENUE_FEATURES, build_enriched,
)
from train_contract22_wicket_v2_batter_state import STATE_FEATURES
from train_phase_calibrated_sharp_range_v33 import male_source_files

VERSION = "contract22_wicket_v8_t20i_downweighted"
T20I_SAMPLE_WEIGHT = 0.2  # matches the proven ipl_wicket_v7_7_t20i_augmented value
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


def evaluate_split(model, calibration: pd.DataFrame, holdout: pd.DataFrame, holdout_is_ipl: np.ndarray, bowler_known: bool) -> dict:
    calibration_raw = model.predict_proba(frame(calibration, bowler_known))[:, 1]
    holdout_raw = model.predict_proba(frame(holdout, bowler_known))[:, 1]
    actual_cal = calibration["wicket_in_over"].to_numpy()
    actual = holdout["wicket_in_over"].to_numpy()
    platt = _platt_fit(calibration_raw, actual_cal)
    platt_proba = _platt_apply(platt, holdout_raw)
    return {
        "blended": _metrics(actual, platt_proba),
        "ipl": _metrics(actual[holdout_is_ipl], platt_proba[holdout_is_ipl]),
        "t20i": _metrics(actual[~holdout_is_ipl], platt_proba[~holdout_is_ipl]),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "models/candidates" / VERSION
    output.mkdir(parents=True, exist_ok=True)

    ipl_files = {p.name for p in (root / "data/raw/cricsheet/ipl").glob("*.json")}
    eligible = male_source_files(root)
    base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    base = base[base["source_file"].isin(eligible)].copy()
    base["match_date"] = pd.to_datetime(base["match_date"])

    print("Building enriched/venue/state features...", flush=True)
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

    data["is_ipl"] = data["source_file"].isin(ipl_files)

    train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
    calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
    holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)
    holdout_is_ipl = holdout["is_ipl"].to_numpy()

    common_kwargs = dict(
        objective="binary", n_estimators=300, learning_rate=0.03, max_depth=5,
        num_leaves=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    )

    print("Training baseline (T20I weight=1.0, exact v2 reproduction)...", flush=True)
    baseline_model = lgb.LGBMClassifier(**common_kwargs)
    baseline_model.fit(frame(train, bowler_known=True), train["wicket_in_over"], categorical_feature=CATEGORICAL)

    print("Training candidate (T20I weight=0.2)...", flush=True)
    sample_weight = np.where(train["is_ipl"].to_numpy(), 1.0, T20I_SAMPLE_WEIGHT)
    candidate_model = lgb.LGBMClassifier(**common_kwargs)
    candidate_model.fit(
        frame(train, bowler_known=True), train["wicket_in_over"],
        sample_weight=sample_weight, categorical_feature=CATEGORICAL,
    )

    baseline_ceiling = evaluate_split(baseline_model, calibration, holdout, holdout_is_ipl, bowler_known=True)
    baseline_realistic = evaluate_split(baseline_model, calibration, holdout, holdout_is_ipl, bowler_known=False)
    candidate_ceiling = evaluate_split(candidate_model, calibration, holdout, holdout_is_ipl, bowler_known=True)
    candidate_realistic = evaluate_split(candidate_model, calibration, holdout, holdout_is_ipl, bowler_known=False)

    beats_on_ipl_known = candidate_ceiling["ipl"].get("auc", 0) > baseline_ceiling["ipl"].get("auc", 0)
    beats_on_ipl_unknown = candidate_realistic["ipl"].get("auc", 0) > baseline_realistic["ipl"].get("auc", 0)
    beats_on_ipl_brier = candidate_ceiling["ipl"].get("brier", 1) < baseline_ceiling["ipl"].get("brier", 1)

    report = {
        "candidate_version": VERSION,
        "candidate_only": True,
        "production_changed": False,
        "note": (
            "Re-tests the proven ipl_wicket_v7_7_t20i_augmented lever "
            "(T20I training rows down-weighted to 0.2, IPL at 1.0) on "
            "contract22_wicket_v2_batter_state's actual live architecture "
            "and full feature set -- the two lineages were never directly "
            "comparable before this. Also reports IPL-only/T20I-only split "
            "for the first time on this model line, per the standing "
            "practice from finding_blended_holdout_masks_ipl_accuracy.md."
        ),
        "split": {"train_rows": len(train), "holdout_rows": len(holdout)},
        "reference_contract22_wicket_v2_batter_state_blended": {
            "auc_known": 0.6081, "auc_unknown": 0.6072, "brier_platt": 0.2049,
        },
        "baseline_t20i_weight_1_0": {
            "bowler_known_ceiling": baseline_ceiling,
            "bowler_unknown_realistic": baseline_realistic,
        },
        "candidate_t20i_weight_0_2": {
            "bowler_known_ceiling": candidate_ceiling,
            "bowler_unknown_realistic": candidate_realistic,
        },
        "beats_baseline_on_ipl_auc_known": bool(beats_on_ipl_known),
        "beats_baseline_on_ipl_auc_unknown": bool(beats_on_ipl_unknown),
        "beats_baseline_on_ipl_brier": bool(beats_on_ipl_brier),
        "decision": (
            "promote_candidate"
            if (beats_on_ipl_known and beats_on_ipl_unknown and beats_on_ipl_brier)
            else "reject_keep_research"
        ),
    }
    import joblib
    joblib.dump(baseline_model, output / "baseline_model_t20i_weight_1_0.pkl")
    joblib.dump(candidate_model, output / "candidate_model_t20i_weight_0_2.pkl")
    joblib.dump(FEATURES, output / "feature_cols.pkl")
    joblib.dump(CATEGORICAL, output / "categorical_cols.pkl")
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
