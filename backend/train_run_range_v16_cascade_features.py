"""Cascade-feature test (2026-08-25): feed contract22_wicket_v16_team_composition's
own predicted wicket_in_over probability as an input feature to the
run-range model -- the reciprocal direction of
train_contract22_wicket_v18_cascade_features.py, same standing thread
from 2026-08-23. See that script's docstring and
app/ml/cascade_features.py for the full rationale and leakage-safe
out-of-fold methodology.

NOT a proposal to merge engines -- one new scalar feature on top of
run_range_v11_partnership_rate (currently live). Same architecture,
split, and calibration/banding methodology as every other run-range
candidate.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize_scalar

from app.ml.cascade_features import (
    RUN_RANGE_CATEGORICAL, RUN_RANGE_FEATURES, build_run_range_base_dataset, build_wicket_cascade_feature,
)
from train_phase_calibrated_sharp_range_v33 import MAX_RUN_CLASS, best_bands, nll, temperature_scale
from train_run_range_enriched_v3_batting_style import features as run_range_frame

VERSION = "run_range_v16_cascade_features"
CASCADE_FEATURE = "wicket_cascade_probability"
FEATURES = RUN_RANGE_FEATURES + [CASCADE_FEATURE]
CATEGORICAL = RUN_RANGE_CATEGORICAL


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "models/candidates" / VERSION
    output.mkdir(parents=True, exist_ok=True)

    print("Building run-range base dataset...", flush=True)
    base = build_run_range_base_dataset(root)
    print("Building leakage-safe wicket cascade feature (5-fold OOF)...", flush=True)
    cascade = build_wicket_cascade_feature(root)
    keys = ["source_file", "innings", "over"]
    data = base.merge(cascade, on=keys, how="inner", validate="one_to_one")
    print(f"Merged rows: {len(data)} (base was {len(base)}, cascade {len(cascade)})")

    train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
    calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
    holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)
    holdout_is_ipl = holdout["is_ipl"].to_numpy()

    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=MAX_RUN_CLASS + 1,
        n_estimators=60, learning_rate=0.08, num_leaves=25, max_depth=6,
        min_child_samples=100, subsample=0.85, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=42, verbose=-1,
    )
    train_actual = np.minimum(train["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
    model.fit(run_range_frame(train, FEATURES, CATEGORICAL), train_actual, categorical_feature=CATEGORICAL)

    calibration_raw = model.predict_proba(run_range_frame(calibration, FEATURES, CATEGORICAL))
    holdout_raw = model.predict_proba(run_range_frame(holdout, FEATURES, CATEGORICAL))
    calibration_actual = np.minimum(calibration["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
    holdout_actual = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)

    temperatures = {}
    calibration_scaled = calibration_raw.copy()
    holdout_scaled = holdout_raw.copy()
    for phase in ("powerplay", "middle", "death"):
        calibration_mask = calibration["phase"].astype(str).to_numpy() == phase
        holdout_mask = holdout["phase"].astype(str).to_numpy() == phase
        result = minimize_scalar(
            lambda value: nll(
                temperature_scale(calibration_raw[calibration_mask], value),
                calibration_actual[calibration_mask],
            ),
            bounds=(0.5, 3.0), method="bounded",
        )
        temperatures[phase] = float(result.x)
        calibration_scaled[calibration_mask] = temperature_scale(calibration_raw[calibration_mask], temperatures[phase])
        holdout_scaled[holdout_mask] = temperature_scale(holdout_raw[holdout_mask], temperatures[phase])

    calibration_low, calibration_high = best_bands(calibration_scaled)
    holdout_low, holdout_high = best_bands(holdout_scaled)
    holdout_hit = (holdout_actual >= holdout_low) & (holdout_actual <= holdout_high)
    raw_calibration_nll = nll(calibration_raw, calibration_actual)
    scaled_calibration_nll = nll(calibration_scaled, calibration_actual)

    v11_blended, v11_ipl, v11_t20i = 0.2880, 0.2516, 0.2945
    new_blended = float(np.mean(holdout_hit))
    new_ipl = float(np.mean(holdout_hit[holdout_is_ipl]))
    new_t20i = float(np.mean(holdout_hit[~holdout_is_ipl]))
    beats_v11 = bool(new_blended > v11_blended and scaled_calibration_nll <= raw_calibration_nll)
    beats_v11_on_ipl_specifically = bool(new_ipl > v11_ipl)

    importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    importance_rank = {name: rank + 1 for rank, (name, _) in enumerate(importances)}

    report = {
        "candidate_version": VERSION,
        "candidate_only": True,
        "production_changed": False,
        "run_model_changed": False,
        "note": (
            "run_range_v11_partnership_rate (currently live) + "
            "contract22_wicket_v16_team_composition's own predicted "
            "wicket_in_over probability, as a leakage-safe (5-fold "
            "GroupKFold OOF on train, real out-of-sample on calibration/"
            "holdout) cascade feature. Reciprocal of the wicket-side "
            "cascade test, same 2026-08-23 standing thread."
        ),
        "split": {"train_rows": len(train), "holdout_rows": len(holdout)},
        "reference_run_range_v11_partnership_rate": {
            "blended": v11_blended, "ipl": v11_ipl, "t20i": v11_t20i,
        },
        "holdout": {
            "blended_hit_rate": new_blended,
            "ipl_hit_rate": new_ipl,
            "t20i_hit_rate": new_t20i,
        },
        "beats_v11_blended_and_calibration_gate": beats_v11,
        "beats_v11_on_ipl_specifically": beats_v11_on_ipl_specifically,
        "cascade_feature_importance_rank": {CASCADE_FEATURE: importance_rank[CASCADE_FEATURE]},
        "total_features": len(FEATURES),
        "decision": "promote_candidate" if beats_v11 else "reject_keep_research",
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
