"""Compare previous and canonical-v2 global CatBoost candidates by segment."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from train_ipl_catboost_phase_moe_v1 import (
    MOE_FEATURES,
    _apply_point,
    _frame,
)

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    old = pd.read_csv(root / "data/candidates/ipl_phase_moe_v1/features.csv")
    canonical = pd.read_csv(
        root / "data/candidates/ipl_canonical_v2/training_overs.csv"
    )
    base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    venue = pd.read_csv(root / "data/candidates/ipl_venue_regime/features.csv")
    keys = ["source_file", "match_date", "innings", "over"]
    for frame in (old, canonical, base, venue):
        frame["match_date"] = frame["match_date"].astype(str)
    old = base.merge(venue, on=keys, validate="one_to_one").merge(
        old, on=keys, validate="one_to_one"
    )
    old["match_date"] = pd.to_datetime(old["match_date"])
    canonical["match_date"] = pd.to_datetime(canonical["match_date"])
    old = old[old["match_date"].dt.year >= 2025].copy()
    canonical = canonical[canonical["match_date"].dt.year >= 2025].copy()

    old_model = CatBoostRegressor()
    old_model.load_model(
        root / "models/candidates/ipl_catboost_phase_moe_v1/global_runs.cbm"
    )
    new_model = CatBoostRegressor()
    new_model.load_model(
        root
        / "models/candidates/ipl_catboost_global_canonical_v2/global_runs.cbm"
    )
    old_report = json.loads(
        (
            root
            / "models/candidates/ipl_catboost_phase_moe_v1/validation_report.json"
        ).read_text()
    )
    new_report = json.loads(
        (
            root
            / "models/candidates/ipl_catboost_global_canonical_v2/validation_report.json"
        ).read_text()
    )
    old_prediction = _apply_point(
        old,
        np.asarray(old_model.predict(_frame(old, MOE_FEATURES))),
        old_report["calibration"]["global_corrections"],
    )
    new_prediction = _apply_point(
        canonical,
        np.asarray(new_model.predict(_frame(canonical, MOE_FEATURES))),
        new_report["corrections"],
    )
    actual = canonical["runs_in_over"].to_numpy()
    old_error = np.abs(old_prediction - actual)
    new_error = np.abs(new_prediction - actual)

    dimensions = {
        "team": canonical["batting_team"],
        "venue": canonical["venue_name"],
        "phase": canonical["phase"],
        "state": canonical["state_regime"],
        "batter_state": canonical["active_batter_state"],
        "chase_pressure": canonical["chase_pressure"],
        "wickets_remaining": canonical["wickets_remaining_bucket"].astype(str),
    }
    segments = {}
    for dimension, labels in dimensions.items():
        entries = {}
        for label in sorted(labels.astype(str).unique()):
            mask = labels.astype(str).to_numpy() == label
            rows = int(mask.sum())
            entries[label] = {
                "rows": rows,
                "old_mae": float(old_error[mask].mean()),
                "canonical_mae": float(new_error[mask].mean()),
                "mae_change": float(
                    new_error[mask].mean() - old_error[mask].mean()
                ),
                "supported": rows >= 100,
            }
        segments[dimension] = entries
    report = {
        "overall_old_mae": float(old_error.mean()),
        "overall_canonical_mae": float(new_error.mean()),
        "overall_change": float(new_error.mean() - old_error.mean()),
        "segments": segments,
    }
    output = root / "data/reports/ipl_global_canonical_v2_comparison"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report))
