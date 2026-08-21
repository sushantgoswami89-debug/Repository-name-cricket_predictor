"""Train separate Sharp Range models by the striker's innings entry phase."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from train_candidate_v33_response import _model
from train_sharp_range_candidate import MAX_RUN_CLASS, _best_bands
from train_validate_candidate_v3 import CATEGORICAL_FEATURES, V3_FEATURES, _features

KEYS = ["source_file", "match_date", "innings", "over"]
PHASES = ("powerplay", "middle", "death")


def train(project_root: Path, output_dir: Path) -> dict[str, Any]:
    base = pd.read_csv(project_root / "data/candidates/v3/verified_training_overs.csv")
    extra = pd.read_csv(
        project_root / "data/candidates/v3.5/batter_entry_phase_features.csv"
    )
    base["match_date"] = base["match_date"].astype(str)
    extra["match_date"] = extra["match_date"].astype(str)
    data = base.merge(extra, on=KEYS, how="inner", validate="one_to_one")
    if len(data) != len(base):
        raise ValueError("Entry-phase features do not reconcile with verified data.")
    data["match_date"] = pd.to_datetime(data["match_date"])
    training = data[data["match_date"].dt.year <= 2023].reset_index(drop=True)
    holdout = data[data["match_date"].dt.year >= 2025].reset_index(drop=True)
    additions = [column for column in extra if column.startswith("entry_phase_batter_")]
    columns = V3_FEATURES + additions
    output_dir.mkdir(parents=True, exist_ok=True)

    all_actual: list[np.ndarray] = []
    all_low: list[np.ndarray] = []
    all_high: list[np.ndarray] = []
    phase_reports: dict[str, Any] = {}
    for entry_phase in PHASES:
        train_segment = training[training["batter_entry_phase"] == entry_phase]
        holdout_segment = holdout[holdout["batter_entry_phase"] == entry_phase]
        model = _model()
        model.fit(
            _features(train_segment, columns),
            np.minimum(train_segment["runs_in_over"], MAX_RUN_CLASS),
            categorical_feature=CATEGORICAL_FEATURES,
        )
        probability = model.predict_proba(_features(holdout_segment, columns))
        low, high, mass = _best_bands(probability, 2)
        actual = np.minimum(holdout_segment["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
        hit = (actual >= low) & (actual <= high)
        phase_reports[entry_phase] = {
            "training_rows": len(train_segment),
            "holdout_rows": len(holdout_segment),
            "holdout_matches": int(holdout_segment["source_file"].nunique()),
            "two_run_hit_rate": float(hit.mean()),
            "mean_band_probability": float(mass.mean()),
            "mean_actual_batter_profile_samples": float(
                holdout_segment["entry_phase_batter_samples"].mean()
            ),
        }
        joblib.dump(model, output_dir / f"{entry_phase}_entry_model.pkl")
        all_actual.append(actual)
        all_low.append(low)
        all_high.append(high)

    actual = np.concatenate(all_actual)
    low = np.concatenate(all_low)
    high = np.concatenate(all_high)
    overall_hit_rate = float(((actual >= low) & (actual <= high)).mean())
    sequence_failures = int(data.duplicated(KEYS).sum())
    gates = {
        "beats_v32": overall_hit_rate > 0.31376664261821996,
        "beats_required_32_38pct": overall_hit_rate >= 0.3238,
        "each_entry_phase_has_100_matches": all(
            values["holdout_matches"] >= 100 for values in phase_reports.values()
        ),
        "complete_match_sequence_failures_zero": sequence_failures == 0,
    }
    report = {
        "candidate_version": "v3.5_batter_entry_phase",
        "model_scope": "candidate_only",
        "production_models_changed": False,
        "entry_definition": "First appearance at crease as striker or non-striker",
        "leakage_rule": "Profiles contain only earlier chronological deliveries",
        "verified_rows": len(data),
        "holdout_rows": len(holdout),
        "features_added": additions,
        "separate_entry_phase_models": phase_reports,
        "overall_two_run_hit_rate": overall_hit_rate,
        "v32_benchmark_hit_rate": 0.31376664261821996,
        "complete_match_sequence_failures": sequence_failures,
        "promotion_gates": gates,
        "integration_recommended": all(gates.values()),
    }
    joblib.dump(columns, output_dir / "feature_cols.pkl")
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            train(root, root / "models/candidates/v3.5_batter_entry_phase"),
            indent=2,
        )
    )
