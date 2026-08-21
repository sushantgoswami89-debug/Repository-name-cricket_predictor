"""Run locked rolling-origin checks for the hierarchical T20-prior design."""

from __future__ import annotations

import json
from pathlib import Path

from train_ipl_cold_start_v3 import CATEGORICAL_V3, FEATURES, train
from train_ipl_cold_start_v5_t20_priors import T20_FEATURES


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = (
        root
        / "data/candidates/ipl_cold_start_v5_t20_priors/training_overs.csv"
    )
    folds = {}
    for validation_year in (2023, 2024):
        report = train(
            root,
            dataset_path=dataset,
            version=f"ipl_cold_start_v5_rolling_{validation_year}",
            features=FEATURES + T20_FEATURES,
            categorical=CATEGORICAL_V3 + ["t20_prior_source"],
            training_end_year=validation_year - 2,
            calibration_year=validation_year - 1,
            holdout_years=(validation_year,),
        )
        folds[str(validation_year)] = {
            "chronology": report["chronology"],
            "baseline": report["baseline_corrected_canonical"],
            "candidate": report["gated_candidate"],
            "cold_start": report["cold_start_only"],
            "decision": report["decision"],
        }
    destination = root / "data/reports/ipl_cold_start_v5_t20_priors"
    destination.mkdir(parents=True, exist_ok=True)
    result = {
        "rolling_origin": True,
        "future_information_used": False,
        "folds": folds,
    }
    (destination / "rolling_validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
