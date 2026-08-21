"""Oracle diagnostic for bowler strike-rate signal when bowler is announced."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from train_ipl_wicket_v7_context import (
    CATEGORICAL_V7,
    FEATURES,
    _apply_platt,
    _features,
    _fit_platt,
    _frame,
    _model,
    _probability_metrics,
)


VERSION = "ipl_wicket_v8_announced_bowler_diagnostic"
BOWLER_FEATURES = [
    "announced_bowler",
    "bowler_prior_balls",
    "bowler_prior_wickets",
    "bowler_prior_strike_rate",
    "bowler_prior_weight",
    "bowler_match_balls",
    "bowler_match_wickets",
    "bowler_balls_since_wicket",
    "bowler_overdue_ratio",
]


def _fit_probability(training, calibration, holdout, features, categoricals):
    model = _model(4, 450)
    model.fit(
        _frame(training, features, categoricals),
        training["wicket_in_over"],
        cat_features=categoricals,
    )
    calibration_raw = model.predict_proba(
        _frame(calibration, features, categoricals)
    )[:, 1]
    calibrator = _fit_platt(
        calibration_raw, calibration["wicket_in_over"].to_numpy()
    )
    holdout_raw = model.predict_proba(
        _frame(holdout, features, categoricals)
    )[:, 1]
    return _apply_platt(calibrator, holdout_raw)


def diagnose(root: Path) -> dict:
    data = _features(
        pd.read_csv(
            root
            / "data/candidates/ipl_wicket_v8_announced_bowler/training_overs.csv"
        )
    )
    data["match_date"] = pd.to_datetime(data["match_date"])
    data = data.sort_values(["match_date", "source_file", "innings", "over"])
    training = data[data["match_date"].dt.year <= 2023].copy()
    calibration = data[data["match_date"].dt.year == 2024].copy()
    holdout = data[data["match_date"].dt.year.isin([2025, 2026])].copy()
    baseline = _fit_probability(
        training, calibration, holdout, FEATURES, CATEGORICAL_V7
    )
    announced = _fit_probability(
        training,
        calibration,
        holdout,
        FEATURES + BOWLER_FEATURES,
        CATEGORICAL_V7 + ["announced_bowler"],
    )
    matches = (
        holdout[["source_file", "match_date"]]
        .drop_duplicates()
        .sort_values(["match_date", "source_file"])
        .tail(20)
    )
    small_mask = holdout["source_file"].isin(matches["source_file"]).to_numpy()
    actual = holdout["wicket_in_over"].to_numpy()
    report = {
        "diagnostic_version": VERSION,
        "candidate_only": True,
        "production_changed": False,
        "promotion_eligible": False,
        "oracle_input": (
            "The actual next-over bowler is taken from the first delivery. "
            "This is valid in production only if explicitly announced by the feed."
        ),
        "chronological_bowler_history": True,
        "full_holdout": {
            "rows": len(holdout),
            "baseline": _probability_metrics(actual, baseline),
            "announced_bowler": _probability_metrics(actual, announced),
        },
        "small_last_20_match_set": {
            "rows": int(small_mask.sum()),
            "matches": 20,
            "baseline": _probability_metrics(
                actual[small_mask], baseline[small_mask]
            ),
            "announced_bowler": _probability_metrics(
                actual[small_mask], announced[small_mask]
            ),
        },
        "bowler_features": BOWLER_FEATURES,
        "decision": "diagnostic_only_not_promotable",
    }
    output = root / f"data/reports/{VERSION}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    diagnose(Path(__file__).resolve().parents[1])
