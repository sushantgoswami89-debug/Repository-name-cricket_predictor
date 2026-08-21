"""Diagnose and calibrate the frozen verified prediction baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error


def _match_dates(source_dirs: list[Path]) -> dict[str, str]:
    dates: dict[str, str] = {}
    for directory in source_dirs:
        for path in directory.glob("*.json"):
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            source_dates = data.get("info", {}).get("dates", [])
            if source_dates:
                dates[path.name] = str(source_dates[0])
    return dates


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    errors = frame["predicted_runs"] - frame["actual_runs"]
    return {
        "rows": int(len(frame)),
        "mean_predicted_runs": round(float(frame["predicted_runs"].mean()), 4),
        "mean_actual_runs": round(float(frame["actual_runs"].mean()), 4),
        "runs_bias": round(float(errors.mean()), 4),
        "runs_mae": round(float(np.abs(errors).mean()), 4),
        "runs_rmse": round(float(np.sqrt(np.square(errors).mean())), 4),
        "mean_wicket_probability": round(
            float(frame["wicket_probability"].mean()), 4
        ),
        "actual_wicket_rate": round(float(frame["actual_wicket"].mean()), 4),
        "wicket_brier": round(
            float(
                np.square(
                    frame["wicket_probability"]
                    - frame["actual_wicket"].astype(float)
                ).mean()
            ),
            4,
        ),
        "wicket_accuracy": round(
            float(frame["wicket_prediction_correct"].mean()), 4
        ),
    }


def _segment_table(data: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, frame in data.groupby(column, observed=True, dropna=False):
        rows.append({column: str(value), **_metrics(frame)})
    return pd.DataFrame(rows)


def _calibration_table(data: pd.DataFrame) -> pd.DataFrame:
    bins = np.arange(0.0, 1.01, 0.1)
    labels = [f"{start:.1f}-{start + 0.1:.1f}" for start in bins[:-1]]
    bucket = pd.cut(
        data["wicket_probability"], bins=bins, labels=labels, include_lowest=True
    )
    rows = []
    for label, frame in data.groupby(bucket, observed=True):
        predicted = float(frame["wicket_probability"].mean())
        actual = float(frame["actual_wicket"].mean())
        rows.append(
            {
                "probability_bin": str(label),
                "rows": int(len(frame)),
                "mean_predicted_probability": predicted,
                "actual_wicket_rate": actual,
                "calibration_gap": predicted - actual,
                "absolute_gap": abs(predicted - actual),
            }
        )
    return pd.DataFrame(rows)


def _candidate_calibration(
    train: pd.DataFrame, holdout: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    y_train = train["actual_wicket"].astype(int).to_numpy()
    y_holdout = holdout["actual_wicket"].astype(int).to_numpy()
    p_train = train["wicket_probability"].to_numpy()
    p_holdout = holdout["wicket_probability"].to_numpy()

    platt = LogisticRegression().fit(p_train.reshape(-1, 1), y_train)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(p_train, y_train)
    probabilities = {
        "uncalibrated": p_holdout,
        "platt": platt.predict_proba(p_holdout.reshape(-1, 1))[:, 1],
        "isotonic": isotonic.predict(p_holdout),
    }
    rows = []
    for name, values in probabilities.items():
        rows.append(
            {
                "candidate": name,
                "holdout_rows": len(holdout),
                "brier_score": brier_score_loss(y_holdout, values),
                "log_loss": log_loss(y_holdout, values),
                "mean_probability": float(np.mean(values)),
                "actual_wicket_rate": float(np.mean(y_holdout)),
            }
        )
    results = pd.DataFrame(rows).sort_values("brier_score")

    runs_model = LinearRegression().fit(
        train[["predicted_runs"]], train["actual_runs"]
    )
    calibrated_runs = np.clip(
        runs_model.predict(holdout[["predicted_runs"]]), 0, None
    )
    raw_runs = holdout["predicted_runs"].to_numpy()
    actual_runs = holdout["actual_runs"].to_numpy()
    runs = {
        "intercept": float(runs_model.intercept_),
        "coefficient": float(runs_model.coef_[0]),
        "raw_holdout_mae": float(mean_absolute_error(actual_runs, raw_runs)),
        "calibrated_holdout_mae": float(
            mean_absolute_error(actual_runs, calibrated_runs)
        ),
        "raw_holdout_rmse": float(mean_squared_error(actual_runs, raw_runs) ** 0.5),
        "calibrated_holdout_rmse": float(
            mean_squared_error(actual_runs, calibrated_runs) ** 0.5
        ),
    }
    return results, runs


def analyze(
    predictions_file: Path, source_dirs: list[Path], output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(predictions_file)
    dates = _match_dates(source_dirs)
    data["match_date"] = pd.to_datetime(data["source_file"].map(dates))
    if data["match_date"].isna().any():
        missing = int(data["match_date"].isna().sum())
        raise ValueError(f"Missing source dates for {missing} prediction rows.")
    data["year"] = data["match_date"].dt.year
    data["phase"] = np.select(
        [data["is_super_over"], data["over"] <= 6, data["over"] <= 15],
        ["super_over", "powerplay", "middle"],
        default="death",
    )
    data["era"] = np.where(data["year"] >= 2022, "2022_and_later", "before_2022")
    striker_counts = data["striker"].value_counts()
    data["striker_sample"] = data["striker"].map(striker_counts)
    data["player_coverage"] = pd.cut(
        data["striker_sample"],
        bins=[0, 49, 499, np.inf],
        labels=["rare_under_50", "established_50_to_499", "high_coverage_500_plus"],
    )

    segment_frames = {
        "scope": _segment_table(data, "eligibility_scope"),
        "innings": _segment_table(data, "innings_number"),
        "phase": _segment_table(data, "phase"),
        "era": _segment_table(data, "era"),
        "player_coverage": _segment_table(data, "player_coverage"),
        "super_over": _segment_table(data, "is_super_over"),
    }
    for name, frame in segment_frames.items():
        frame.to_csv(output_dir / f"segment_{name}.csv", index=False)

    calibration = _calibration_table(data)
    calibration.to_csv(output_dir / "wicket_calibration.csv", index=False)
    ece = float(
        np.average(calibration["absolute_gap"], weights=calibration["rows"])
    )

    years = sorted(data["year"].unique())
    holdout_start = max(2024, years[-1] - 1)
    train = data[data["year"] < holdout_start]
    holdout = data[data["year"] >= holdout_start]
    candidates, runs_calibration = _candidate_calibration(train, holdout)
    candidates.to_csv(output_dir / "calibration_candidates.csv", index=False)

    overall = _metrics(data)
    best = candidates.iloc[0]
    recommendations = [
        {
            "priority": 1,
            "change": "Correct systematic runs underprediction",
            "evidence": (
                f"Mean prediction {overall['mean_predicted_runs']} versus "
                f"actual {overall['mean_actual_runs']}; bias {overall['runs_bias']}."
            ),
        },
        {
            "priority": 2,
            "change": f"Validate {best['candidate']} wicket calibration",
            "evidence": (
                f"Unseen-match Brier score {best['brier_score']:.4f}; "
                f"uncalibrated {float(candidates.loc[candidates['candidate'] == 'uncalibrated', 'brier_score'].iloc[0]):.4f}."
            ),
        },
        {
            "priority": 3,
            "change": "Retrain with time-aware validation and phase interactions",
            "evidence": "Segment tables show material variation by phase and era.",
        },
        {
            "priority": 4,
            "change": "Improve low-coverage player fallbacks",
            "evidence": "Player coverage report isolates rare-player performance.",
        },
    ]

    report: dict[str, Any] = {
        "baseline_file": str(predictions_file),
        "baseline_frozen": True,
        "overall": overall,
        "date_split": {
            "calibration_years": f"{years[0]}-{holdout_start - 1}",
            "unseen_holdout_years": f"{holdout_start}-{years[-1]}",
            "calibration_rows": len(train),
            "holdout_rows": len(holdout),
        },
        "wicket_expected_calibration_error": round(ece, 6),
        "best_wicket_candidate": {
            key: (int(value) if key == "holdout_rows" else round(float(value), 6))
            if key != "candidate"
            else value
            for key, value in best.to_dict().items()
        },
        "runs_linear_calibration": {
            key: round(value, 6) for key, value in runs_calibration.items()
        },
        "recommendations": recommendations,
        "model_replaced": False,
    }
    (output_dir / "accuracy_diagnostic.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=project_root / "data/reports/verified_baseline_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data/reports/accuracy_diagnostic",
    )
    args = parser.parse_args()
    report = analyze(
        args.predictions,
        [
            project_root / "data/raw/cricsheet/ipl",
            project_root / "data/raw/cricsheet/t20i",
        ],
        args.output_dir,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
