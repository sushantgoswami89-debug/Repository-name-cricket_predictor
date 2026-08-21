"""Merge batched prediction reports into one CSV and one summary JSON file."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _summary_template() -> dict[str, float | int]:
    return {
        "predictions": 0,
        "absolute_run_error": 0.0,
        "squared_run_error": 0.0,
        "wicket_brier_score": 0.0,
        "correct_wicket_predictions": 0,
    }


def _update(summary: dict[str, float | int], row: dict[str, str]) -> None:
    run_error = float(row["run_error"])
    wicket_probability = float(row["wicket_probability"])
    actual_wicket = row["actual_wicket"] == "True"
    summary["predictions"] += 1
    summary["absolute_run_error"] += abs(run_error)
    summary["squared_run_error"] += run_error**2
    summary["wicket_brier_score"] += (wicket_probability - int(actual_wicket)) ** 2
    summary["correct_wicket_predictions"] += int(
        row["wicket_prediction_correct"] == "True"
    )


def _finalize(summary: dict[str, float | int]) -> dict[str, float | int]:
    count = int(summary["predictions"])
    return {
        "predictions": count,
        "runs_mae": round(float(summary["absolute_run_error"]) / count, 4),
        "runs_rmse": round((float(summary["squared_run_error"]) / count) ** 0.5, 4),
        "wicket_brier_score": round(float(summary["wicket_brier_score"]) / count, 4),
        "wicket_accuracy": round(
            int(summary["correct_wicket_predictions"]) / count,
            4,
        ),
    }


def merge_reports(
    report_files: list[Path], output_file: Path, summary_file: Path
) -> dict[str, Any]:
    """Merge report rows and recompute their cross-check summary."""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    overall = _summary_template()
    by_format: dict[str, dict[str, float | int]] = defaultdict(_summary_template)
    row_count = 0
    header: list[str] | None = None

    with output_file.open("w", newline="", encoding="utf-8") as output_handle:
        writer: csv.DictWriter[str] | None = None
        for report_file in report_files:
            with report_file.open(newline="", encoding="utf-8") as input_handle:
                reader = csv.DictReader(input_handle)
                if reader.fieldnames is None:
                    continue
                if header is None:
                    header = reader.fieldnames
                    writer = csv.DictWriter(output_handle, fieldnames=header)
                    writer.writeheader()
                elif reader.fieldnames != header:
                    raise ValueError(f"Unexpected report columns in {report_file}.")

                for row in reader:
                    if writer is None:
                        raise RuntimeError("A report writer was not created.")
                    writer.writerow(row)
                    _update(overall, row)
                    _update(by_format[row["match_format"]], row)
                    row_count += 1

    report = {
        "prediction_rows": row_count,
        "overall": _finalize(overall),
        "by_match_format": {
            match_format: _finalize(summary)
            for match_format, summary in sorted(by_format.items())
        },
        "notes": [
            (
                "Confidence is currently the model's fixed 0.90 value, "
                "not calibrated confidence."
            ),
            "The current model is trained for T20; ODI rows are labelled exploratory.",
        ],
    }
    summary_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """Merge CSV reports passed on the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(merge_reports(args.reports, args.output, args.summary), indent=2))


if __name__ == "__main__":
    main()
