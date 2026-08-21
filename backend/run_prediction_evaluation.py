"""Create prediction-versus-actual reports from local Cricsheet archives."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.replay.evaluation_policy import MatchExcludedError
from app.replay.match_state_verifier import MatchStateVerificationError
from app.services.prediction_evaluator import PredictionEvaluation, PredictionEvaluator

logger = logging.getLogger(__name__)


def _summary_template() -> dict[str, float | int]:
    return {
        "predictions": 0,
        "absolute_run_error": 0.0,
        "squared_run_error": 0.0,
        "wicket_brier_score": 0.0,
        "correct_wicket_predictions": 0,
    }


def _update_summary(summary: dict[str, float | int], row: PredictionEvaluation) -> None:
    summary["predictions"] += 1
    summary["absolute_run_error"] += abs(row.run_error)
    summary["squared_run_error"] += row.run_error**2
    summary["wicket_brier_score"] += (
        row.wicket_probability - int(row.actual_wicket)
    ) ** 2
    summary["correct_wicket_predictions"] += int(row.wicket_prediction_correct)


def _finalize_summary(summary: dict[str, float | int]) -> dict[str, float | int]:
    predictions = int(summary["predictions"])
    if predictions == 0:
        return summary
    return {
        "predictions": predictions,
        "runs_mae": round(float(summary["absolute_run_error"]) / predictions, 4),
        "runs_rmse": round(
            (float(summary["squared_run_error"]) / predictions) ** 0.5, 4
        ),
        "wicket_brier_score": round(
            float(summary["wicket_brier_score"]) / predictions, 4
        ),
        "wicket_accuracy": round(
            int(summary["correct_wicket_predictions"]) / predictions,
            4,
        ),
    }


def generate_report(
    source_directories: list[Path],
    output_file: Path,
    summary_file: Path,
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Write one prediction-versus-actual row for every replayed over."""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    files = [
        file_path
        for directory in source_directories
        for file_path in sorted(directory.glob("*.json"))
    ]
    files = files[offset:] if limit is None else files[offset : offset + limit]
    loader = JsonLoader()
    parser = MatchParser()
    evaluator = PredictionEvaluator()
    by_format: dict[str, dict[str, float | int]] = defaultdict(_summary_template)
    overall = _summary_template()
    failures: list[dict[str, Any]] = []
    verification_failures = 0
    exclusions: list[dict[str, str]] = []
    ineligible_matches = 0
    anomaly_exclusions = 0
    rows_written = 0
    rule_exception_rows = 0
    rule_exception_matches: set[str] = set()

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer: csv.DictWriter[str] | None = None
        for position, file_path in enumerate(files, start=1):
            try:
                match = parser.parse(loader.load(file_path))
                evaluations = evaluator.evaluate_match(match, file_path.name)
                for evaluation in evaluations:
                    if writer is None:
                        writer = csv.DictWriter(
                            handle, fieldnames=evaluation.to_dict().keys()
                        )
                        writer.writeheader()
                    writer.writerow(evaluation.to_dict())
                    _update_summary(overall, evaluation)
                    _update_summary(by_format[evaluation.match_format], evaluation)
                    rows_written += 1
                    if evaluation.rule_exception:
                        rule_exception_rows += 1
                        rule_exception_matches.add(file_path.name)
            except MatchExcludedError as error:
                logger.info("Excluding %s: %s", file_path, error.reason)
                exclusions.append(
                    {
                        "source_file": file_path.name,
                        "match_id": error.match_id,
                        "reason": error.reason,
                        "category": error.category,
                    }
                )
                if error.category == "ineligible_competition":
                    ineligible_matches += 1
                else:
                    anomaly_exclusions += 1
            except MatchStateVerificationError as error:
                logger.warning("Replay verification failed for %s: %s", file_path, error)
                verification_failures += 1
                failures.append(
                    {
                        "source_file": file_path.name,
                        "stage": "replay_verification",
                        "error": str(error),
                        "issues": [
                            {
                                "innings_index": issue.innings_index,
                                "over_number": issue.over_number,
                                "field": issue.field,
                                "expected": repr(issue.expected),
                                "actual": repr(issue.actual),
                            }
                            for issue in error.result.issues
                        ],
                    }
                )
            except Exception as error:
                logger.exception("Evaluation failed for %s", file_path)
                failures.append(
                    {
                        "source_file": file_path.name,
                        "stage": "evaluation",
                        "error": str(error),
                    }
                )

            if position % 100 == 0 or position == len(files):
                logger.info("Processed %d/%d matches.", position, len(files))

    report = {
        "matches_seen": len(files),
        "matches_failed": len(failures),
        "matches_excluded": len(exclusions),
        "ineligible_matches": ineligible_matches,
        "anomaly_exclusions": anomaly_exclusions,
        "verification_failures": verification_failures,
        "prediction_rows": rows_written,
        "rule_exception_rows": rule_exception_rows,
        "rule_exception_matches": len(rule_exception_matches),
        "overall": _finalize_summary(overall),
        "by_match_format": {
            match_format: _finalize_summary(summary)
            for match_format, summary in sorted(by_format.items())
        },
        "failures": failures,
        "exclusions": exclusions,
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
    """Run report generation from the command line."""

    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("ipl", "t20i", "odi"),
        default=("ipl", "t20i", "odi"),
        help="Archives to evaluate (default: all local archives).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data/reports/over_predictions.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=project_root / "data/reports/over_prediction_summary.json",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of source files to skip before evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of source files to evaluate.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    report = generate_report(
        [project_root / "data/raw/cricsheet" / name for name in args.formats],
        args.output,
        args.summary,
        offset=args.offset,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
