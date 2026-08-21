"""Integration tests for replay verification in evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path

import run_prediction_evaluation
from app.replay.match_state_verifier import (
    MatchStateVerificationError,
    VerificationIssue,
    VerificationResult,
)


class _RejectingEvaluator:
    def evaluate_match(self, match: object, source_file: str) -> list[object]:
        result = VerificationResult(
            source_file,
            1,
            (VerificationIssue(0, 1, "state.score", 0, 99),),
        )
        raise MatchStateVerificationError(result)


def test_report_skips_invalid_match_and_records_verification_details(
    tmp_path: Path, monkeypatch: object
) -> None:
    source = tmp_path / "matches"
    source.mkdir()
    (source / "bad.json").write_text(
        json.dumps(
            {
                "info": {
                    "match_type": "T20",
                    "match_type_number": 999,
                    "teams": ["A", "B"],
                },
                "innings": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        run_prediction_evaluation, "PredictionEvaluator", _RejectingEvaluator
    )
    output = tmp_path / "predictions.csv"
    summary = tmp_path / "summary.json"

    report = run_prediction_evaluation.generate_report([source], output, summary)

    assert report["matches_seen"] == 1
    assert report["matches_failed"] == 1
    assert report["matches_excluded"] == 0
    assert report["ineligible_matches"] == 0
    assert report["anomaly_exclusions"] == 0
    assert report["verification_failures"] == 1
    assert report["prediction_rows"] == 0
    assert report["rule_exception_rows"] == 0
    assert report["rule_exception_matches"] == 0
    assert report["failures"][0]["stage"] == "replay_verification"
    assert report["failures"][0]["issues"][0]["field"] == "state.score"
    assert report["exclusions"] == []
    assert json.loads(summary.read_text(encoding="utf-8")) == report


def test_report_records_known_anomaly_as_exclusion_not_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "matches"
    source.mkdir()
    (source / "1179017.json").write_text(
        json.dumps(
            {
                "info": {
                    "match_type": "T20",
                    "match_type_number": 607,
                    "teams": ["A", "B"],
                },
                "innings": [],
            }
        ),
        encoding="utf-8",
    )

    report = run_prediction_evaluation.generate_report(
        [source], tmp_path / "predictions.csv", tmp_path / "summary.json"
    )

    assert report["matches_failed"] == 0
    assert report["matches_excluded"] == 1
    assert report["anomaly_exclusions"] == 1
    assert report["ineligible_matches"] == 0
    assert report["exclusions"][0]["match_id"] == "1179017"
