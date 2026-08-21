from pathlib import Path

from app.ml.ipl_dataset_validator import (
    REVIEWED_EXCEPTIONS,
    certify_ipl_raw_data,
)


def test_complete_ipl_raw_dataset_is_certified():
    root = Path(__file__).resolve().parents[2]
    report = certify_ipl_raw_data(root)

    assert report["certified"] is True
    assert report["failure_count"] == 0
    assert report["source_files"] == 1243
    assert report["over_rows"] == 47756
    assert report["reviewed_exception_count"] == len(REVIEWED_EXCEPTIONS) == 6


def test_every_exception_is_specific_and_reviewed():
    assert ("335994.json", 2, 10) in REVIEWED_EXCEPTIONS
    assert ("392198.json", 2, 10) in REVIEWED_EXCEPTIONS
    assert ("419155.json", 1, 18) in REVIEWED_EXCEPTIONS
    assert ("1136564.json", 2, 11) in REVIEWED_EXCEPTIONS
    assert ("501247.json", 2, 2) in REVIEWED_EXCEPTIONS
    assert ("1254076.json", 1, 19) in REVIEWED_EXCEPTIONS
