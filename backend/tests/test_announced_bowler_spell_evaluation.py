from pathlib import Path

import pandas as pd
import pytest

from evaluate_announced_bowler_spell_candidate import _load_ledger


def _valid_row() -> dict:
    return {
        "source_file": "match.json",
        "match_date": "2026-01-01",
        "innings": 1,
        "over": 7,
        "bowler": "Safe Bowler",
        "bowler_source": "scoreboard_pre_over",
        "scoreboard_over": 7,
        "row_is_empty": True,
        "captured_before_first_ball": True,
    }


def test_missing_announcement_ledger_means_no_announced_rows(tmp_path: Path):
    assert _load_ledger(tmp_path / "missing.csv").empty


def test_valid_pre_over_ledger_row_is_accepted(tmp_path: Path):
    path = tmp_path / "ledger.csv"
    pd.DataFrame([_valid_row()]).to_csv(path, index=False)
    ledger = _load_ledger(path)
    assert len(ledger) == 1
    assert ledger.iloc[0]["bowler"] == "Safe Bowler"


@pytest.mark.parametrize(
    "change",
    [
        {"scoreboard_over": 8},
        {"row_is_empty": False},
        {"captured_before_first_ball": False},
        {"bowler": ""},
    ],
)
def test_invalid_pre_over_claim_fails_closed(tmp_path: Path, change: dict):
    row = _valid_row()
    row.update(change)
    path = tmp_path / "ledger.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="scoreboard_pre_over"):
        _load_ledger(path)
