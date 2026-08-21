from __future__ import annotations

from app.ml.candidate_v3_dataset import _is_legal, _is_wicket, _phase


def test_delivery_classification_handles_extras_and_non_dismissals() -> None:
    assert not _is_legal({"extras": {"wides": 1}})
    assert not _is_legal({"extras": {"noballs": 1}})
    assert _is_legal({"extras": {"byes": 1}})
    assert not _is_wicket({"wickets": [{"kind": "retired hurt"}]})
    assert _is_wicket({"wickets": [{"kind": "bowled"}]})


def test_t20_phase_boundaries_are_one_based() -> None:
    assert _phase(6) == "powerplay"
    assert _phase(7) == "middle"
    assert _phase(15) == "middle"
    assert _phase(16) == "death"
