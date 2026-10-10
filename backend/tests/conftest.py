"""Keep every test away from the real runtime logs: a test that finishes a fake match or builds a
briefing must not write into data/live/prematch-*.jsonl, which is the live pre-match record."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_prematch_log(monkeypatch, tmp_path_factory):
    from app.live import prematch_log

    folder = tmp_path_factory.mktemp("prematch-log")
    monkeypatch.setattr(prematch_log, "LOG_PATH", folder / "predictions.jsonl")
    monkeypatch.setattr(prematch_log, "RESULTS_PATH", folder / "results.jsonl")
    prematch_log._LAST.clear()


@pytest.fixture(autouse=True)
def _no_cricbuzz_capture(monkeypatch):
    """A finished fake match must not go and fetch eight real Cricbuzz pages."""
    from app.live import cricbuzz_capture

    monkeypatch.setattr(cricbuzz_capture, "capture_match", lambda match_id, root=None: {"saved": [], "winner": None, "curve_points": 0})
