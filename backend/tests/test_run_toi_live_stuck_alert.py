"""Regression test for the stuck-feed alert in run_toi_live.py.

Found live: a real match's TOI feed can stay unreconciled (ToiFeedError)
continuously for minutes, not just "briefly cross at a live boundary" as
the retry logic originally assumed. Without an alert, that state is
indistinguishable from the process being broken. This asserts exactly one
alert fires once the configured threshold is crossed, and it is not
repeated on continued failure.
"""

from __future__ import annotations

import sys
import threading
import time
from unittest.mock import MagicMock, patch

import run_toi_live
from app.live.toi_reader import ToiFeedError


class _RecordingPublisher:
    def __init__(self, *_args, **_kwargs) -> None:
        self.sent: list[tuple[str, str]] = []

    def publish(self, key: str, text: str):
        self.sent.append((key, text))
        return MagicMock(sent=True, message_id=1)


def _always_fail(self, url):
    raise ToiFeedError("simulated: feed never reconciles.")


def test_stuck_feed_sends_exactly_one_alert_and_no_duplicates() -> None:
    publisher = _RecordingPublisher()
    argv = [
        "run_toi_live.py",
        "https://timesofindia.indiatimes.com/sports/cricket/match-center-scorecard/fake-match/abc123",
        "--poll-seconds",
        "0.02",
        "--stuck-alert-seconds",
        "1.5",
        "--telegram",
    ]

    with (
        patch.object(sys, "argv", argv),
        patch("run_toi_live.TelegramPublisher", return_value=publisher),
        patch.object(
            run_toi_live.ToiLiveReader, "fetch", _always_fail
        ),
    ):
        thread = threading.Thread(target=run_toi_live.main, daemon=True)
        thread.start()
        # The retry loop floors sleep at 1s regardless of --poll-seconds
        # (time.sleep(max(1.0, feed_retry))), so this covers ~4 iterations,
        # crossing the 1.5s stuck threshold with room to check no repeat.
        time.sleep(4.0)

    assert len(publisher.sent) == 1
    key, text = publisher.sent[0]
    assert key.startswith("stuck_alert:abc123:")
    assert "inconsistent" in text
    assert "resume automatically" in text
