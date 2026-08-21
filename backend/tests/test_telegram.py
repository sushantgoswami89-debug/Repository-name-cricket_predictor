"""Production Telegram retry and failure classification tests."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import requests

from app.live.telegram import TelegramPublisher, TelegramPublishError


def _response(status: int, payload: dict) -> Mock:
    response = Mock()
    response.status_code = status
    response.ok = status < 400
    response.json.return_value = payload
    return response


@patch("app.live.telegram.time.sleep")
@patch("app.live.telegram.requests.post")
def test_transient_telegram_failure_is_retried(
    post: Mock, _sleep: Mock
) -> None:
    post.side_effect = [
        requests.ConnectionError("temporary"),
        _response(200, {"ok": True, "result": {"message_id": 42}}),
    ]

    result = TelegramPublisher("token", "chat").publish("m:1:2", "hello")

    assert result.sent is True
    assert result.message_id == 42
    assert post.call_count == 2


@patch("app.live.telegram.requests.post")
def test_permanent_telegram_4xx_is_not_retried(post: Mock) -> None:
    post.return_value = _response(400, {"ok": False, "description": "bad request"})

    with pytest.raises(TelegramPublishError) as caught:
        TelegramPublisher("token", "chat").publish("m:1:2", "hello")

    assert caught.value.retryable is False
    assert post.call_count == 1
