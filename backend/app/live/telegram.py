"""Idempotent Telegram publisher for verified predictions and feed alerts."""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests


class TelegramPublishError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class TelegramPublishResult:
    sent: bool
    message_id: int | None = None


class TelegramPublisher:
    def __init__(
        self, bot_token: str, chat_id: str, timeout_seconds: float = 10.0
    ) -> None:
        if not bot_token or not chat_id:
            raise ValueError("Telegram bot token and chat id are required.")
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._chat_id = chat_id
        self._timeout = timeout_seconds
        self._published: set[str] = set()

    def restore_published(self, keys: set[str]) -> None:
        self._published.update(keys)

    def publish(self, key: str, text: str) -> TelegramPublishResult:
        if key in self._published:
            return TelegramPublishResult(False)
        payload = self._send_with_retry(text)
        self._published.add(key)
        result = payload.get("result", {})
        message_id = result.get("message_id") if isinstance(result, dict) else None
        return TelegramPublishResult(
            True, int(message_id) if isinstance(message_id, int) else None
        )

    def publish_correction(self, key: str, text: str) -> TelegramPublishResult:
        return self.publish(key, text)

    def _send_with_retry(self, text: str) -> dict:
        last_error: TelegramPublishError | None = None
        for attempt in range(3):
            try:
                response = requests.post(
                    self._url,
                    json={"chat_id": self._chat_id, "text": text},
                    timeout=self._timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = TelegramPublishError(
                    "Telegram connection failed temporarily.", retryable=True
                )
                if attempt < 2:
                    time.sleep(0.25 * (2**attempt))
                    continue
                raise last_error from exc
            try:
                payload = response.json()
            except requests.JSONDecodeError as exc:
                raise TelegramPublishError(
                    "Telegram returned a non-JSON response.",
                    retryable=response.status_code >= 500,
                ) from exc
            if response.ok and payload.get("ok"):
                return payload
            retryable = response.status_code == 429 or response.status_code >= 500
            last_error = TelegramPublishError(
                f"Telegram rejected the message: {payload!r}",
                retryable=retryable,
            )
            if not retryable or attempt == 2:
                raise last_error
            time.sleep(0.25 * (2**attempt))
        assert last_error is not None
        raise last_error
