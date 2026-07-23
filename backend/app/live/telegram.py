"""Idempotent Telegram publisher for verified predictions and feed alerts."""

from __future__ import annotations

import requests


class TelegramPublishError(RuntimeError):
    pass


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

    def publish(self, key: str, text: str) -> bool:
        if key in self._published:
            return False
        response = requests.post(
            self._url,
            json={"chat_id": self._chat_id, "text": text},
            timeout=self._timeout,
        )
        try:
            payload = response.json()
        except requests.JSONDecodeError as exc:
            raise TelegramPublishError(
                "Telegram returned a non-JSON response."
            ) from exc
        if not response.ok or not payload.get("ok"):
            raise TelegramPublishError(f"Telegram rejected the message: {payload!r}")
        self._published.add(key)
        return True
