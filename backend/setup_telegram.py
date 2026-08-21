"""Interactively configure and test CricketBaba's Telegram bot."""

from __future__ import annotations

import getpass
import json
import os
from pathlib import Path

import requests


def main() -> None:
    token = getpass.getpass("Paste the BotFather token (hidden): ").strip()
    chat_id = input("Telegram chat ID [6229606870]: ").strip() or "6229606870"
    if not token or ":" not in token:
        raise SystemExit("That does not look like a BotFather token.")

    base_url = f"https://api.telegram.org/bot{token}"
    identity = requests.get(f"{base_url}/getMe", timeout=10).json()
    if not identity.get("ok"):
        raise SystemExit(f"Telegram rejected the token: {identity!r}")
    sent = requests.post(
        f"{base_url}/sendMessage",
        json={"chat_id": chat_id, "text": "CricketBaba connected"},
        timeout=10,
    ).json()
    if not sent.get("ok"):
        raise SystemExit(f"Telegram could not send the test: {sent!r}")

    config_path = Path(__file__).resolve().parent / ".telegram.json"
    config_path.write_text(
        json.dumps({"bot_token": token, "chat_id": chat_id}), encoding="utf-8"
    )
    os.chmod(config_path, 0o600)
    username = identity["result"].get("username", "configured bot")
    print(f"Connected @{username}. Settings saved securely in {config_path.name}.")


if __name__ == "__main__":
    main()
