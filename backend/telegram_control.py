"""Private Telegram command listener for CricketBaba automatic mode."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

from app.live.toi_discovery import ToiMatchDiscovery

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
CONFIG_PATH = BACKEND_DIR / ".telegram.json"
RUNTIME_DIR = PROJECT_ROOT / "data" / "live"
PID_PATH = RUNTIME_DIR / "auto-watcher.pid"
OFFSET_PATH = RUNTIME_DIR / "telegram-offset"
LOG_PATH = RUNTIME_DIR / "auto-watcher.log"


class TelegramControl:
    def __init__(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self._chat_id = str(config["chat_id"])
        self._base = f"https://api.telegram.org/bot{config['bot_token']}"
        self._session = requests.Session()
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        offset = self._read_offset()
        while True:
            try:
                response = self._session.get(
                    f"{self._base}/getUpdates",
                    params={
                        "offset": offset,
                        "timeout": 25,
                        "allowed_updates": '["message"]',
                    },
                    timeout=35,
                )
                response.raise_for_status()
                payload = response.json()
                if not payload.get("ok"):
                    raise RuntimeError(f"Telegram polling failed: {payload!r}")
                for update in payload["result"]:
                    offset = int(update["update_id"]) + 1
                    self._write_offset(offset)
                    self._handle(update)
            except (requests.RequestException, RuntimeError, ValueError, KeyError):
                time.sleep(5)

    def _handle(self, update: dict[str, Any]) -> None:
        message = update.get("message", {})
        chat_id = str(message.get("chat", {}).get("id", ""))
        if chat_id != self._chat_id:
            return
        text = " ".join(str(message.get("text", "")).lower().split())
        if text in {
            "cricket predictor match is live",
            "cricket-predictor match is live",
            "match is live",
            "/startwatching",
        }:
            self._send(self._start_watcher())
        elif text in {"status", "/status"}:
            self._send(self._status())
        elif text in {"prematch status", "pre-match status", "/prematch"}:
            self._send(self._prematch_status())
        elif text in {"stop", "/stopwatching"}:
            self._send(self._stop_watcher())
        elif text in {"help", "/help", "/start"}:
            self._send(
                "Commands:\n"
                "• cricket predictor match is live\n"
                "• prematch status\n"
                "• status\n"
                "• stop"
            )

    def _start_watcher(self) -> str:
        pid = self._running_pid()
        if pid is not None:
            return f"CricketBaba is already watching for live matches (process {pid})."
        with LOG_PATH.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, str(BACKEND_DIR / "run_toi_auto.py")],
                cwd=BACKEND_DIR,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        PID_PATH.write_text(str(process.pid), encoding="utf-8")
        return (
            "CricketBaba is now watching TOI. I will send verified "
            "predictions automatically."
        )

    def _status(self) -> str:
        pid = self._running_pid()
        if pid is None:
            return "CricketBaba is ready, but the live-match watcher is stopped."
        return f"CricketBaba is watching TOI for live matches (process {pid})."

    def _prematch_status(self) -> str:
        self._start_watcher()
        matches = ToiMatchDiscovery().matches()
        candidates = [match for match in matches if match.is_live or match.is_upcoming]
        if not candidates:
            return (
                "No upcoming or live match is currently listed by TOI. "
                "Telegram and Candidate v3 are ready; automatic watching is armed."
            )
        lines = ["🏏 Pre-match status"]
        for match in candidates[:5]:
            state = "LIVE" if match.is_live else "Upcoming"
            lines.extend(
                [
                    "",
                    f"{match.label} — {state}",
                    f"Format: {match.match_format}",
                    f"Series: {match.series}",
                    f"Venue: {match.venue}",
                    f"Start: {match.start_time}",
                    f"TOI status: {match.status}",
                    f"Weather: {match.weather}",
                    f"Temperature: {match.temperature}",
                    f"Humidity: {match.humidity}",
                    f"Wind: {match.wind}",
                    f"Rain: {match.rain}",
                    f"Pitch (TOI): {match.pitch}",
                    f"Likely track (inference): {match.likely_track}",
                    f"Dew outlook (inference): {match.dew_outlook}",
                ]
            )
        lines.extend(
            [
                "",
                "Feed: ready",
                "Candidate v3: ready",
                "Telegram: connected",
                "Automatic watcher: armed",
            ]
        )
        return "\n".join(lines)

    def _stop_watcher(self) -> str:
        pid = self._running_pid()
        if pid is None:
            return "The live-match watcher is already stopped."
        os.kill(pid, signal.SIGTERM)
        PID_PATH.unlink(missing_ok=True)
        return "CricketBaba live-match watcher stopped."

    def _running_pid(self) -> int | None:
        try:
            pid = int(PID_PATH.read_text(encoding="utf-8"))
            os.kill(pid, 0)
            return pid
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            PID_PATH.unlink(missing_ok=True)
            return None

    def _send(self, text: str) -> None:
        response = self._session.post(
            f"{self._base}/sendMessage",
            json={"chat_id": self._chat_id, "text": text},
            timeout=10,
        )
        response.raise_for_status()

    @staticmethod
    def _read_offset() -> int:
        try:
            return int(OFFSET_PATH.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return 0

    @staticmethod
    def _write_offset(offset: int) -> None:
        OFFSET_PATH.write_text(str(offset), encoding="utf-8")


if __name__ == "__main__":
    TelegramControl().run()
