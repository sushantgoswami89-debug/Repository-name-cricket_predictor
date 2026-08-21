"""Crash-safe persistence for the verified live prediction pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


class LiveStateStore:
    """Store a small, versioned live-state document atomically."""

    SCHEMA_VERSION = 2

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self, match_id: str, innings: int) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._quarantine()
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != self.SCHEMA_VERSION
        ):
            self._quarantine()
            return None
        scopes = payload.get("scopes")
        if not isinstance(scopes, dict):
            self._quarantine()
            return None
        state = scopes.get(self._scope_key(match_id, innings))
        return state if isinstance(state, dict) else None

    def save(self, payload: dict[str, Any]) -> None:
        match_id = str(payload["match_id"])
        innings = int(payload["innings"])
        scopes: dict[str, Any] = {}
        if self.path.exists():
            try:
                existing = json.loads(self.path.read_text(encoding="utf-8"))
                if (
                    isinstance(existing, dict)
                    and existing.get("schema_version") == self.SCHEMA_VERSION
                    and isinstance(existing.get("scopes"), dict)
                ):
                    scopes = existing["scopes"]
            except (OSError, json.JSONDecodeError):
                self._quarantine()
        scopes[self._scope_key(match_id, innings)] = payload
        document = {"schema_version": self.SCHEMA_VERSION, "scopes": scopes}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, self.path)

    @staticmethod
    def _scope_key(match_id: str, innings: int) -> str:
        return f"{match_id}:{innings}"

    def _quarantine(self) -> None:
        try:
            destination = self.path.with_suffix(self.path.suffix + ".corrupt")
            counter = 1
            while destination.exists():
                destination = self.path.with_suffix(
                    self.path.suffix + f".corrupt.{counter}"
                )
                counter += 1
            self.path.replace(destination)
        except OSError:
            pass

    def quarantine(self) -> None:
        """Move an incompatible document aside after semantic validation fails."""
        self._quarantine()
