"""Safe CricketData.org adapter for match discovery and aggregate scores."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import requests


class CricketDataError(RuntimeError):
    """Raised when CricketData cannot provide a valid response."""


class CricketDataCapabilityError(CricketDataError):
    """Raised when data is insufficient for the requested operation."""


@dataclass(frozen=True)
class CricketDataUsage:
    hits_today: int | None
    hits_used: int | None
    hits_limit: int | None


@dataclass(frozen=True)
class CricketDataInnings:
    runs: int
    wickets: int
    overs: float
    label: str


@dataclass(frozen=True)
class CricketDataMatch:
    match_id: str
    name: str
    match_type: str
    status: str
    started: bool
    ended: bool
    innings: tuple[CricketDataInnings, ...]
    usage: CricketDataUsage

    @property
    def has_ball_by_ball(self) -> bool:
        """These low-cost endpoints expose totals, not verified deliveries."""
        return False

    def require_ball_by_ball(self) -> None:
        if not self.has_ball_by_ball:
            raise CricketDataCapabilityError(
                "CricketData currentMatches/match_info has no verified ball-by-ball "
                "deliveries and cannot feed Candidate v3."
            )


class CricketDataReader:
    BASE_URL = "https://api.cricapi.com/v1"

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        session: requests.Session | None = None,
        timeout: float = 10.0,
    ) -> None:
        backend_dir = Path(__file__).resolve().parents[2]
        self.config_path = Path(config_path) if config_path else backend_dir / ".cricketdata.json"
        self.session = session or requests.Session()
        self.timeout = timeout

    def current_matches(self, offset: int = 0) -> tuple[CricketDataMatch, ...]:
        payload = self._request("currentMatches", offset=offset)
        usage = self._usage(payload)
        return tuple(self._parse_match(item, usage) for item in payload.get("data", []))

    def match_info(self, match_id: str) -> CricketDataMatch:
        if not match_id.strip():
            raise ValueError("match_id must not be empty")
        payload = self._request("match_info", id=match_id)
        return self._parse_match(payload.get("data") or {}, self._usage(payload))

    def find_match(self, first_team: str, second_team: str) -> CricketDataMatch | None:
        needles = (first_team.casefold(), second_team.casefold())
        for match in self.current_matches():
            name = match.name.casefold()
            if all(team in name for team in needles):
                return match
        return None

    def _api_key(self) -> str:
        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CricketDataError(
                f"Unable to read CricketData credentials from {self.config_path}"
            ) from exc
        key = str(config.get("api_key", "")).strip()
        if not key:
            raise CricketDataError("CricketData api_key is missing")
        return key

    def _request(self, endpoint: str, **params: Any) -> dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.BASE_URL}/{endpoint}",
                params={"apikey": self._api_key(), **params},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise CricketDataError(f"CricketData {endpoint} request failed") from exc

        if not isinstance(payload, dict):
            raise CricketDataError(f"CricketData {endpoint} returned an invalid payload")
        if payload.get("status") != "success":
            reason = payload.get("reason") or payload.get("message") or "unknown provider error"
            raise CricketDataError(f"CricketData {endpoint} failed: {reason}")
        return payload

    @staticmethod
    def _usage(payload: dict[str, Any]) -> CricketDataUsage:
        info = payload.get("info") or {}
        return CricketDataUsage(
            hits_today=info.get("hitsToday"),
            hits_used=info.get("hitsUsed"),
            hits_limit=info.get("hitsLimit"),
        )

    @staticmethod
    def _parse_match(item: dict[str, Any], usage: CricketDataUsage) -> CricketDataMatch:
        if not item.get("id"):
            raise CricketDataError("CricketData match payload is missing id")
        innings = tuple(
            CricketDataInnings(
                runs=int(score.get("r", 0)),
                wickets=int(score.get("w", 0)),
                overs=float(score.get("o", 0)),
                label=str(score.get("inning", "")),
            )
            for score in (item.get("score") or [])
        )
        return CricketDataMatch(
            match_id=str(item["id"]),
            name=str(item.get("name", "")),
            match_type=str(item.get("matchType", "")),
            status=str(item.get("status", "")),
            started=bool(item.get("matchStarted", False)),
            ended=bool(item.get("matchEnded", False)),
            innings=innings,
            usage=usage,
        )
