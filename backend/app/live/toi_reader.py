"""Read and normalize Times of India live ball-by-ball cricket data.

TOI's match page embeds a ``window.App`` JSON object.  The object contains the
match identifiers, scorecard, and a live commentary array.  The separate
``toicri`` JSON is used as the authoritative scorecard.  No CSS selectors are
used, so presentation changes cannot silently alter cricket state.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import requests


class ToiFeedError(RuntimeError):
    """Raised when a TOI response cannot be trusted or normalized."""


@dataclass(frozen=True, slots=True)
class ToiDelivery:
    match_id: str
    innings: int
    over: int
    ball: int
    bowler: str
    striker: str
    total_runs: int
    batter_runs: int
    extras: dict[str, int]
    wicket_kind: str | None
    feed_total: int
    feed_wickets: int
    timestamp_ms: int
    commentary: str

    @property
    def key(self) -> tuple[int, int, int]:
        return self.innings, self.over, self.ball

    @property
    def is_legal(self) -> bool:
        return "wides" not in self.extras and "noballs" not in self.extras


@dataclass(frozen=True, slots=True)
class ToiSnapshot:
    match_id: str
    match_format: str
    batting_team: str
    bowling_team: str
    innings: int
    score: int
    wickets: int
    overs: str
    is_live: bool
    deliveries: tuple[ToiDelivery, ...]
    target: int = 0
    scheduled_overs: int = 0
    innings_complete: bool = False
    match_complete: bool = False
    is_super_over: bool = False
    competition: str = ""
    announced_bowler: str = ""
    announced_bowler_over: int = 0
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ToiLiveReader:
    """Poll TOI and return a deterministic, chronological delivery snapshot."""

    RAW_URL = "https://toicri.timesofindia.indiatimes.com/jsons/{match_id}.json"
    SCORECARD_URL = (
        "https://timesofindia.indiatimes.com/feed_cricket_scorecard.cms"
        "?matchid={match_id}&feedtype=sjson&upcache=2"
    )
    _APP_PREFIX = "window.App="
    _COMMENTARY_RE = re.compile(
        r"^(?P<over>\d+)\.(?P<ball>\d+):\s*"
        r"(?P<bowler>.+?)\s+to\s+(?P<striker>.+?),\s*(?P<result>.*)$",
        re.IGNORECASE,
    )

    def __init__(
        self, timeout_seconds: float = 12.0, session: requests.Session | None = None
    ) -> None:
        self._timeout = timeout_seconds
        self._session = session or requests.Session()
        self._session.headers.update(
            {"User-Agent": "CricketBaba/3.0 (+verified live match reader)"}
        )

    @staticmethod
    def match_id_from_url(url_or_id: str) -> str:
        value = url_or_id.strip()
        if not value:
            raise ToiFeedError("A TOI match URL or match id is required.")
        if "://" not in value:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
                raise ToiFeedError("Invalid TOI match id.")
            return value
        parsed = urlparse(value)
        if not parsed.hostname or not parsed.hostname.endswith(
            "timesofindia.indiatimes.com"
        ):
            raise ToiFeedError(
                "The live URL must be hosted by timesofindia.indiatimes.com."
            )
        match_id = parsed.path.rstrip("/").split("/")[-1]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", match_id):
            raise ToiFeedError("Could not determine the TOI match id from the URL.")
        return match_id

    def fetch(self, url_or_id: str) -> ToiSnapshot:
        match_id = self.match_id_from_url(url_or_id)
        try:
            raw = self._get_json(self.RAW_URL.format(match_id=match_id))
            scorecard = self._get_json(self.SCORECARD_URL.format(match_id=match_id))
            commentary = self._fetch_commentary(match_id, url_or_id)
        except requests.RequestException as exc:
            raise ToiFeedError(
                "TOI connection failed temporarily; no snapshot was accepted."
            ) from exc
        return self.parse(match_id, raw, scorecard, commentary)

    def _get_json(self, url: str) -> dict[str, Any]:
        response = self._session.get(url, timeout=self._timeout)
        response.raise_for_status()
        try:
            payload = response.json()
        except requests.JSONDecodeError as exc:
            raise ToiFeedError(f"TOI returned non-JSON data from {url}.") from exc
        if not isinstance(payload, dict):
            raise ToiFeedError(f"TOI returned an invalid object from {url}.")
        return payload

    def _fetch_commentary(self, match_id: str, url_or_id: str) -> list[dict[str, Any]]:
        if "://" not in url_or_id:
            raise ToiFeedError(
                "A full TOI match-center URL is required to read "
                "ball-by-ball commentary."
            )
        response = self._session.get(url_or_id, timeout=self._timeout)
        response.raise_for_status()
        marker = self._APP_PREFIX
        start = response.text.find(marker)
        if start < 0:
            raise ToiFeedError("TOI page did not contain its live application payload.")
        start += len(marker)
        decoder = json.JSONDecoder()
        try:
            app, _ = decoder.raw_decode(response.text[start:])
            items = app["state"]["newslisting"]["data"]["sectionitems"]
            item = next(entry for entry in items if entry.get("matchID") == match_id)
            rows = item["matchCenterScorcard"]["matchLiveBlogData"]
        except (KeyError, TypeError, StopIteration, json.JSONDecodeError) as exc:
            raise ToiFeedError(
                "TOI live commentary payload has an unknown schema."
            ) from exc
        if not isinstance(rows, list):
            raise ToiFeedError("TOI live commentary was not a delivery list.")
        return [row for row in rows if isinstance(row, dict)]

    @classmethod
    def parse(
        cls,
        match_id: str,
        raw: dict[str, Any],
        scorecard: dict[str, Any],
        commentary: list[dict[str, Any]],
    ) -> ToiSnapshot:
        try:
            raw_match = raw["Matchdetail"]["Match"]
            innings_rows = scorecard["innings"]
            current_name = scorecard["currentInning"]
            innings = {"First": 1, "Second": 2, "Third": 3, "Fourth": 4}[current_name]
            current = innings_rows[innings - 1]
            raw_innings = raw["Innings"][innings - 1]
            batting_team = current["battingTeam"]
            bowling_team = current["bowlingTeam"]
            score = int(raw_innings["Total"])
            wickets = int(raw_innings["Wickets"])
            overs = str(raw_innings["Overs"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ToiFeedError("TOI scorecard payload has an unknown schema.") from exc

        # The commentary widget is capped, while scorecard.ballByBall contains
        # the complete innings.  Start with the full scorecard and overlay the
        # richer commentary records (batter names and timestamps) where present.
        parsed = cls._scorecard_deliveries(match_id, innings, current)
        innings_started_ms = 0
        if innings >= 2:
            try:
                innings_started_ms = int(
                    datetime.fromisoformat(
                        re.sub(
                            r"([+-])(\d):",
                            r"\g<1>0\g<2>:",
                            str(raw_innings["StartTimeStamp"]).replace("Z", "+00:00"),
                        )
                    ).timestamp()
                    * 1000
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ToiFeedError(
                    "TOI did not provide a valid current-innings start time."
                ) from exc
        commentary_deliveries = [
            delivery
            for row in commentary
            if (delivery := cls._parse_delivery(match_id, innings, row)) is not None
            if delivery.timestamp_ms >= innings_started_ms
        ]
        by_over: dict[int, list[ToiDelivery]] = {}
        for delivery in commentary_deliveries:
            by_over.setdefault(delivery.over, []).append(delivery)
        for over_number, live_rows in by_over.items():
            # Illegal balls repeat the visible label (for example two 5.1s).
            # Map by timestamp order and cumulative match state instead.  The
            # commentary stream can lag behind the scorecard, so aligning it
            # with the final N scorecard balls would shift every live row
            # forward while an over is still in progress.
            # A reviewed delivery may receive a later edit timestamp than the
            # following ball.  The visible ball label is chronological; use
            # timestamps only to order repeated labels from illegal balls.
            live_rows.sort(key=lambda value: (value.ball, value.timestamp_ms))
            score_rows = [
                value for value in parsed.values() if value.over == over_number
            ]
            score_rows.sort(key=lambda value: value.ball)
            if len(live_rows) > len(score_rows):
                raise ToiFeedError(
                    f"TOI commentary has extra balls in over {over_number}."
                )
            score_index = 0
            for delivery in live_rows:
                matching_index = next(
                    (
                        index
                        for index in range(score_index, len(score_rows))
                        if score_rows[index].total_runs == delivery.total_runs
                        and bool(score_rows[index].wicket_kind)
                        == bool(delivery.wicket_kind)
                        and score_rows[index].feed_total == delivery.feed_total
                        and score_rows[index].feed_wickets == delivery.feed_wickets
                    ),
                    None,
                )
                if matching_index is None:
                    raise ToiFeedError(
                        f"TOI commentary conflicts with scorecard in over "
                        f"{over_number}: no scorecard delivery matches commentary "
                        f"state {delivery.feed_total}/{delivery.feed_wickets}."
                    )
                score_delivery = score_rows[matching_index]
                score_index = matching_index + 1
                delivery = replace(
                    delivery,
                    over=score_delivery.over,
                    ball=score_delivery.ball,
                )
                existing = parsed.get(delivery.key)
                if existing is not None and (
                    existing.total_runs != delivery.total_runs
                    or bool(existing.wicket_kind) != bool(delivery.wicket_kind)
                ):
                    raise ToiFeedError(
                        f"TOI commentary conflicts with scorecard at "
                        f"{delivery.over}.{delivery.ball}: scorecard has "
                        f"{existing.total_runs} run(s)/"
                        f"wicket={bool(existing.wicket_kind)}, "
                        f"commentary has {delivery.total_runs} run(s)/"
                        f"wicket={bool(delivery.wicket_kind)}."
                    )
                parsed[delivery.key] = delivery
        deliveries = tuple(parsed[key] for key in sorted(parsed))
        if deliveries and (
            deliveries[-1].feed_total != score or deliveries[-1].feed_wickets != wickets
        ):
            raise ToiFeedError(
                "TOI ball-by-ball data does not reconcile with its scorecard."
            )
        is_super_over = innings > 2
        target = 0
        if innings >= 2:
            try:
                target_innings = innings - 2 if is_super_over else 0
                target = int(raw["Innings"][target_innings]["Total"]) + 1
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise ToiFeedError(
                    "TOI did not provide a valid chase target."
                ) from exc
        scheduled_overs = cls._scheduled_overs(raw_match, current, is_super_over)
        announced_bowler, announced_bowler_over = cls._announced_bowler(
            current, deliveries
        )
        innings_complete = (
            wickets >= 10
            or (target > 0 and score >= target)
            or cls._legal_delivery_count(deliveries) >= scheduled_overs * 6
            or not bool(raw_match.get("Live", False))
        )
        return ToiSnapshot(
            match_id=match_id,
            match_format=str(raw_match.get("Type", "")),
            batting_team=str(batting_team),
            bowling_team=str(bowling_team),
            innings=innings,
            score=score,
            wickets=wickets,
            overs=overs,
            is_live=bool(raw_match.get("Live", False)),
            deliveries=deliveries,
            target=target,
            scheduled_overs=scheduled_overs,
            innings_complete=innings_complete,
            match_complete=not bool(raw_match.get("Live", False)),
            is_super_over=is_super_over,
            competition=cls._competition_name(raw_match),
            announced_bowler=announced_bowler,
            announced_bowler_over=announced_bowler_over,
        )

    @classmethod
    def _announced_bowler(
        cls,
        current: dict[str, Any],
        deliveries: tuple[ToiDelivery, ...],
    ) -> tuple[str, int]:
        """Return only a scorecard bowler announced before the over starts.

        A bowler attached to an over containing a delivery is not an advance
        announcement: it may have arrived with the first ball.  Restricting
        this signal to an explicitly numbered, empty next-over row prevents
        first-delivery leakage from commentary or the scorecard.
        """
        next_over = cls._legal_delivery_count(deliveries) // 6 + 1
        for over in current.get("ballByBall", []):
            if not isinstance(over, dict) or over.get("balls"):
                continue
            try:
                over_number = int(over["number"])
            except (KeyError, TypeError, ValueError):
                continue
            bowler = str(over.get("bowler", "")).strip()
            if over_number == next_over and bowler:
                return bowler, over_number
        return "", 0

    @classmethod
    def _parse_delivery(
        cls, match_id: str, innings: int, row: dict[str, Any]
    ) -> ToiDelivery | None:
        text = str(row.get("smalldesc", "")).strip()
        match = cls._COMMENTARY_RE.match(text)
        if not match or row.get("type") != "cricket":
            return None
        result = match.group("result")
        extras = cls._extras(result)
        total_runs = cls._integer(row, "runs")
        batter_runs = max(0, total_runs - sum(extras.values()))
        wicket_text = str(row.get("wicket", "")).strip()
        wicket_kind = cls._wicket_kind(result, wicket_text)
        return ToiDelivery(
            match_id=match_id,
            innings=innings,
            # Commentary labels the first over 0.x; scorecard labels it 1.x.
            over=int(match.group("over")) + 1,
            ball=int(match.group("ball")),
            bowler=match.group("bowler").strip(),
            striker=match.group("striker").strip(),
            total_runs=total_runs,
            batter_runs=batter_runs,
            extras=extras,
            wicket_kind=wicket_kind,
            feed_total=cls._integer(row, "totalRuns"),
            feed_wickets=cls._integer(row, "totalWickets"),
            timestamp_ms=cls._integer(row, "timestamp"),
            commentary=text,
        )

    @staticmethod
    def _integer(row: dict[str, Any], field: str) -> int:
        try:
            return int(row[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ToiFeedError(
                f"TOI delivery has invalid {field!r}: {row.get(field)!r}."
            ) from exc

    @staticmethod
    def _extras(result: str) -> dict[str, int]:
        lower = result.lower()
        patterns = {
            "wides": r"(?:(\d+)\s*)?wide",
            "noballs": r"(?:(\d+)\s*)?no[ -]?ball",
            "legbyes": r"(?:(\d+)\s*)?leg[ -]?bye",
            "byes": r"(?<!leg )(?:(\d+)\s*)?bye",
        }
        extras: dict[str, int] = {}
        for kind, pattern in patterns.items():
            found = re.search(pattern, lower)
            if found:
                extras[kind] = int(found.group(1) or 1)
        return extras

    @classmethod
    def _scorecard_deliveries(
        cls, match_id: str, innings: int, current: dict[str, Any]
    ) -> dict[tuple[int, int, int], ToiDelivery]:
        visible_rows: list[tuple[int, int, int, str, dict[str, Any]]] = []
        for over in current.get("ballByBall", []):
            if not isinstance(over, dict):
                continue
            bowler = str(over.get("bowler", "")).strip()
            for source_index, ball in enumerate(over.get("balls", []), start=1):
                if not isinstance(ball, dict):
                    continue
                try:
                    over_number = int(ball["over"])
                    visible_ball = int(ball["number"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ToiFeedError(
                        "TOI scorecard has an invalid delivery number."
                    ) from exc
                visible_rows.append(
                    (over_number, visible_ball, source_index, bowler, ball)
                )
        visible_rows.sort(key=lambda value: (value[0], value[1], value[2]))
        rows: list[tuple[int, int, str, dict[str, Any]]] = []
        sequence_by_over: dict[int, int] = {}
        for over_number, _, _, bowler, ball in visible_rows:
            # TOI repeats the visible label for wides/no-balls.  After ordering
            # by that label, assign every event a unique sequence identity.
            sequence_by_over[over_number] = sequence_by_over.get(over_number, 0) + 1
            rows.append((over_number, sequence_by_over[over_number], bowler, ball))
        parsed: dict[tuple[int, int, int], ToiDelivery] = {}
        total = wickets = 0
        for over_number, ball_number, bowler, ball in rows:
            details = str(ball.get("details", "")).strip()
            extras = cls._scorecard_extras(details)
            try:
                batter_runs = int(ball.get("runs", 0))
            except (TypeError, ValueError) as exc:
                raise ToiFeedError("TOI scorecard has invalid delivery runs.") from exc
            delivery_runs = batter_runs + sum(extras.values())
            is_wicket = str(ball.get("wicket", "")).lower() == "yes"
            total += delivery_runs
            wickets += int(is_wicket)
            delivery = ToiDelivery(
                match_id=match_id,
                innings=innings,
                over=over_number,
                ball=ball_number,
                bowler=bowler,
                striker="",
                total_runs=delivery_runs,
                batter_runs=batter_runs,
                extras=extras,
                wicket_kind=cls._wicket_kind(details, "wicket" if is_wicket else ""),
                feed_total=total,
                feed_wickets=wickets,
                timestamp_ms=0,
                commentary=details,
            )
            if delivery.key in parsed:
                raise ToiFeedError(
                    f"TOI scorecard repeats delivery {over_number}.{ball_number}."
                )
            parsed[delivery.key] = delivery
        return parsed

    @staticmethod
    def _legal_delivery_count(deliveries: tuple[ToiDelivery, ...]) -> int:
        return sum(delivery.is_legal for delivery in deliveries)

    @staticmethod
    def _scheduled_overs(
        raw_match: dict[str, Any], current: dict[str, Any], is_super_over: bool
    ) -> int:
        if is_super_over:
            return 1
        for source in (current, raw_match):
            for key in ("overs", "Overs", "maxOvers", "MaxOvers"):
                try:
                    value = int(source[key])
                except (KeyError, TypeError, ValueError):
                    continue
                if value > 0:
                    return value
        return 50 if str(raw_match.get("Type", "")).upper() == "ODI" else 20

    @staticmethod
    def _competition_name(raw_match: dict[str, Any]) -> str:
        for key in (
            "Competition",
            "competition",
            "League",
            "league",
            "Series",
            "series",
            "Tournament",
            "tournament",
        ):
            value = str(raw_match.get(key, "")).strip()
            if value:
                return value
        return ""

    @staticmethod
    def _scorecard_extras(details: str) -> dict[str, int]:
        compact = details.upper().replace(" ", "")
        labels = (("legbyes", "LB"), ("noballs", "NB"), ("wides", "WD"), ("byes", "B"))
        for kind, suffix in labels:
            match = re.fullmatch(rf"(\d*){suffix}", compact)
            if match:
                return {kind: int(match.group(1) or 1)}
        return {}

    @staticmethod
    def _wicket_kind(result: str, wicket: str) -> str | None:
        # TOI appends prose to the delivery outcome.  Searching the full prose
        # misclassifies phrases such as "deep mid-wicket" as dismissals.
        outcome = re.split(r"[.!]", result, maxsplit=1)[0]
        if not wicket and not re.search(r"\b(out|wicket)\b", outcome, re.IGNORECASE):
            return None
        lower = f"{wicket} {outcome}".lower()
        for kind in (
            "run out",
            "retired hurt",
            "stumped",
            "lbw",
            "bowled",
            "caught",
            "hit wicket",
            "obstructing the field",
        ):
            if kind in lower:
                return kind
        return "wicket"
