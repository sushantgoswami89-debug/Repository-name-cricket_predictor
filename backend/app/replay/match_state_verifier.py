"""Verify replay frames against their authoritative parsed match data."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from app.models.match_data import Delivery, Match
from app.replay.replay_loader import ReplayOverFrame


@dataclass(frozen=True, slots=True)
class VerificationIssue:
    """One difference between a replay frame and its source match."""

    innings_index: int
    over_number: int
    field: str
    expected: Any
    actual: Any

    def __str__(self) -> str:
        return (
            f"innings {self.innings_index}, over {self.over_number}, {self.field}: "
            f"expected {self.expected!r}, got {self.actual!r}"
        )


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Summary of a complete replay verification."""

    match_id: str
    frames_checked: int
    issues: tuple[VerificationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def raise_for_errors(self) -> None:
        """Raise a single useful error when verification found differences."""

        if self.issues:
            raise MatchStateVerificationError(self)


class MatchStateVerificationError(ValueError):
    """Raised when replayed match state differs from the source match."""

    def __init__(self, result: VerificationResult) -> None:
        self.result = result
        preview = "; ".join(str(issue) for issue in result.issues[:3])
        remaining = len(result.issues) - 3
        if remaining > 0:
            preview += f"; and {remaining} more issue(s)"
        super().__init__(f"Replay verification failed for {result.match_id}: {preview}")


class MatchStateVerifier:
    """Independently compare every replay frame with cumulative source totals."""

    _NON_DISMISSAL_KINDS = frozenset({"retired hurt"})

    def verify(
        self,
        match: Match,
        frames: Iterable[ReplayOverFrame],
        match_id: str = "unknown",
    ) -> VerificationResult:
        """Return all state, result, ordering, and metadata discrepancies."""

        actual_frames = list(frames)
        expected = self._expected_frames(match)
        issues: list[VerificationIssue] = []

        for position in range(max(len(expected), len(actual_frames))):
            if position >= len(expected):
                frame = actual_frames[position]
                issues.append(
                    VerificationIssue(
                        frame.innings_index,
                        frame.over_number,
                        "frame",
                        "absent",
                        "unexpected frame",
                    )
                )
                continue
            expected_frame = expected[position]
            innings_index, source_over, values = expected_frame
            if position >= len(actual_frames):
                issues.append(
                    VerificationIssue(
                        innings_index,
                        source_over + 1,
                        "frame",
                        "present",
                        "missing",
                    )
                )
                continue

            frame = actual_frames[position]
            comparisons = {
                "match_id": (match_id, frame.match_id),
                "innings_index": (innings_index, frame.innings_index),
                "innings_team": (values["innings_team"], frame.innings_team),
                "is_super_over": (values["is_super_over"], frame.is_super_over),
                "venue": (match.info.venue, frame.venue),
                "city": (match.info.city, frame.city),
                "over_number": (source_over + 1, frame.over_number),
                "state.score": (values["score"], frame.state.score),
                "state.wickets": (values["wickets"], frame.state.wickets),
                "state.completed_overs": (
                    source_over,
                    frame.state.completed_overs,
                ),
                "state.batting_team": (
                    values["innings_team"],
                    frame.state.batting_team,
                ),
                "state.bowling_team": (
                    values["bowling_team"],
                    frame.state.bowling_team,
                ),
                "state.current_bowler": (values["bowler"], frame.state.current_bowler),
                "actual.runs": (values["over_runs"], frame.actual.runs),
                "actual.wickets": (values["over_wickets"], frame.actual.wickets),
                "actual.deliveries": (
                    values["deliveries"],
                    frame.actual.deliveries,
                ),
            }
            for field, (wanted, observed) in comparisons.items():
                if wanted != observed:
                    issues.append(
                        VerificationIssue(
                            innings_index,
                            source_over + 1,
                            field,
                            wanted,
                            observed,
                        )
                    )

        return VerificationResult(match_id, len(actual_frames), tuple(issues))

    def assert_valid(
        self,
        match: Match,
        frames: Iterable[ReplayOverFrame],
        match_id: str = "unknown",
    ) -> VerificationResult:
        """Verify frames, raising when any discrepancy is found."""

        result = self.verify(match, frames, match_id)
        result.raise_for_errors()
        return result

    def _expected_frames(
        self, match: Match
    ) -> list[tuple[int, int, dict[str, Any]]]:
        expected: list[tuple[int, int, dict[str, Any]]] = []
        for innings_index, innings in enumerate(match.innings):
            score = 0
            wickets = 0
            bowling_team = next(
                (team for team in match.info.teams if team != innings.team), ""
            )
            for over in sorted(innings.overs, key=lambda item: item.over_number):
                if not over.deliveries:
                    continue
                over_runs = sum(ball.runs.get("total", 0) for ball in over.deliveries)
                over_wickets = sum(
                    self._dismissals(ball) for ball in over.deliveries
                )
                expected.append(
                    (
                        innings_index,
                        over.over_number,
                        {
                            "innings_team": innings.team,
                            "is_super_over": innings.super_over,
                            "bowling_team": bowling_team,
                            "score": score,
                            "wickets": wickets,
                            "bowler": over.deliveries[0].bowler,
                            "over_runs": over_runs,
                            "over_wickets": over_wickets,
                            "deliveries": tuple(over.deliveries),
                        },
                    )
                )
                score += over_runs
                wickets += over_wickets
        return expected

    def _dismissals(self, delivery: Delivery) -> int:
        return sum(
            wicket.get("kind") not in self._NON_DISMISSAL_KINDS
            for wicket in delivery.wickets
        )
