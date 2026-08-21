"""Eligibility policy for matches used in prediction-model evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from app.models.match_data import Match


KNOWN_RULE_ANOMALY_EXCLUSIONS: dict[str, str] = {
    "1179017": "confirmed historical bowling-limit violation",
    "1267304": "unresolved five-over scorecard anomaly",
    "1283040": "consecutive-over bowler-attribution inconsistency",
    "1419932": "confirmed historical bowling-limit violation",
    "1490882": "ball-by-ball bowler-attribution error",
    "1499795": "final-over bowler-attribution error",
}


APPROVED_DOMESTIC_COMPETITIONS = frozenset({"Indian Premier League"})


@dataclass(frozen=True, slots=True)
class MatchEligibility:
    """Approved prediction scope assigned from authoritative metadata."""

    eligible: bool
    scope: str
    reason: str


class MatchExcludedError(ValueError):
    """Raised before prediction when a match is excluded by evaluation policy."""

    def __init__(self, match_id: str, reason: str, category: str) -> None:
        self.match_id = match_id
        self.reason = reason
        self.category = category
        super().__init__(f"Match {match_id} excluded from model evaluation: {reason}.")


def exclusion_reason(source_file: str) -> str | None:
    """Return a documented exclusion reason for a source filename or match ID."""

    match_id = source_file.removesuffix(".json")
    return KNOWN_RULE_ANOMALY_EXCLUSIONS.get(match_id)


def match_eligibility(match: Match) -> MatchEligibility:
    """Allow official internationals and explicitly approved domestic leagues."""

    info = match.info
    if info.match_type_number is not None:
        return MatchEligibility(
            True,
            "icc_recognized_international",
            f"official international match number {info.match_type_number}",
        )
    if info.event_name in APPROVED_DOMESTIC_COMPETITIONS:
        return MatchEligibility(
            True,
            "approved_domestic_ipl",
            "approved BCCI-sanctioned Indian Premier League match",
        )
    return MatchEligibility(
        False,
        "ineligible",
        "not an ICC-recognized international or approved IPL match",
    )
