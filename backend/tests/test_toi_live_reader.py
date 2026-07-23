"""Complete delivery-path tests for the TOI live reader and verifier."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.live.toi_reader import ToiDelivery, ToiLiveReader, ToiSnapshot
from app.live.verification import LiveDeliveryVerifier, VerificationError


def _delivery(
    ball: int,
    runs: int,
    total: int,
    wickets: int = 0,
    wicket_kind: str | None = None,
    extras: dict[str, int] | None = None,
) -> ToiDelivery:
    return ToiDelivery(
        match_id="match",
        innings=1,
        over=1,
        ball=ball,
        bowler="Bowler",
        striker="Batter",
        total_runs=runs,
        batter_runs=max(0, runs - sum((extras or {}).values())),
        extras=extras or {},
        wicket_kind=wicket_kind,
        feed_total=total,
        feed_wickets=wickets,
        timestamp_ms=ball,
        commentary="",
    )


def test_parser_handles_illegal_ball_with_repeated_commentary_label() -> None:
    scorecard = {
        "currentInning": "First",
        "innings": [
            {
                "battingTeam": "A",
                "bowlingTeam": "B",
                "ballByBall": [
                    {
                        "bowler": "Bowler",
                        "number": "1",
                        "balls": [
                            {
                                "wicket": "No",
                                "details": "",
                                "runs": "0",
                                "over": "1",
                                "number": "2",
                            },
                            {
                                "wicket": "No",
                                "details": "1WD",
                                "runs": "0",
                                "over": "1",
                                "number": "1",
                            },
                        ],
                    }
                ],
            }
        ],
    }
    raw = {
        "Matchdetail": {"Match": {"Type": "T20", "Live": True}},
        "Innings": [{"Total": "1", "Wickets": "0", "Overs": "0.1"}],
    }
    commentary = [
        {
            "type": "cricket",
            "overs": "0",
            "balls": "1",
            "runs": "0",
            "totalRuns": "1",
            "totalWickets": "0",
            "timestamp": 2,
            "wicket": "",
            "smalldesc": "0.1: Bowler to Batter, No run.No run.",
        },
        {
            "type": "cricket",
            "overs": "0",
            "balls": "1",
            "runs": "1",
            "totalRuns": "1",
            "totalWickets": "0",
            "timestamp": 1,
            "wicket": "",
            "smalldesc": "0.1: Bowler to Batter, Wide.Wide.",
        },
    ]

    snapshot = ToiLiveReader.parse("match", raw, scorecard, commentary)

    assert [delivery.ball for delivery in snapshot.deliveries] == [1, 2]
    assert snapshot.deliveries[0].extras == {"wides": 1}
    assert snapshot.deliveries[1].total_runs == 0


def test_parser_accepts_commentary_one_ball_behind_scorecard() -> None:
    scorecard = {
        "currentInning": "First",
        "innings": [
            {
                "battingTeam": "A",
                "bowlingTeam": "B",
                "ballByBall": [
                    {
                        "bowler": "Bowler",
                        "number": "1",
                        "balls": [
                            {
                                "wicket": "No",
                                "details": "",
                                "runs": "1",
                                "over": "1",
                                "number": "1",
                            },
                            {
                                "wicket": "No",
                                "details": "",
                                "runs": "6",
                                "over": "1",
                                "number": "2",
                            },
                        ],
                    }
                ],
            }
        ],
    }
    raw = {
        "Matchdetail": {"Match": {"Type": "T20", "Live": True}},
        "Innings": [{"Total": "7", "Wickets": "0", "Overs": "0.2"}],
    }
    commentary = [
        {
            "type": "cricket",
            "overs": "0",
            "balls": "1",
            "runs": "1",
            "totalRuns": "1",
            "totalWickets": "0",
            "timestamp": 1,
            "wicket": "",
            "smalldesc": "0.1: Bowler to Batter, 1 run.",
        }
    ]

    snapshot = ToiLiveReader.parse("match", raw, scorecard, commentary)

    assert [(delivery.ball, delivery.total_runs) for delivery in snapshot.deliveries] == [
        (1, 1),
        (2, 6),
    ]


def test_commentary_prose_with_mid_wicket_is_not_a_dismissal() -> None:
    delivery = ToiLiveReader._parse_delivery(
        "match",
        1,
        {
            "type": "cricket",
            "overs": "0",
            "balls": "2",
            "runs": "6",
            "totalRuns": "7",
            "totalWickets": "0",
            "timestamp": 2,
            "wicket": "",
            "smalldesc": (
                "0.2: Bowler to Batter, Six! Swung over deep mid-wicket "
                "for the first six."
            ),
        },
    )

    assert delivery is not None
    assert delivery.wicket_kind is None


def test_complete_over_is_verified_ball_by_ball_and_deduplicated() -> None:
    deliveries = tuple(_delivery(ball, 1, ball) for ball in range(1, 7))
    snapshot = ToiSnapshot("match", "T20", "A", "B", 1, 6, 0, "1.0", True, deliveries)
    verifier = LiveDeliveryVerifier("match", 1)

    assert len(verifier.apply_snapshot(snapshot)) == 6
    assert verifier.completed_over == 1
    assert verifier.apply_snapshot(snapshot) == ()
    assert verifier.score == 6


def test_score_mismatch_and_silent_revision_fail_closed() -> None:
    verifier = LiveDeliveryVerifier("match", 1)
    first = _delivery(1, 1, 1)
    verifier.apply(first)
    with pytest.raises(VerificationError, match="Score mismatch"):
        verifier.apply(_delivery(2, 1, 99))

    revised = replace(first, total_runs=4, feed_total=4)
    with pytest.raises(VerificationError, match="changed between polls"):
        verifier.apply_snapshot(
            ToiSnapshot("match", "T20", "A", "B", 1, 4, 0, "0.1", True, (revised,))
        )


def test_commentary_enrichment_is_not_a_silent_revision() -> None:
    verifier = LiveDeliveryVerifier("match", 1)
    scorecard_delivery = _delivery(1, 1, 1)
    verifier.apply(scorecard_delivery)
    enriched = replace(
        scorecard_delivery,
        striker="Batter Name",
        timestamp_ms=123,
        commentary="A richer commentary description.",
    )

    assert verifier.apply(enriched) is False
    assert verifier.accepted[enriched.key].striker == "Batter Name"


def test_completed_over_remains_visible_after_next_over_starts_with_wide() -> None:
    verifier = LiveDeliveryVerifier("match", 1)
    for ball in range(1, 7):
        verifier.apply(_delivery(ball, 1, ball))
    wide = replace(
        _delivery(1, 1, 7, extras={"wides": 1}),
        over=2,
        batter_runs=0,
    )
    verifier.apply(wide)

    assert verifier.completed_over == 1
