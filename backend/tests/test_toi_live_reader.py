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


def test_scoreboard_empty_next_over_exposes_announced_bowler() -> None:
    scorecard = {
        "currentInning": "First",
        "innings": [{
            "battingTeam": "A",
            "bowlingTeam": "B",
            "ballByBall": [
                {
                    "bowler": "Opening Bowler",
                    "number": "1",
                    "balls": [
                        {
                            "wicket": "No",
                            "details": "",
                            "runs": "0",
                            "over": "1",
                            "number": str(ball),
                        }
                        for ball in range(1, 7)
                    ],
                },
                {"bowler": "Change Bowler", "number": "2", "balls": []},
            ],
        }],
    }
    raw = {
        "Matchdetail": {"Match": {"Type": "T20", "Live": True}},
        "Innings": [{"Total": "0", "Wickets": "0", "Overs": "1.0"}],
    }

    snapshot = ToiLiveReader.parse("match", raw, scorecard, [])

    assert snapshot.announced_bowler == "Change Bowler"
    assert snapshot.announced_bowler_over == 2


def test_bowler_first_seen_with_delivery_is_not_advance_announcement() -> None:
    scorecard = {
        "currentInning": "First",
        "innings": [{
            "battingTeam": "A",
            "bowlingTeam": "B",
            "ballByBall": [{
                "bowler": "Opening Bowler",
                "number": "1",
                "balls": [{
                    "wicket": "No",
                    "details": "",
                    "runs": "0",
                    "over": "1",
                    "number": "1",
                }],
            }],
        }],
    }
    raw = {
        "Matchdetail": {"Match": {"Type": "T20", "Live": True}},
        "Innings": [{"Total": "0", "Wickets": "0", "Overs": "0.1"}],
    }

    snapshot = ToiLiveReader.parse("match", raw, scorecard, [])

    assert snapshot.announced_bowler == ""
    assert snapshot.announced_bowler_over == 0


def test_scorecard_handles_consecutive_illegals_with_repeated_labels() -> None:
    scorecard = {
        "currentInning": "First",
        "innings": [{
            "battingTeam": "A",
            "bowlingTeam": "B",
            "ballByBall": [{
                "bowler": "Bowler",
                "balls": [
                    {
                        "wicket": "No", "details": "1WD", "runs": "0",
                        "over": "1", "number": "1",
                    },
                    {
                        "wicket": "No", "details": "1NB", "runs": "2",
                        "over": "1", "number": "1",
                    },
                    {
                        "wicket": "No", "details": "1LB", "runs": "0",
                        "over": "1", "number": "1",
                    },
                    {
                        "wicket": "No", "details": "1B", "runs": "0",
                        "over": "1", "number": "2",
                    },
                ],
            }],
        }],
    }
    raw = {
        "Matchdetail": {"Match": {"Type": "T20", "Live": True}},
        "Innings": [{"Total": "6", "Wickets": "0", "Overs": "0.2"}],
    }

    snapshot = ToiLiveReader.parse("match", raw, scorecard, [])

    assert [delivery.ball for delivery in snapshot.deliveries] == [1, 2, 3, 4]
    assert [delivery.is_legal for delivery in snapshot.deliveries] == [
        False, False, True, True
    ]
    assert [
        (delivery.total_runs, delivery.batter_runs, sum(delivery.extras.values()))
        for delivery in snapshot.deliveries
    ] == [(1, 0, 1), (3, 2, 1), (1, 0, 1), (1, 0, 1)]


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

    assert [
        (delivery.ball, delivery.total_runs)
        for delivery in snapshot.deliveries
    ] == [(1, 1), (2, 6)]


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


def test_parser_preserves_competition_for_engine_routing() -> None:
    raw = {
        "Matchdetail": {
            "Match": {
                "Type": "T20",
                "Live": True,
                "Series": "Indian Premier League",
            }
        },
        "Innings": [{"Total": "0", "Wickets": "0", "Overs": "0.0"}],
    }
    scorecard = {
        "currentInning": "First",
        "innings": [{
            "battingTeam": "A",
            "bowlingTeam": "B",
            "ballByBall": [],
        }],
    }

    snapshot = ToiLiveReader.parse("match", raw, scorecard, [])

    assert snapshot.competition == "Indian Premier League"


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


def _raw_with_enrichment() -> dict:
    return {
        "Matchdetail": {
            "Match": {"Type": "T20", "Live": True},
            "Tosswonby": "1",
            "Toss_elected_to": "field",
            "Venue": {
                "Pitch_Detail": {
                    "Pitch_Suited_For": "Batting friendly",
                    "Pitch_Surface": "Dry",
                },
                "Venue_Weather": {
                    "Weather": "Clear",
                    "Humidity": "88%",
                    "Temperature": "15.99C",
                    "Wind_Speed": "2.06 meter/sec",
                },
            },
        },
        "Teams": {
            "1": {
                "Name_Full": "Australia",
                "Players": {
                    "1": {
                        "Position": "1",
                        "Name_Full": "Matt Renshaw",
                        "Role": "Batter",
                        "Confirm_XI": True,
                    },
                    "2": {
                        "Position": "8",
                        "Name_Full": "Pat Cummins",
                        "Role": "Bowler",
                        "Confirm_XI": True,
                    },
                },
            },
            "2": {"Name_Full": "Bangladesh", "Players": {}},
        },
    }


def test_team_players_extracts_and_sorts_by_position() -> None:
    players = ToiLiveReader._team_players(_raw_with_enrichment())

    assert [p.name for p in players["Australia"]] == ["Matt Renshaw", "Pat Cummins"]
    assert players["Australia"][1].role == "Bowler"
    assert players["Bangladesh"] == ()


def test_team_players_degrades_gracefully_when_missing() -> None:
    assert ToiLiveReader._team_players({}) == {}
    assert ToiLiveReader._team_players({"Teams": "not a dict"}) == {}


def test_toss_resolves_team_index_to_name() -> None:
    won_by, decision = ToiLiveReader._toss(_raw_with_enrichment())

    assert won_by == "Australia"
    assert decision == "field"


def test_toss_degrades_gracefully_when_missing() -> None:
    assert ToiLiveReader._toss({}) == ("", "")


def test_pitch_extraction() -> None:
    assert ToiLiveReader._pitch(_raw_with_enrichment()) == ("Batting friendly", "Dry")
    assert ToiLiveReader._pitch({}) == ("", "")


def test_weather_extraction_strips_units() -> None:
    condition, humidity, temperature, wind = ToiLiveReader._weather(
        _raw_with_enrichment()
    )

    assert condition == "Clear"
    assert humidity == 88.0
    assert temperature == 15.99
    assert wind == 2.06


def test_weather_degrades_gracefully_when_missing() -> None:
    assert ToiLiveReader._weather({}) == ("", None, None, None)
