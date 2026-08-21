"""Tests for parsing Cricsheet match data."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParseError, MatchParser
from app.data_ingestion.validator import MatchValidator


def test_parser_converts_a_real_cricsheet_match() -> None:
    """A real archived match becomes a valid internal Match object."""

    project_root = Path(__file__).resolve().parents[2]
    sample_file = next((project_root / "data/raw/cricsheet/ipl").glob("*.json"))

    match = MatchParser().parse(JsonLoader().load(sample_file))

    assert match.info.match_type == "T20"
    assert len(match.info.teams) == 2
    assert len(match.innings) == 2
    assert match.innings[0].overs[0].deliveries[0].runs["total"] >= 0
    assert MatchValidator().validate(match) is match


def test_parser_generates_a_ball_identifier_when_source_omits_it() -> None:
    """Raw Cricsheet data without CricketBaba's optional field still parses."""

    data = {
        "info": {"match_type": "T20", "teams": ["A", "B"]},
        "innings": [
            {
                "team": "A",
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "Batter",
                                "bowler": "Bowler",
                                "non_striker": "Partner",
                                "runs": {"batter": 1, "extras": 0, "total": 1},
                            }
                        ],
                    }
                ],
            }
        ],
    }

    delivery = MatchParser().parse(data).innings[0].overs[0].deliveries[0]

    assert delivery.ball == "0.1"
    assert delivery.extras == {}
    assert delivery.wickets == []


def test_parser_preserves_a_super_over_as_a_separate_innings() -> None:
    data = {
        "info": {"match_type": "T20", "teams": ["A", "B"]},
        "innings": [
            {
                "team": "A",
                "super_over": True,
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "Batter",
                                "bowler": "Bowler",
                                "non_striker": "Partner",
                                "runs": {"total": 1},
                            }
                        ],
                    }
                ],
            }
        ],
    }

    innings = MatchParser().parse(data).innings[0]

    assert innings.super_over is True


def test_parser_preserves_official_competition_metadata() -> None:
    data = {
        "info": {
            "match_type": "T20",
            "match_type_number": 123,
            "team_type": "international",
            "teams": ["A", "B"],
            "event": {"name": "Official Cup"},
        },
        "innings": [],
    }

    info = MatchParser().parse(data).info

    assert info.match_type_number == 123
    assert info.team_type == "international"
    assert info.event_name == "Official Cup"


def test_parser_ignores_the_non_boundary_run_flag() -> None:
    """Cricsheet's descriptive non-boundary flag is not treated as runs."""

    data = {
        "info": {"match_type": "T20", "teams": ["A", "B"]},
        "innings": [
            {
                "team": "A",
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "Batter",
                                "bowler": "Bowler",
                                "non_striker": "Partner",
                                "runs": {
                                    "batter": 4,
                                    "extras": 0,
                                    "total": 4,
                                    "non_boundary": True,
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }

    delivery = MatchParser().parse(data).innings[0].overs[0].deliveries[0]

    assert delivery.runs == {"batter": 4, "extras": 0, "total": 4}


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, "info must be an object"),
        ({"info": [], "innings": []}, "info must be an object"),
        ({"info": {"teams": "A"}, "innings": []}, "info.teams must be an array"),
    ],
)
def test_parser_rejects_invalid_structure(
    data: dict[str, object], message: str
) -> None:
    """Malformed input fails early with a useful location-aware message."""

    with pytest.raises(MatchParseError, match=message):
        MatchParser().parse(data)
