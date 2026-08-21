"""Tests for generating live match state from completed match data."""

from __future__ import annotations

from pathlib import Path

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.models.match_data import Delivery, Innings, Over
from app.services.match_replay import MatchReplay


def test_replay_builds_prediction_ready_state_before_each_over() -> None:
    """Score, wickets, striker, and balls faced carry into the next over."""

    innings = Innings(
        team="Team A",
        overs=[
            Over(
                over_number=0,
                deliveries=[
                    Delivery("0.1", "A Batter", "Bowler 1", "B Batter", {"total": 1}),
                    Delivery("0.2", "B Batter", "Bowler 1", "A Batter", {"total": 4}),
                    Delivery(
                        "0.3",
                        "B Batter",
                        "Bowler 1",
                        "A Batter",
                        {"total": 1},
                        wickets=[{"kind": "bowled"}],
                    ),
                ],
            ),
            Over(
                over_number=1,
                deliveries=[
                    Delivery("1.1", "B Batter", "Bowler 2", "A Batter", {"total": 0}),
                    Delivery(
                        "1.2",
                        "B Batter",
                        "Bowler 2",
                        "A Batter",
                        {"total": 1},
                        extras={"wides": 1},
                    ),
                ],
            ),
        ],
    )

    frames = list(MatchReplay().frames(innings))

    assert len(frames) == 2
    assert frames[0].state.over == 1
    assert frames[0].state.score_before_over == 0
    assert frames[0].actual_runs == 6
    assert frames[0].actual_wickets == 1

    second_state = frames[1].state
    assert second_state.over == 2
    assert second_state.score_before_over == 6
    assert second_state.wkts_down_before_over == 1
    assert second_state.balls_faced_before_over == 2
    assert second_state.striker == "B Batter"
    assert second_state.non_striker == "A Batter"
    assert second_state.bowler == "Bowler 2"


def test_replay_uses_a_real_parsed_ipl_match() -> None:
    """A full IPL innings produces one valid frame for every recorded over."""

    project_root = Path(__file__).resolve().parents[2]
    sample_file = next((project_root / "data/raw/cricsheet/ipl").glob("*.json"))
    match = MatchParser().parse(JsonLoader().load(sample_file))

    frames = list(MatchReplay().frames(match.innings[0]))

    assert len(frames) == len(match.innings[0].overs)
    assert frames[0].state.score_before_over == 0
    assert frames[0].state.wkts_down_before_over == 0
    assert frames[0].state.striker == match.innings[0].overs[0].deliveries[0].batter
    assert sum(frame.actual_runs for frame in frames) > 0


def test_replay_supports_a_real_fifty_over_odi_match() -> None:
    """The replay engine handles an ODI innings that reaches over 50."""

    project_root = Path(__file__).resolve().parents[2]
    sample_file = project_root / "data/raw/cricsheet/odi/351689.json"
    match = MatchParser().parse(JsonLoader().load(sample_file))

    frames = list(MatchReplay().frames(match.innings[1]))

    assert match.info.match_type == "ODI"
    assert len(frames) == 50
    assert frames[-1].state.over == 50
    assert sum(frame.actual_runs for frame in frames) > 0
