"""Tests for prediction-safe historical replay loading."""

from __future__ import annotations

from pathlib import Path

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.models.match_data import Delivery, Innings, Match, MatchInfo, Over
from app.models.match_state import MatchFormat
from app.replay.replay_loader import ReplayLoader, ReplayRuleMode
from app.models.match_state import MatchStateError
import pytest


def test_loader_yields_pre_over_state_before_revealing_actuals() -> None:
    match = Match(
        info=MatchInfo(match_type="T20", teams=["Team A", "Team B"], venue="Test"),
        innings=[
            Innings(
                team="Team A",
                overs=[
                    Over(
                        over_number=0,
                        deliveries=[
                            Delivery(
                                "0.1",
                                "A Batter",
                                "Bowler 1",
                                "B Batter",
                                {"total": 1, "batter": 1},
                            ),
                            Delivery(
                                "0.2",
                                "B Batter",
                                "Bowler 1",
                                "A Batter",
                                {"total": 4, "batter": 4},
                            ),
                        ],
                    ),
                    Over(
                        over_number=1,
                        deliveries=[
                            Delivery(
                                "1.1",
                                "B Batter",
                                "Bowler 2",
                                "A Batter",
                                {"total": 0, "batter": 0},
                                wickets=[{"kind": "bowled"}],
                            )
                        ],
                    ),
                ],
            )
        ],
    )

    frames = list(ReplayLoader().frames(match, match_id="unit"))

    assert len(frames) == 2
    assert frames[0].match_id == "unit"
    assert frames[0].is_super_over is False
    assert frames[0].venue == "Test"
    assert frames[0].city is None
    assert frames[0].state.match_format is MatchFormat.T20I
    assert frames[0].state.score == 0
    assert frames[0].state.wickets == 0
    assert frames[0].state.current_bowler == "Bowler 1"
    assert frames[0].actual.runs == 5
    assert frames[0].actual.wickets == 0

    assert frames[1].state.over_number == 2
    assert frames[1].state.score == 5
    assert frames[1].state.wickets == 0
    assert frames[1].state.striker == "B Batter"
    assert frames[1].state.non_striker == "A Batter"
    assert frames[1].state.current_bowler == "Bowler 2"
    assert frames[1].actual.runs == 0
    assert frames[1].actual.wickets == 1


def test_loader_reads_a_real_parsed_match_file() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sample_file = next((project_root / "data/raw/cricsheet/ipl").glob("*.json"))
    match = MatchParser().parse(JsonLoader().load(sample_file))

    frames = list(ReplayLoader().frames(match, match_id=sample_file.stem))

    assert frames
    assert frames[0].match_id == sample_file.stem
    assert frames[0].state.batting_team == match.innings[0].team
    assert frames[0].state.score == 0
    assert frames[0].state.wickets == 0
    assert frames[0].actual.runs >= 0


def test_loader_attributes_a_split_over_to_each_bowler() -> None:
    """An injury replacement does not inflate the starting bowler's figures."""

    match = Match(
        info=MatchInfo(match_type="T20", teams=["A", "B"]),
        innings=[
            Innings(
                team="A",
                overs=[
                    Over(
                        0,
                        [
                            Delivery("0.1", "One", "Starter", "Two", {"total": 0}),
                            Delivery("0.2", "One", "Replacement", "Two", {"total": 1}),
                        ],
                    ),
                    Over(
                        1,
                        [Delivery("1.1", "Two", "Next", "One", {"total": 0})],
                    ),
                ],
            )
        ],
    )

    frames = list(ReplayLoader().frames(match, match_id="split"))

    figures = frames[1].state.bowler_figures
    assert figures["Starter"].legal_balls == 1
    assert figures["Replacement"].legal_balls == 1
    assert frames[1].state.previous_over_bowler == "Replacement"


def test_loader_resets_bowling_figures_for_a_super_over_innings() -> None:
    match = Match(
        info=MatchInfo(match_type="T20", teams=["A", "B"]),
        innings=[
            Innings(
                team="A",
                super_over=True,
                overs=[
                    Over(
                        0,
                        [Delivery("0.1", "One", "Bowler", "Two", {"total": 1})],
                    )
                ],
            )
        ],
    )

    frame = next(ReplayLoader().frames(match, match_id="tie"))

    assert frame.is_super_over is True
    assert frame.state.completed_overs == 0
    assert frame.state.bowler_figures["Bowler"].legal_balls == 0


def test_archival_mode_labels_a_recorded_fifth_over_without_weakening_strict_mode() -> None:
    overs = []
    for over_number in range(9):
        bowler = "Limited" if over_number % 2 == 0 else f"Other {over_number}"
        overs.append(
            Over(
                over_number,
                [
                    Delivery(
                        f"{over_number}.{ball}", "One", bowler, "Two", {"total": 0}
                    )
                    for ball in range(1, 7)
                ],
            )
        )
    match = Match(
        info=MatchInfo(match_type="T20", teams=["A", "B"]),
        innings=[Innings(team="A", overs=overs)],
    )

    with pytest.raises(MatchStateError, match="4-over limit"):
        list(ReplayLoader().frames(match))

    frames = list(
        ReplayLoader(rule_mode=ReplayRuleMode.ARCHIVAL).frames(match)
    )

    assert len(frames) == 9
    assert frames[8].rule_anomalies[0].code == "bowler_over_limit"
