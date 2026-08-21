"""Tests for chronological IPL venue scoring regimes."""

from __future__ import annotations

import json

from app.ml.ipl_venue_regime_dataset import (
    build_ipl_venue_regime_dataset,
    venue_regime,
)
from app.ml.ipl_venues import normalize_ipl_venue, team_venue_context


def _match(path, date: str, total: int) -> None:
    payload = {
        "info": {"dates": [date], "venue": "Ground"},
        "innings": [{
            "team": "A",
            "overs": [{
                "over": 0,
                "deliveries": [{"runs": {"total": total}}],
            }],
        }],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_regime_boundaries_cover_every_score() -> None:
    assert venue_regime(210, 5) == "high"
    assert venue_regime(200, 5) == "high"
    assert venue_regime(180, 5) == "high"
    assert venue_regime(179.9, 5) == "medium"
    assert venue_regime(150, 5) == "medium"
    assert venue_regime(149.9, 5) == "low"
    assert venue_regime(220, 4) == "high"


def test_current_match_never_enters_its_own_venue_history(tmp_path) -> None:
    directory = tmp_path / "data/raw/cricsheet/ipl"
    directory.mkdir(parents=True)
    for index, total in enumerate((200, 200, 200, 200, 200, 300), start=1):
        _match(directory / f"{index}.json", f"202{index}-01-01", total)

    frame = build_ipl_venue_regime_dataset(tmp_path)

    sixth = frame[frame["source_file"] == "6.json"].iloc[0]
    assert sixth["venue_prior_innings"] == 5
    assert sixth["venue_par_score"] == 200
    assert sixth["venue_scoring_regime"] == "high"
    assert set(frame["venue_scoring_regime"]) <= {"low", "medium", "high"}
    assert {
        "phase_venue_regime",
        "momentum_score",
        "current_pair_strike_rate",
        "opening_batters_prior_average",
    } <= set(frame.columns)
    assert frame["momentum_score"].between(0, 100).all()


def test_known_ipl_venue_aliases_share_one_identity() -> None:
    assert normalize_ipl_venue("Feroz Shah Kotla") == normalize_ipl_venue(
        "Arun Jaitley Stadium, Delhi"
    )
    assert normalize_ipl_venue(
        "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur"
    ) == normalize_ipl_venue(
        "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh"
    )
    assert team_venue_context("Delhi Capitals", "Arun Jaitley Stadium") == "home"
    assert team_venue_context("Delhi Capitals", "Wankhede Stadium") == "away"
